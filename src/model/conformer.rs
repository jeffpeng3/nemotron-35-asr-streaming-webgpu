use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use crate::model::attention::RelPositionMultiHeadAttention;

#[derive(Module, Debug)]
pub struct FeedForwardModule<B: Backend> {
    pub ln: LayerNorm<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> FeedForwardModule<B> {
    pub fn new(hidden_size: usize, expansion_factor: usize, device: &B::Device) -> Self {
        let ln = LayerNormConfig::new(hidden_size).init(device);
        let linear1 = LinearConfig::new(hidden_size, hidden_size * expansion_factor).init(device);
        let linear2 = LinearConfig::new(hidden_size * expansion_factor, hidden_size).init(device);
        
        Self { ln, linear1, linear2 }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.ln.forward(x);
        let h = self.linear1.forward(h);
        let h = burn::tensor::activation::silu(h);
        self.linear2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct ConformerConvolution<B: Backend> {
    pub ln: LayerNorm<B>,
    pub pointwise_conv1: Conv1d<B>,
    pub depthwise_conv: Conv1d<B>,
    pub pointwise_conv2: Conv1d<B>,
    pub kernel_size: usize,
}

impl<B: Backend> ConformerConvolution<B> {
    pub fn new(hidden_size: usize, kernel_size: usize, device: &B::Device) -> Self {
        let ln = LayerNormConfig::new(hidden_size).init(device);
        
        // Pointwise Conv 1: [B, D, L] -> [B, 2*D, L] (NeMo 通常用 Linear 或 1x1 Conv，我們用 1x1 Conv)
        let pointwise_conv1 = Conv1dConfig::new(hidden_size, hidden_size * 2, 1).init(device);
        
        // Depthwise Conv: [B, D, L] -> [B, D, L] (使用 groups = D 實現深度卷積，kernel_size=31)
        let depthwise_conv = Conv1dConfig::new(hidden_size, hidden_size, kernel_size)
            .with_groups(hidden_size)
            .init(device);
            
        let pointwise_conv2 = Conv1dConfig::new(hidden_size, hidden_size, 1).init(device);

        Self {
            ln,
            pointwise_conv1,
            depthwise_conv,
            pointwise_conv2,
            kernel_size,
        }
    }

    /// 帶有狀態快取的因果一維卷積
    /// x: 當前輸入特徵 [Batch, Length_new, hidden_size]
    /// conv_cache: 卷積快取 [Batch, kernel_size - 1, hidden_size]
    /// 返回: (輸出特徵 [Batch, Length_new, hidden_size], 新卷積快取)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        conv_cache: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let device = x.device();
        let shape = x.shape();
        let dims = shape.dims::<3>();
        let batch = dims[0];
        // let len_new = dims[1];
        let hidden = dims[2];
        let cache_len = self.kernel_size - 1;

        // 1. LayerNorm
        let h = self.ln.forward(x.clone());

        // 2. 拼接一維卷積的歷史快取
        let h_total = match conv_cache {
            Some(cc) => Tensor::cat(vec![cc, h], 1),
            _ => {
                // 如果是首次運行，用零特徵填充作為左側 padding
                let pad = Tensor::zeros([batch, cache_len, hidden], &device);
                Tensor::cat(vec![pad, h], 1)
            }
        };

        // 3. 儲存新的 conv_cache (最後的 kernel_size - 1 步)
        let total_len = h_total.shape().dims::<3>()[1];
        let next_cache = h_total.clone().slice([
            0..batch,
            (total_len - cache_len)..total_len,
        ]);

        // 4. 轉置為 [Batch, D, Length] 格式以適應一維卷積
        let h_conv = h_total.swap_dims(1, 2); // -> [Batch, D, Length]

        // 5. Pointwise Conv 1
        let h_conv = self.pointwise_conv1.forward(h_conv);

        // 6. Gated Linear Unit (GLU) 激活
        // 沿著維度 1 (通道維度) 拆分，前 D 個通道作為門，後 D 個通道作為值
        let d = hidden;
        let gate = h_conv.clone().slice([0..batch, 0..d]);
        let val = h_conv.slice([0..batch, d..(2 * d)]);
        let h_conv = gate * burn::tensor::activation::sigmoid(val);

        // 7. Depthwise Separable 1D Conv (因果卷積)
        let h_conv = self.depthwise_conv.forward(h_conv); // -> 輸出長度會減去 kernel_size - 1，剛好恢復為 len_new

        // 8. SiLU (Swish) 激活
        let h_conv = burn::tensor::activation::silu(h_conv);

        // 9. Pointwise Conv 2
        let h_conv = self.pointwise_conv2.forward(h_conv);

        // 10. 轉置回 [Batch, Length_new, D]
        let output = h_conv.swap_dims(1, 2);

        (output, next_cache)
    }
}

#[derive(Module, Debug)]
pub struct ConformerBlock<B: Backend> {
    pub ffn1: FeedForwardModule<B>,
    pub attn: RelPositionMultiHeadAttention<B>,
    pub conv: ConformerConvolution<B>,
    pub ffn2: FeedForwardModule<B>,
    pub post_ln: LayerNorm<B>,
}

impl<B: Backend> ConformerBlock<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        left_context: usize,
        conv_kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        let ffn1 = FeedForwardModule::new(hidden_size, 4, device);
        let attn = RelPositionMultiHeadAttention::new(hidden_size, num_heads, left_context, device);
        let conv = ConformerConvolution::new(hidden_size, conv_kernel_size, device);
        let ffn2 = FeedForwardModule::new(hidden_size, 4, device);
        let post_ln = LayerNormConfig::new(hidden_size).init(device);

        Self { ffn1, attn, conv, ffn2, post_ln }
    }

    /// 帶有狀態快取的 Conformer Block 前向傳播
    /// x: [Batch, Length_new, hidden_size]
    /// pos_emb: 相對位置編碼
    /// cache_k, cache_v: Attention 快取
    /// conv_cache: 卷積快取
    /// 返回: (輸出, 新 cache_k, 新 cache_v, 新 conv_cache)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        cache_k: Option<Tensor<B, 3>>,
        cache_v: Option<Tensor<B, 3>>,
        conv_cache: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // 1. Macaron FFN1: x = x + 0.5 * FFN1(x)
        let x = x.clone() + self.ffn1.forward(x) * 0.5;

        // 2. RelPositionMHSA: x = x + MHSA(x)
        let (attn_out, next_k, next_v) = self.attn.forward(
            x.clone(), 
            pos_emb, 
            cache_k, 
            cache_v
        );
        let x = x + attn_out;

        // 3. Conformer Convolution: x = x + Conv(x)
        let (conv_out, next_conv_cache) = self.conv.forward(x.clone(), conv_cache);
        let x = x + conv_out;

        // 4. Macaron FFN2: x = x + 0.5 * FFN2(x)
        let x = x.clone() + self.ffn2.forward(x) * 0.5;

        // 5. Post LayerNorm
        let output = self.post_ln.forward(x);

        (output, next_k, next_v, next_conv_cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use burn::tensor::Distribution;

    type TestBackend = burn_flex::Flex;

    /// 測試 FeedForwardModule 的輸出形狀
    /// 輸入 [1,4,64]，預期輸出 [1,4,64]
    #[test]
    #[wasm_bindgen_test]
    fn test_feedforward_output_shape() {
        let device = &Default::default();
        let ffn = FeedForwardModule::<TestBackend>::new(64, 4, device);
        let x = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, device);
        let out = ffn.forward(x);
        assert_eq!(out.shape().dims::<3>(), [1, 4, 64]);
    }

    /// 測試 ConformerConvolution 的輸出形狀（無快取情況）
    /// 輸入 [1,4,64]，kernel_size=9
    /// 預期輸出 [1,4,64]，conv_cache 形狀 [1,8,64]（kernel_size-1=8）
    #[test]
    #[wasm_bindgen_test]
    fn test_conformer_conv_output_shape() {
        let device = &Default::default();
        let conv = ConformerConvolution::<TestBackend>::new(64, 9, device);
        let x = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, device);
        let (out, cache) = conv.forward(x, None);
        assert_eq!(out.shape().dims::<3>(), [1, 4, 64]);
        // conv_cache 長度 = kernel_size - 1 = 8
        assert_eq!(cache.shape().dims::<3>(), [1, 8, 64]);
    }

    /// 測試帶有快取的 ConformerConvolution
    /// 先執行一次取得 conv_cache，再用新輸入和快取執行第二次
    /// 驗證輸出形狀保持一致
    #[test]
    #[wasm_bindgen_test]
    fn test_conformer_conv_with_cache() {
        let device = &Default::default();
        let conv = ConformerConvolution::<TestBackend>::new(64, 9, device);

        // 第一次前向：無快取
        let x1 = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, device);
        let (out1, cache1) = conv.forward(x1, None);
        assert_eq!(out1.shape().dims::<3>(), [1, 4, 64]);
        assert_eq!(cache1.shape().dims::<3>(), [1, 8, 64]);

        // 第二次前向：帶上快取
        let x2 = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, device);
        let (out2, cache2) = conv.forward(x2, Some(cache1));
        assert_eq!(out2.shape().dims::<3>(), [1, 4, 64]);
        // 快取形狀仍應為 [1, kernel_size-1, 64]
        assert_eq!(cache2.shape().dims::<3>(), [1, 8, 64]);
    }

    /// 測試 ConformerBlock 的輸出形狀及快取返回
    /// hidden_size=64, num_heads=4, left_context=8, conv_kernel_size=9
    /// 輸入 x=[1,4,64]，pos_emb=[1,7,64]（2*4-1=7）
    /// 預期輸出 [1,4,64]，並返回 cache_k, cache_v, conv_cache
    #[test]
    #[wasm_bindgen_test]
    fn test_conformer_block_output_shape() {
        let device = &Default::default();
        let block = ConformerBlock::<TestBackend>::new(64, 4, 8, 9, device);
        let x = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, device);
        // pos_emb 長度 = 2 * total_len - 1 = 2 * 4 - 1 = 7（無快取時 total_len = len_new）
        let pos_emb = Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Default, device);

        let (out, cache_k, cache_v, conv_cache) =
            block.forward(x, pos_emb, None, None, None);

        // 輸出形狀 [1, 4, 64]
        assert_eq!(out.shape().dims::<3>(), [1, 4, 64]);

        // Attention KV 快取已返回（形狀取決於 left_context 裁剪，但至少包含 len_new=4 步）
        assert_eq!(cache_k.shape().dims::<3>()[0], 1);
        assert_eq!(cache_k.shape().dims::<3>()[2], 64);
        assert_eq!(cache_v.shape().dims::<3>()[0], 1);
        assert_eq!(cache_v.shape().dims::<3>()[2], 64);

        // Conv 快取形狀 [1, kernel_size-1, 64] = [1, 8, 64]
        assert_eq!(conv_cache.shape().dims::<3>(), [1, 8, 64]);
    }
}
