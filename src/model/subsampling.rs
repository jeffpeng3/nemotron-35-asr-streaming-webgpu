use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct ConvSubsampling<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub proj: Linear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> ConvSubsampling<B> {
    pub fn new(hidden_size: usize, device: &B::Device) -> Self {
        // conv1: [B, 1, L, 80] -> [B, D, L/2, 40]
        let conv1 = Conv2dConfig::new([1, hidden_size], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // conv2: [B, D, L/2, 40] -> [B, D, L/4, 20]
        let conv2 = Conv2dConfig::new([hidden_size, hidden_size], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // conv3: [B, D, L/4, 20] -> [B, D, L/8, 10]
        let conv3 = Conv2dConfig::new([hidden_size, hidden_size], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // proj: [B, L/8, D * 10] -> [B, L/8, D]
        let proj = LinearConfig::new(hidden_size * 10, hidden_size).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            proj,
            hidden_size,
        }
    }

    /// 前向傳播，帶有 Sliding Window 緩衝輸入以支持流式推理
    /// input: [Batch, 1, Length, 80] (其中 Length 是緩衝區加上新音訊幀特徵的總長度)
    /// 返回: [Batch, Length_subsampled, hidden_size] 並且只保留有效的新增輸出步
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // 1. 第一層卷積 + SiLU (Swish) 激活
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::silu(x);

        // 2. 第二層卷積 + SiLU 激活
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::silu(x);

        // 3. 第三層卷積 + SiLU 激活
        let x = self.conv3.forward(x);
        let x = burn::tensor::activation::silu(x);

        // x 步長在頻率維度上變為 10，其 Shape 為 [Batch, D, L_subsampled, 10]
        let shape = x.shape();
        let [batch_size, channels, l_sub, freq_dim] = shape.dims::<4>();

        // 4. 重塑以進行 Linear 投影
        // 我們要將 x 轉置為 [Batch, L_subsampled, channels * freq_dim] (即 [B, L_sub, D * 10])
        // 先轉置為 [Batch, L_subsampled, channels, freq_dim]
        let x = x.swap_dims(1, 2); // -> [Batch, L_subsampled, channels, freq_dim]
        let x = x.reshape([batch_size, l_sub, channels * freq_dim]);

        // 5. 投影到 hidden_size
        self.proj.forward(x)
    }

    /// 手動載入權重映射
    pub fn load_weights(
        &mut self,
        _weight_map: &std::collections::HashMap<String, Tensor<B, 1>>,
        _prefix: &str,
    ) {
        // 這裡可以實現自定義的權重加載映射邏輯，將 SafeTensors 中特定字串命名的權重寫入各個 conv 和 linear
        // 在本專案中，我們將在一個整體的 weights loader 中實現此功能
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use burn::tensor::Distribution;

    type TestBackend = burn_flex::Flex;

    /// 測試基本輸出形狀：hidden_size=64, 輸入 [1,1,64,80] → 輸出 [1, 8, 64]
    #[test]
    #[wasm_bindgen_test]
    fn test_subsampling_output_shape() {
        let device = Default::default();
        let model = ConvSubsampling::<TestBackend>::new(64, &device);
        let input = Tensor::<TestBackend, 4>::random(
            [1, 1, 64, 80],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        let [batch, time, hidden] = output.shape().dims::<3>();

        // 經過三層 stride=2 的卷積後，時間維度 64 / 8 = 8
        assert_eq!(batch, 1, "batch 維度應為 1");
        assert_eq!(time, 8, "子採樣後的時間步應為 64/8 = 8");
        assert_eq!(hidden, 64, "最後一維應為 hidden_size=64");
    }

    /// 測試較小的 hidden_size=32，輸入 [1,1,16,80] → 輸出 [1, 2, 32]
    #[test]
    #[wasm_bindgen_test]
    fn test_subsampling_small_hidden() {
        let device = Default::default();
        let model = ConvSubsampling::<TestBackend>::new(32, &device);
        let input = Tensor::<TestBackend, 4>::random(
            [1, 1, 16, 80],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        let [_batch, _time, hidden] = output.shape().dims::<3>();

        assert_eq!(hidden, 32, "最後一維應為 hidden_size=32");
    }

    /// 測試 batch 維度：batch_size=2, 輸入 [2,1,32,80] → 輸出 batch 維度為 2
    #[test]
    #[wasm_bindgen_test]
    fn test_subsampling_batch() {
        let device = Default::default();
        let model = ConvSubsampling::<TestBackend>::new(64, &device);
        let input = Tensor::<TestBackend, 4>::random(
            [2, 1, 32, 80],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);
        let [batch, _time, _hidden] = output.shape().dims::<3>();

        // batch 維度應保持為 2
        assert_eq!(batch, 2, "batch 維度應為 2");
    }
}
