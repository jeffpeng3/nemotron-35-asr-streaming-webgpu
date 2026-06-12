use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};
use crate::model::subsampling::ConvSubsampling;
use crate::model::conformer::ConformerBlock;

/// 儲存 Encoder 快取狀態的結構體
#[derive(Clone)]
pub struct EncoderCache<B: Backend> {
    // 每個 Conformer Block 的快取，包含：(cache_k, cache_v, conv_cache)
    pub blocks: Vec<(Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>)>,
    // Subsampling 的音訊特徵緩衝區（Log-mel 幀快取，長度最多為 14 幀）
    pub subsampling_buffer: Option<Tensor<B, 4>>,
}

impl<B: Backend> EncoderCache<B> {
    pub fn new(_num_layers: usize) -> Self {
        Self {
            blocks: vec![],
            subsampling_buffer: None,
        }
    }
}

#[derive(Module, Debug)]
pub struct FastConformerEncoder<B: Backend> {
    pub subsampling: ConvSubsampling<B>,
    pub blocks: Vec<ConformerBlock<B>>,
    pub hidden_size: usize,
    pub num_layers: usize,
}

impl<B: Backend> FastConformerEncoder<B> {
    pub fn new(hidden_size: usize, num_layers: usize, num_heads: usize, left_context: usize, device: &B::Device) -> Self {
        let subsampling = ConvSubsampling::new(hidden_size, device);
        
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Conv kernel size typically 31 for Conformer Convolution
            blocks.push(ConformerBlock::new(hidden_size, num_heads, left_context, 31, device));
        }

        Self {
            subsampling,
            blocks,
            hidden_size,
            num_layers,
        }
    }

    /// 帶有狀態快取的流式 Encoder 前向傳播
    /// new_mel: 新輸入的 Log-mel 幀 [Batch, 1, L_new, 80]
    /// cache: 歷史快取狀態
    /// 返回: (下採樣後的隱表徵 [Batch, L_sub_new, hidden_size], 更新後的快取)
    pub fn forward(
        &self,
        new_mel: Tensor<B, 4>,
        mut cache: EncoderCache<B>,
    ) -> (Tensor<B, 3>, EncoderCache<B>) {
        let shape = new_mel.shape();
        let [batch, _, _, _] = shape.dims::<4>();
        
        // 1. 處理 Subsampling 快取與拼接
        // Subsampling 需要最近的 14 幀作為左側上下文
        let subsampling_context = 14;
        let x_sub_input = match cache.subsampling_buffer {
            Some(buf) => Tensor::cat(vec![buf, new_mel], 2),
            _ => new_mel,
        };

        // 2. 儲存新的 Subsampling 快取 (最後的 14 幀)
        let [_, _, total_mel_len, _] = x_sub_input.shape().dims::<4>();
        let next_sub_buffer = if total_mel_len > subsampling_context {
            let start = total_mel_len - subsampling_context;
            Some(x_sub_input.clone().slice([0..batch, 0..1, start..total_mel_len, 0..80]))
        } else {
            Some(x_sub_input.clone())
        };
        cache.subsampling_buffer = next_sub_buffer;

        // 3. 運行卷積下採樣
        let x = self.subsampling.forward(x_sub_input.clone()); // -> [Batch, L_sub_total, hidden_size]
        
        // 計算由於下採樣帶來的位置映射：
        // 比如當前輸入了 N 幀 Log-mel，加上快取共 L 幀。下採樣後長度為 L_sub_total。
        // 我們需要對新增加的時間步進行切片輸出。
        // 對於 8x 下採樣，如果輸入增長了 N 幀，輸出大約增長 N / 8 幀。
        // 在流式中，我們希望只輸出這新增的部分。
        // 設當前輸出的總長度是 L_sub_total。如果上一次的 cache.blocks 中已有快取，
        // 則代表此時之前的特徵都已經在之前的時間步被編碼過了。
        // 所以，我們在 24 層 Conformer 推理後，會對隱特徵進行切片，只取最新的部分。
        let [_, len_sub_total, _] = x.shape().dims::<3>();
        
        // 4. 生成相對位置正弦編碼
        // 相對位置範圍通常是 2 * len_sub_total - 1
        let device = x.device();
        let pos_emb = Self::generate_positional_embedding(len_sub_total, self.hidden_size, &device);

        // 5. 逐層運行 Conformer Blocks
        let mut h = x;
        let mut next_blocks_cache = Vec::with_capacity(self.num_layers);
        
        for i in 0..self.num_layers {
            let block = &self.blocks[i];
            
            // 讀取當前層的快取
            let (cache_k, cache_v, conv_cache) = if i < cache.blocks.len() {
                let (ck, cv, cc) = &cache.blocks[i];
                (Some(ck.clone()), Some(cv.clone()), Some(cc.clone()))
            } else {
                (None, None, None)
            };
            
            let (h_next, nk, nv, ncc) = block.forward(
                h, 
                pos_emb.clone(), 
                cache_k, 
                cache_v, 
                conv_cache
            );
            
            h = h_next;
            next_blocks_cache.push((nk, nv, ncc));
        }
        
        cache.blocks = next_blocks_cache;

        // 6. 對 Encoder Output 進行切片，只取新增的部分
        // 設上一次 Attention 快取長度是 L_prev。在流式中，新增的編碼隱特徵對應的長度是
        // L_sub_total - L_prev。
        // 如果沒有歷史快取，則全部輸出。
        // 我們可以透過檢查第一層 Attention 快取長度來確定 L_prev。
        let [_, cache_len, _] = cache.blocks[0].0.shape().dims::<3>();
        let new_encoded = if len_sub_total > cache_len {
            // 當前已經經過裁剪，cache 的長度被限制在 left_context (如 56)
            // 故我們直接取後部 len_new_sub = len_sub_total - prev_len
            // 但注意，我們之前在 Block 內部裁剪了快取。
            // 為了簡化，如果我們知道每次餵入 320ms 音訊 (即 32 幀 Log-mel，下採樣後增長 4 幀)，
            // 我們可以直接輸出後部的 4 幀，或者動態計算增長差。
            // 增長差就是：當前下採樣總長度減去上一幀在 subsampling 後所對應的長度。
            // 更精準的做法是：
            // 在 forward 之前，subsampling_buffer 的長度為 L_prev_mel。
            // new_mel 長度為 L_new_mel。
            // 下採樣前總長度為 L_prev_mel + L_new_mel。
            // 則新增的下採樣幀數為 (L_prev_mel + L_new_mel)/8 - (L_prev_mel)/8。
            // 我們直接取 h.slice( [0..batch, (L_prev_sub)..(L_total_sub)] )
            let [_, _, x_sub_mel_len, _] = x_sub_input.shape().dims::<4>();
            let prev_mel_len = x_sub_mel_len.saturating_sub(total_mel_len);
            let prev_sub_len = prev_mel_len / 8;
            let total_sub_len = x_sub_mel_len / 8;
            let new_sub_len = total_sub_len - prev_sub_len;
            
            let start = len_sub_total.saturating_sub(new_sub_len);
            h.slice([0..batch, start..len_sub_total])
        } else {
            h
        };

        (new_encoded, cache)
    }

    /// 動態生成正弦相對位置編碼
    fn generate_positional_embedding(length: usize, d_model: usize, device: &B::Device) -> Tensor<B, 3> {
        let max_len = length * 2;
        let mut pe = vec![vec![0.0f32; d_model]; max_len];
        
        for pos in 0..max_len {
            let rel_pos = (pos as f32) - (length as f32);
            for i in (0..d_model).step_by(2) {
                let div_term = (10000.0_f32).powf((i as f32) / (d_model as f32));
                let sin_val = (rel_pos / div_term).sin();
                let cos_val = (rel_pos / div_term).cos();
                pe[pos][i] = sin_val;
                if i + 1 < d_model {
                    pe[pos][i + 1] = cos_val;
                }
            }
        }
        
        // 將二維 Vec 轉換為 Tensor [1, max_len, d_model]
        // 由於 Burn 初始化二維 Tensor 常用 `Tensor::from_data`
        // 為了支援 Wasm 特性且不依賴特定後端硬體數據複製，我們可以：
        let data: Vec<f32> = pe.into_iter().flatten().collect();
        let tensor: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(data, [max_len, d_model]),
            device,
        );
        tensor.unsqueeze() // -> [1, max_len, d_model]
    }
}

#[derive(Module, Debug)]
pub struct StatelessPredictor<B: Backend> {
    pub embed: Embedding<B>,
    pub proj: Linear<B>,
}

impl<B: Backend> StatelessPredictor<B> {
    pub fn new(vocab_size: usize, pred_hidden: usize, joiner_hidden: usize, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(vocab_size, pred_hidden).init(device);
        let proj = LinearConfig::new(pred_hidden, joiner_hidden).init(device);
        Self { embed, proj }
    }

    /// stateless Predictor 前向傳播
    /// prev_token: 最近一個 (或多個) Token ID [Batch, 1]
    /// 返回: [Batch, 1, joiner_hidden]
    pub fn forward(&self, prev_token: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let x = self.embed.forward(prev_token);
        self.proj.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct RNNTJoiner<B: Backend> {
    pub enc_proj: Linear<B>,
    pub pred_proj: Linear<B>,
    pub joint_proj: Linear<B>,
}

impl<B: Backend> RNNTJoiner<B> {
    pub fn new(enc_hidden: usize, pred_hidden: usize, joiner_hidden: usize, vocab_size: usize, device: &B::Device) -> Self {
        let enc_proj = LinearConfig::new(enc_hidden, joiner_hidden).init(device);
        let pred_proj = LinearConfig::new(pred_hidden, joiner_hidden).init(device);
        let joint_proj = LinearConfig::new(joiner_hidden, vocab_size).init(device);
        
        Self {
            enc_proj,
            pred_proj,
            joint_proj,
        }
    }

    /// 結合 Encoder 隱特徵與 Predictor 隱特徵，預測 Token Logits
    /// enc_out: [Batch, 1, enc_hidden]
    /// pred_out: [Batch, 1, pred_hidden]
    /// 返回: [Batch, 1, vocab_size]
    pub fn forward(&self, enc_out: Tensor<B, 3>, pred_out: Tensor<B, 3>) -> Tensor<B, 3> {
        let h_enc = self.enc_proj.forward(enc_out);
        let h_pred = self.pred_proj.forward(pred_out);
        
        // 廣播相加
        let h_joint = h_enc + h_pred;
        
        // 激活與投影
        let h_joint = burn::tensor::activation::relu(h_joint);
        self.joint_proj.forward(h_joint)
    }
}

#[derive(Module, Debug)]
pub struct RNNTModel<B: Backend> {
    pub encoder: FastConformerEncoder<B>,
    pub predictor: StatelessPredictor<B>,
    pub joiner: RNNTJoiner<B>,
    pub vocab_size: usize,
}

impl<B: Backend> RNNTModel<B> {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        left_context: usize,
        pred_hidden: usize,
        joiner_hidden: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = FastConformerEncoder::new(hidden_size, num_layers, num_heads, left_context, device);
        let predictor = StatelessPredictor::new(vocab_size, pred_hidden, joiner_hidden, device);
        let joiner = RNNTJoiner::new(hidden_size, joiner_hidden, joiner_hidden, vocab_size, device);
        
        Self {
            encoder,
            predictor,
            joiner,
            vocab_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    type TestBackend = burn_flex::Flex;

    /// 測試 StatelessPredictor 前向傳播的輸出形狀
    #[test]
    #[wasm_bindgen_test]
    fn test_predictor_output_shape() {
        let device = Default::default();
        let predictor = StatelessPredictor::<TestBackend>::new(100, 64, 64, &device);

        // prev_token: [1, 1]，值為 0（代表 <blank> 或初始 Token）
        let prev_token = Tensor::<TestBackend, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![0i32], [1, 1]),
            &device,
        );

        let output = predictor.forward(prev_token);
        let shape = output.shape();

        assert_eq!(shape.dims::<3>(), [1, 1, 64], "Predictor 輸出形狀應為 [1, 1, 64]");
    }

    /// 測試 RNNTJoiner 前向傳播的輸出形狀
    #[test]
    #[wasm_bindgen_test]
    fn test_joiner_output_shape() {
        let device = Default::default();
        let joiner = RNNTJoiner::<TestBackend>::new(64, 64, 64, 100, &device);

        // 模擬 Encoder 和 Predictor 的輸出，都是 [1, 1, 64] 的零張量
        let enc_out = Tensor::<TestBackend, 3>::zeros([1, 1, 64], &device);
        let pred_out = Tensor::<TestBackend, 3>::zeros([1, 1, 64], &device);

        let output = joiner.forward(enc_out, pred_out);
        let shape = output.shape();

        assert_eq!(shape.dims::<3>(), [1, 1, 100], "Joiner 輸出形狀應為 [1, 1, vocab_size=100]");
    }

    /// 測試 EncoderCache 初始化狀態
    #[test]
    #[wasm_bindgen_test]
    fn test_encoder_cache_init() {
        let cache = EncoderCache::<TestBackend>::new(4);

        assert!(cache.blocks.is_empty(), "初始化時 blocks 應為空");
        assert!(cache.subsampling_buffer.is_none(), "初始化時 subsampling_buffer 應為 None");
    }

    /// 測試 RNNTModel 初始化後的 vocab_size 欄位
    #[test]
    #[wasm_bindgen_test]
    fn test_rnnt_model_init() {
        let device = Default::default();
        let model = RNNTModel::<TestBackend>::new(
            100, // vocab_size
            64,  // hidden_size
            2,   // num_layers
            4,   // num_heads
            8,   // left_context
            64,  // pred_hidden
            64,  // joiner_hidden
            &device,
        );

        assert_eq!(model.vocab_size, 100, "vocab_size 應為 100");
    }
}
