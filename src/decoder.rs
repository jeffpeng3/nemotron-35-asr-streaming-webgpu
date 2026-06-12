use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};
use crate::model::rnnt::RNNTModel;

pub struct RnntGreedyDecoder {
    blank_id: u32,
    last_token_id: u32,
    max_symbols_per_step: usize,
}

impl RnntGreedyDecoder {
    pub fn new(blank_id: u32, start_token_id: u32) -> Self {
        Self {
            blank_id,
            last_token_id: start_token_id,
            max_symbols_per_step: 10, // 防範死循環
        }
    }

    /// 重設解碼狀態，重新指定起點 Token (例如切換語言 prompt)
    pub fn reset(&mut self, start_token_id: u32) {
        self.last_token_id = start_token_id;
    }

    /// 進行流式 RNN-T 貪婪解碼
    /// model: RNNT 模型實例
    /// enc_out: Encoder 新增輸出的隱特徵 [Batch=1, Length_new, hidden_size]
    /// 返回: 新解碼出的 Token ID 列表
    pub fn decode<B: Backend>(&mut self, model: &RNNTModel<B>, enc_out: Tensor<B, 3>, device: &B::Device) -> Vec<u32> {
        let shape = enc_out.shape();
        let len_new = shape.dims::<3>()[1];
        
        let mut decoded_tokens = Vec::new();
        
        // 逐個時間步進行解碼
        for t in 0..len_new {
            // 取得當前時間步的 Encoder 隱向量: Shape [1, 1, hidden_size]
            let h_t = enc_out.clone().slice([0..1, t..t+1]);
            
            let mut symbols_in_step = 0;
            
            loop {
                // 1. 準備 Predictor 輸入 (上一個 Predictor 預測的 Token ID)
                let prev_token_tensor = Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(vec![self.last_token_id as i32], [1, 1]),
                    device,
                );
                
                // 2. 預測
                let pred_out = model.predictor.forward(prev_token_tensor);
                
                // 3. 結合
                let logits = model.joiner.forward(h_t.clone(), pred_out); // -> [1, 1, vocab_size]
                
                // 4. Argmax 取得最大機率 Token ID
                let argmax_tensor = logits.argmax(2); // -> [1, 1]
                
                // 讀回 ID
                let pred_id = argmax_tensor.into_data().to_vec::<i32>().unwrap()[0] as u32;
                
                // 5. 判斷是否為 Blank Token
                if pred_id == self.blank_id {
                    // 時間步前進，跳出當前迴圈
                    break;
                }
                
                // 6. 記錄新預測的 Token
                decoded_tokens.push(pred_id);
                self.last_token_id = pred_id;
                
                symbols_in_step += 1;
                if symbols_in_step >= self.max_symbols_per_step {
                    // 超過限制，防範模型卡死
                    break;
                }
            }
        }
        
        decoded_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    // 使用 Flex 作為測試後端，不需要 GPU
    type TestBackend = burn_flex::Flex;

    /// 測試初始化：blank_id 和 last_token_id 應正確設置
    #[test]
    #[wasm_bindgen_test]
    fn test_decoder_init() {
        let decoder = RnntGreedyDecoder::new(99, 0);
        assert_eq!(decoder.blank_id, 99, "blank_id 應為 99");
        assert_eq!(decoder.last_token_id, 0, "last_token_id 應等於 start_token_id=0");
    }

    /// 測試重設：reset 後 last_token_id 應更新為新值
    #[test]
    #[wasm_bindgen_test]
    fn test_decoder_reset() {
        let mut decoder = RnntGreedyDecoder::new(99, 0);
        assert_eq!(decoder.last_token_id, 0);

        decoder.reset(5);
        assert_eq!(decoder.last_token_id, 5, "reset 後 last_token_id 應為 5");
    }

    /// 測試 max_symbols_per_step 預設值為 10
    #[test]
    #[wasm_bindgen_test]
    fn test_decoder_max_symbols_limit() {
        let decoder = RnntGreedyDecoder::new(0, 0);
        assert_eq!(
            decoder.max_symbols_per_step, 10,
            "max_symbols_per_step 預設應為 10，用於防範死循環"
        );
    }
}
