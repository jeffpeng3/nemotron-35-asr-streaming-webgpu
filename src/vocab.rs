use tokenizers::Tokenizer;
use std::collections::HashMap;

pub struct VocabProcessor {
    tokenizer: Tokenizer,
    blank_id: u32,
    lang_prompts: HashMap<String, u32>,
}

impl VocabProcessor {
    /// 載入 tokenizer.json 檔案的字串數據，並初始化分詞器
    pub fn new(json_content: &str, blank_id: Option<u32>, lang_map: Option<HashMap<String, u32>>) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_bytes(json_content.as_bytes())
            .map_err(|e| format!("解析 tokenizer.json 失敗: {:?}", e))?;
            
        // 如果沒有指定 blank_id，預設設為詞表大小減一 (NeMo RNNT 的標準做法)
        let vocab_size = tokenizer.get_vocab_size(true);
        let blank_id = blank_id.unwrap_or((vocab_size - 1) as u32);
        
        let mut lang_prompts = HashMap::new();
        if let Some(map) = lang_map {
            lang_prompts = map;
        } else {
            // 如果沒有傳入，我們嘗試在詞表中自動檢索常見的語言特殊 token
            // 比如 NeMo 常用特殊 Token 格式: "<|en|>", "<|zh-CN|>" 等，或 "en-US", "zh-CN"
            let locales = vec![
                ("en", vec!["<|en|>", "<|en-US|>", "en", "en-US"]),
                ("zh", vec!["<|zh|>", "<|zh-CN|>", "zh", "zh-CN"]),
                ("ja", vec!["<|ja|>", "<|ja-JP|>", "ja", "ja-JP"]),
                ("de", vec!["<|de|>", "<|de-DE|>", "de", "de-DE"]),
                ("es", vec!["<|es|>", "<|es-ES|>", "es", "es-ES"]),
                ("fr", vec!["<|fr|>", "<|fr-FR|>", "fr", "fr-FR"]),
                ("ko", vec!["<|ko|>", "<|ko-KR|>", "ko", "ko-KR"]),
            ];
            
            for (lang, tokens) in locales {
                for token in tokens {
                    if let Some(id) = tokenizer.token_to_id(token) {
                        lang_prompts.insert(lang.to_string(), id);
                        break;
                    }
                }
            }
        }

        Ok(Self {
            tokenizer,
            blank_id,
            lang_prompts,
        })
    }

    /// 取得 Blank Token ID
    pub fn get_blank_id(&self) -> u32 {
        self.blank_id
    }

    /// 根據語言簡稱（如 "zh", "en"）獲取其在模型中的 Prompt Token ID
    pub fn get_prompt_id(&self, lang: &str) -> Option<u32> {
        self.lang_prompts.get(lang).cloned()
    }

    /// 解碼一組 Token ID 為文字
    /// skip_special_tokens: 是否過濾特殊控制字元
    pub fn decode(&self, ids: &[u32]) -> String {
        // 過濾掉 blank id，因為 SentencePiece 可能不將其視為 special token，但 ASR 解碼必須過濾
        let filtered_ids: Vec<u32> = ids.iter()
            .cloned()
            .filter(|&id| id != self.blank_id)
            .collect();
            
        self.tokenizer.decode(&filtered_ids, true).unwrap_or_default()
    }

    /// 將單個 Token ID 解碼為字串（用於逐字增量流式輸出）
    pub fn decode_single(&self, id: u32) -> String {
        if id == self.blank_id {
            return String::new();
        }
        self.tokenizer.decode(&[id], true).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use std::collections::HashMap;

    /// 內聯的最小化 BPE tokenizer JSON，用於無需外部檔案的測試
    /// 詞表: <unk>=0, <s>=1, </s>=2, a=3, b=4, c=5, ab=6
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "c": 5, "ab": 6},
            "merges": ["a b"]
        }
    }"#;

    /// 測試：使用自定義 blank_id 建構 VocabProcessor，驗證 get_blank_id() 回傳正確值
    #[test]
    #[wasm_bindgen_test]
    fn test_vocab_with_minimal_tokenizer() {
        // 指定 blank_id = 3（對應 token "a"，僅做邏輯驗證用途）
        let vocab = VocabProcessor::new(MINIMAL_TOKENIZER_JSON, Some(3), None)
            .expect("應能從內聯 JSON 建立 VocabProcessor");

        assert_eq!(vocab.get_blank_id(), 3, "blank_id 應為指定的 3");
    }

    /// 測試：decode() 應過濾掉 blank_id 對應的 token
    #[test]
    #[wasm_bindgen_test]
    fn test_vocab_decode_filters_blank() {
        // blank_id = 3 → token "a" 會被過濾
        let vocab = VocabProcessor::new(MINIMAL_TOKENIZER_JSON, Some(3), None)
            .expect("應能從內聯 JSON 建立 VocabProcessor");

        // ids = [3, 4, 3, 5] → 過濾 blank(3) 後剩 [4, 5] → "b" + "c" = "bc"
        let result = vocab.decode(&[3, 4, 3, 5]);
        assert!(
            !result.contains('a'),
            "解碼結果不應包含 blank token 'a'，實際: {result}"
        );
        assert!(
            result.contains('b') && result.contains('c'),
            "解碼結果應包含 'b' 和 'c'，實際: {result}"
        );
    }

    /// 測試：使用自定義 lang_map，驗證 get_prompt_id() 正確查詢或返回 None
    #[test]
    #[wasm_bindgen_test]
    fn test_vocab_get_prompt_id_custom_map() {
        let mut lang_map = HashMap::new();
        lang_map.insert("en".to_string(), 10u32);
        lang_map.insert("zh".to_string(), 20u32);

        let vocab = VocabProcessor::new(MINIMAL_TOKENIZER_JSON, Some(3), Some(lang_map))
            .expect("應能從內聯 JSON 建立 VocabProcessor");

        // 存在的語言應回傳對應 ID
        assert_eq!(vocab.get_prompt_id("en"), Some(10), "en 應對應 prompt_id 10");
        assert_eq!(vocab.get_prompt_id("zh"), Some(20), "zh 應對應 prompt_id 20");

        // 不存在的語言應回傳 None
        assert_eq!(vocab.get_prompt_id("fr"), None, "fr 不在 lang_map 中，應為 None");
    }
}
