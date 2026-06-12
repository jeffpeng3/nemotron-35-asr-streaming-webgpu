pub mod audio;
pub mod vocab;
pub mod decoder;
pub mod model;

use wasm_bindgen::prelude::*;
use std::sync::Arc;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::module::Param;

use crate::audio::AudioProcessor;
use crate::vocab::VocabProcessor;
use crate::decoder::RnntGreedyDecoder;
use crate::model::rnnt::{RNNTModel, EncoderCache};

// -------------------------------------------------------------
// SafeTensors 數據加載輔助函數
// -------------------------------------------------------------

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_i32_vec(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn get_burn_tensor<B: Backend, const D: usize>(
    safetensors: &safetensors::SafeTensors,
    keys: &[&str],
    shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D>, String> {
    for &key in keys {
        if let Ok(t) = safetensors.tensor(key) {
            let data = bytes_to_f32_vec(t.data());
            return Ok(Tensor::from_data(burn::tensor::TensorData::new(data, shape), device));
        }
    }
    Err(format!("找不到張量，嘗試了: {:?}", keys))
}

fn get_burn_tensor_int<B: Backend, const D: usize>(
    safetensors: &safetensors::SafeTensors,
    keys: &[&str],
    shape: [usize; D],
    device: &B::Device,
) -> Result<Tensor<B, D, burn::tensor::Int>, String> {
    for &key in keys {
        if let Ok(t) = safetensors.tensor(key) {
            let data = bytes_to_i32_vec(t.data());
            return Ok(Tensor::from_data(burn::tensor::TensorData::new(data, shape), device));
        }
    }
    Err(format!("找不到整型張量，嘗試了: {:?}", keys))
}

// -------------------------------------------------------------
// RNNTModel 權重加載擴展
// -------------------------------------------------------------

impl<B: Backend> RNNTModel<B> {
    pub fn load_from_safetensors(&mut self, data: &[u8], device: &B::Device) -> Result<(), String> {
        let safetensors = safetensors::SafeTensors::deserialize(data)
            .map_err(|e| format!("SafeTensors 解析失敗: {:?}", e))?;

        let hidden_size = self.vocab_size; // 通常 vocab_size 與預測層等維度對齊，我們根據初始化定義覆寫
        
        // 1. 載入 Subsampling 卷積層
        // 由於 Burn 的 Conv2d 的 weight shape 是 [out_channels, in_channels, k1, k2]
        // 與 PyTorch 格式一致，我們直接讀取
        let c1_shape = self.encoder.subsampling.conv1.weight.shape().dims::<4>();
        self.encoder.subsampling.conv1.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.0.weight",
            "encoder.subsampling.conv1.weight"
        ], c1_shape, device)?);
        
        let c1_bias_shape = self.encoder.subsampling.conv1.bias.as_ref().unwrap().shape().dims::<1>();
        self.encoder.subsampling.conv1.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.0.bias",
            "encoder.subsampling.conv1.bias"
        ], c1_bias_shape, device)?));

        let c2_shape = self.encoder.subsampling.conv2.weight.shape().dims::<4>();
        self.encoder.subsampling.conv2.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.1.weight",
            "encoder.subsampling.conv2.weight"
        ], c2_shape, device)?);
        
        let c2_bias_shape = self.encoder.subsampling.conv2.bias.as_ref().unwrap().shape().dims::<1>();
        self.encoder.subsampling.conv2.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.1.bias",
            "encoder.subsampling.conv2.bias"
        ], c2_bias_shape, device)?));

        let c3_shape = self.encoder.subsampling.conv3.weight.shape().dims::<4>();
        self.encoder.subsampling.conv3.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.2.weight",
            "encoder.subsampling.conv3.weight"
        ], c3_shape, device)?);
        
        let c3_bias_shape = self.encoder.subsampling.conv3.bias.as_ref().unwrap().shape().dims::<1>();
        self.encoder.subsampling.conv3.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.conv.2.bias",
            "encoder.subsampling.conv3.bias"
        ], c3_bias_shape, device)?));

        let proj_shape = self.encoder.subsampling.proj.weight.shape().dims::<2>();
        self.encoder.subsampling.proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.proj.weight",
            "encoder.subsampling.proj.weight"
        ], proj_shape, device)?);
        
        let proj_bias_shape = self.encoder.subsampling.proj.bias.as_ref().unwrap().shape().dims::<1>();
        self.encoder.subsampling.proj.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
            "encoder.pre_encode_table.proj.bias",
            "encoder.subsampling.proj.bias"
        ], proj_bias_shape, device)?));

        // 2. 載入 Conformer Blocks
        for i in 0..self.encoder.blocks.len() {
            let block = &mut self.encoder.blocks[i];
            let prefix = format!("encoder.layers.{}", i);
            
            // FFN1
            let ffn1_w1_shape = block.ffn1.linear1.weight.shape().dims::<2>();
            block.ffn1.linear1.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward1.linear1.weight", prefix),
                &format!("{}.ffn1.linear1.weight", prefix)
            ], ffn1_w1_shape, device)?);
            let ffn1_b1_shape = block.ffn1.linear1.bias.as_ref().unwrap().shape().dims::<1>();
            block.ffn1.linear1.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward1.linear1.bias", prefix),
                &format!("{}.ffn1.linear1.bias", prefix)
            ], ffn1_b1_shape, device)?));
            
            let ffn1_w2_shape = block.ffn1.linear2.weight.shape().dims::<2>();
            block.ffn1.linear2.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward1.linear2.weight", prefix),
                &format!("{}.ffn1.linear2.weight", prefix)
            ], ffn1_w2_shape, device)?);
            let ffn1_b2_shape = block.ffn1.linear2.bias.as_ref().unwrap().shape().dims::<1>();
            block.ffn1.linear2.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward1.linear2.bias", prefix),
                &format!("{}.ffn1.linear2.bias", prefix)
            ], ffn1_b2_shape, device)?));

            // Attention (MHSA)
            let q_shape = block.attn.q_proj.weight.shape().dims::<2>();
            block.attn.q_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.q_proj.weight", prefix),
                &format!("{}.attn.q_proj.weight", prefix)
            ], q_shape, device)?);
            let k_shape = block.attn.k_proj.weight.shape().dims::<2>();
            block.attn.k_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.k_proj.weight", prefix),
                &format!("{}.attn.k_proj.weight", prefix)
            ], k_shape, device)?);
            let v_shape = block.attn.v_proj.weight.shape().dims::<2>();
            block.attn.v_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.v_proj.weight", prefix),
                &format!("{}.attn.v_proj.weight", prefix)
            ], v_shape, device)?);
            let out_shape = block.attn.out_proj.weight.shape().dims::<2>();
            block.attn.out_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.out_proj.weight", prefix),
                &format!("{}.attn.out_proj.weight", prefix)
            ], out_shape, device)?);
            let pos_shape = block.attn.pos_proj.weight.shape().dims::<2>();
            block.attn.pos_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.pos_proj.weight", prefix),
                &format!("{}.attn.pos_proj.weight", prefix)
            ], pos_shape, device)?);

            // Attention相對位置偏置
            let u_shape = block.attn.pos_bias_u.shape().dims::<2>();
            block.attn.pos_bias_u = get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.pos_bias_u", prefix),
                &format!("{}.attn.pos_bias_u", prefix)
            ], u_shape, device)?;
            let v_shape = block.attn.pos_bias_v.shape().dims::<2>();
            block.attn.pos_bias_v = get_burn_tensor(&safetensors, &[
                &format!("{}.self_attn.pos_bias_v", prefix),
                &format!("{}.attn.pos_bias_v", prefix)
            ], v_shape, device)?;

            // Convolution Module
            let pconv1_shape = block.conv.pointwise_conv1.weight.shape().dims::<3>();
            block.conv.pointwise_conv1.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.conv_module.pointwise_conv1.weight", prefix),
                &format!("{}.conv.pointwise_conv1.weight", prefix)
            ], pconv1_shape, device)?);
            
            let dpconv_shape = block.conv.depthwise_conv.weight.shape().dims::<3>();
            block.conv.depthwise_conv.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.conv_module.depthwise_conv.weight", prefix),
                &format!("{}.conv.depthwise_conv.weight", prefix)
            ], dpconv_shape, device)?);

            let pconv2_shape = block.conv.pointwise_conv2.weight.shape().dims::<3>();
            block.conv.pointwise_conv2.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.conv_module.pointwise_conv2.weight", prefix),
                &format!("{}.conv.pointwise_conv2.weight", prefix)
            ], pconv2_shape, device)?);

            // FFN2
            let ffn2_w1_shape = block.ffn2.linear1.weight.shape().dims::<2>();
            block.ffn2.linear1.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward2.linear1.weight", prefix),
                &format!("{}.ffn2.linear1.weight", prefix)
            ], ffn2_w1_shape, device)?);
            let ffn2_b1_shape = block.ffn2.linear1.bias.as_ref().unwrap().shape().dims::<1>();
            block.ffn2.linear1.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward2.linear1.bias", prefix),
                &format!("{}.ffn2.linear1.bias", prefix)
            ], ffn2_b1_shape, device)?));
            
            let ffn2_w2_shape = block.ffn2.linear2.weight.shape().dims::<2>();
            block.ffn2.linear2.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward2.linear2.weight", prefix),
                &format!("{}.ffn2.linear2.weight", prefix)
            ], ffn2_w2_shape, device)?);
            let ffn2_b2_shape = block.ffn2.linear2.bias.as_ref().unwrap().shape().dims::<1>();
            block.ffn2.linear2.bias = Some(Param::from_tensor(get_burn_tensor(&safetensors, &[
                &format!("{}.feed_forward2.linear2.bias", prefix),
                &format!("{}.ffn2.linear2.bias", prefix)
            ], ffn2_b2_shape, device)?));
        }

        // 3. 載入 Predictor
        let pred_embed_shape = self.predictor.embed.weight.shape().dims::<2>();
        self.predictor.embed.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "decoder.prediction_net.embed.weight",
            "predictor.embed.weight"
        ], pred_embed_shape, device)?);

        let pred_proj_shape = self.predictor.proj.weight.shape().dims::<2>();
        self.predictor.proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "decoder.prediction_net.proj.weight",
            "predictor.proj.weight"
        ], pred_proj_shape, device)?);

        // 4. 載入 Joiner
        let j_enc_shape = self.joiner.enc_proj.weight.shape().dims::<2>();
        self.joiner.enc_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "joint_net.enc_proj.weight",
            "joiner.enc_proj.weight"
        ], j_enc_shape, device)?);

        let j_pred_shape = self.joiner.pred_proj.weight.shape().dims::<2>();
        self.joiner.pred_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "joint_net.pred_proj.weight",
            "joiner.pred_proj.weight"
        ], j_pred_shape, device)?);

        let j_joint_shape = self.joiner.joint_proj.weight.shape().dims::<2>();
        self.joiner.joint_proj.weight = Param::from_tensor(get_burn_tensor(&safetensors, &[
            "joint_net.joint_proj.weight",
            "joiner.joint_proj.weight"
        ], j_joint_shape, device)?);

        Ok(())
    }
}

// -------------------------------------------------------------
// Wasm 導出的 Session 實作
// -------------------------------------------------------------

pub struct SessionImpl<B: Backend> {
    model: RNNTModel<B>,
    audio_processor: AudioProcessor,
    vocab_processor: VocabProcessor,
    decoder: RnntGreedyDecoder,
    encoder_cache: EncoderCache<B>,
    language_prompt_id: u32,
}

enum ActiveSession {
    #[cfg(feature = "gpu")]
    Wgpu(SessionImpl<burn_wgpu::Wgpu>),
    Cpu(SessionImpl<burn_flex::Flex>),
}

#[wasm_bindgen]
pub struct AsrSessionConfig {
    pub use_gpu: bool,
    language: String,     // "zh", "en", "ja", "auto"
    latency_mode: String, // "low", "balanced", "high"
}

#[wasm_bindgen]
impl AsrSessionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(use_gpu: bool, language: &str, latency_mode: &str) -> Self {
        Self {
            use_gpu,
            language: language.to_string(),
            latency_mode: latency_mode.to_string(),
        }
    }
}

#[wasm_bindgen]
pub struct AsrSession {
    session: ActiveSession,
}

#[wasm_bindgen]
impl AsrSession {
    /// 靜態工廠方法，載入權重、配置並初始化 ASR Session
    #[wasm_bindgen]
    pub fn create(
        config: AsrSessionConfig,
        weights_bytes: &[u8],
        tokenizer_json: &str,
    ) -> Result<AsrSession, JsValue> {
        let use_gpu = config.use_gpu;
        let language = config.language.clone();
        
        // 1. 初始化 VocabProcessor
        let vocab = VocabProcessor::new(tokenizer_json, None, None)
            .map_err(|e| JsValue::from_str(&e))?;
            
        let blank_id = vocab.get_blank_id();
        let prompt_id = vocab.get_prompt_id(&language).unwrap_or(blank_id); // 預設使用 blank 作为 SOS

        #[cfg(feature = "gpu")]
        if use_gpu {
            // 初始化 WebGPU 後端
            let device = burn_wgpu::WgpuDevice::default();
            // 設定 24 層 Conformer，1024 隱維度，8 頭注意力
            let mut model = RNNTModel::<burn_wgpu::Wgpu>::new(
                vocab.get_blank_id() as usize + 1, // vocab size
                1024,                              // hidden size
                24,                                // num layers
                8,                                 // num heads
                56,                                // left context
                1024,                              // pred hidden
                1024,                              // joiner hidden
                &device,
            );
            
            // 載入 SafeTensors
            model.load_from_safetensors(weights_bytes, &device)
                .map_err(|e| JsValue::from_str(&format!("GPU 模型載入權重失敗: {}", e)))?;
                
            let audio_processor = AudioProcessor::new();
            let decoder = RnntGreedyDecoder::new(blank_id, prompt_id);
            let encoder_cache = EncoderCache::new(24);
            
            let session_impl = SessionImpl {
                model,
                audio_processor,
                vocab_processor: vocab,
                decoder,
                encoder_cache,
                language_prompt_id: prompt_id,
            };
            
            return Ok(AsrSession {
                session: ActiveSession::Wgpu(session_impl),
            });
        }
        // 初始化 CPU 後端
        let device = burn_flex::FlexDevice::default();
        let mut model = RNNTModel::<burn_flex::Flex>::new(
            vocab.get_blank_id() as usize + 1,
            1024,
            24,
            8,
            56,
            1024,
            1024,
            &device,
        );
        
        model.load_from_safetensors(weights_bytes, &device)
            .map_err(|e| JsValue::from_str(&format!("CPU 模型載入權重失敗: {}", e)))?;
            
        let audio_processor = AudioProcessor::new();
        let decoder = RnntGreedyDecoder::new(blank_id, prompt_id);
        let encoder_cache = EncoderCache::new(24);
        
        let session_impl = SessionImpl {
            model,
            audio_processor,
            vocab_processor: vocab,
            decoder,
            encoder_cache,
            language_prompt_id: prompt_id,
        };
        
        Ok(AsrSession {
            session: ActiveSession::Cpu(session_impl),
        })
    }

    /// 輸入音訊訊號，進行即時特徵提取與流式語音解碼，返回當前新增的文字結果
    #[wasm_bindgen]
    pub fn feed_audio(&mut self, pcm_samples: &[f32], sample_rate: f32) -> Result<String, JsValue> {
        match &mut self.session {
            #[cfg(feature = "gpu")]
            ActiveSession::Wgpu(s) => {
                // 1. 特徵提取
                let new_mel_frames = s.audio_processor.process_audio(pcm_samples, sample_rate);
                if new_mel_frames.is_empty() {
                    return Ok(String::new());
                }
                
                // 2. 轉換為 Burn Tensor [1, 1, L_new, 80]
                let num_frames = new_mel_frames.len();
                let flat_mel: Vec<f32> = new_mel_frames.into_iter().flatten().collect();
                let device = burn_wgpu::WgpuDevice::default();
                let mel_tensor = Tensor::<burn_wgpu::Wgpu, 4>::from_data(
                    burn::tensor::TensorData::new(flat_mel, [1, 1, num_frames, 80]),
                    &device,
                );
                
                // 3. 執行 Encoder
                let (enc_out, next_cache) = s.model.encoder.forward(mel_tensor, s.encoder_cache.clone());
                s.encoder_cache = next_cache;
                
                // 4. 解碼出 Token ID
                let new_tokens = s.decoder.decode(&s.model, enc_out, &device);
                
                // 5. 轉換為字串文字
                let decoded_text = s.vocab_processor.decode(&new_tokens);
                Ok(decoded_text)
            }
            ActiveSession::Cpu(s) => {
                let new_mel_frames = s.audio_processor.process_audio(pcm_samples, sample_rate);
                if new_mel_frames.is_empty() {
                    return Ok(String::new());
                }
                
                let num_frames = new_mel_frames.len();
                let flat_mel: Vec<f32> = new_mel_frames.into_iter().flatten().collect();
                let device = burn_flex::FlexDevice::default();
                let mel_tensor = Tensor::<burn_flex::Flex, 4>::from_data(
                    burn::tensor::TensorData::new(flat_mel, [1, 1, num_frames, 80]),
                    &device,
                );
                
                let (enc_out, next_cache) = s.model.encoder.forward(mel_tensor, s.encoder_cache.clone());
                s.encoder_cache = next_cache;
                
                let new_tokens = s.decoder.decode(&s.model, enc_out, &device);
                let decoded_text = s.vocab_processor.decode(&new_tokens);
                Ok(decoded_text)
            }
        }
    }

    /// 重設解碼與緩衝狀態，清除 KV Cache 以開始一段新的識別語音
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        match &mut self.session {
            #[cfg(feature = "gpu")]
            ActiveSession::Wgpu(s) => {
                s.audio_processor.reset();
                s.decoder.reset(s.language_prompt_id);
                s.encoder_cache = EncoderCache::new(24);
            }
            ActiveSession::Cpu(s) => {
                s.audio_processor.reset();
                s.decoder.reset(s.language_prompt_id);
                s.encoder_cache = EncoderCache::new(24);
            }
        }
    }
}

// -------------------------------------------------------------
// 保留模板預設導出，以防 template 運行測試報錯
// -------------------------------------------------------------

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[wasm_bindgen]
extern "C" {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}
