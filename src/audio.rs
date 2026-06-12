use rustfft::{FftPlanner, num_complex::Complex};

pub struct AudioProcessor {
    target_sample_rate: f32,
    window_size: usize,      // 400
    hop_size: usize,         // 160
    fft_size: usize,         // 512
    n_mels: usize,           // 80
    
    // 歷史狀態緩衝
    audio_buffer: Vec<f32>,  // 儲存重採樣後的音訊
    last_sample: f32,        // 用於預加重 (pre-emphasis) 的跨幀快取
    mel_fb: Vec<Vec<f32>>,   // 80 x 257 (fft_size/2 + 1) Mel 濾波器矩陣
    hann_window: Vec<f32>,   // 長度 400 的 Hann 窗
    
    fft_planner: FftPlanner<f32>,
}

impl AudioProcessor {
    pub fn new() -> Self {
        let target_sample_rate = 16000.0;
        let window_size = 400;
        let hop_size = 160;
        let fft_size = 512;
        let n_mels = 80;
        
        let mel_fb = Self::generate_mel_filterbank(
            n_mels, 
            fft_size, 
            target_sample_rate, 
            0.0, 
            8000.0
        );
        let hann_window = Self::generate_hann_window(window_size);
        
        Self {
            target_sample_rate,
            window_size,
            hop_size,
            fft_size,
            n_mels,
            audio_buffer: Vec::new(),
            last_sample: 0.0,
            mel_fb,
            hann_window,
            fft_planner: FftPlanner::new(),
        }
    }

    /// 重設音訊處理狀態（用於開啟新的 ASR session）
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.last_sample = 0.0;
    }

    /// 線性重採樣：將任意採樣率轉換成 16kHz
    fn resample(&self, input: &[f32], input_sr: f32) -> Vec<f32> {
        if (input_sr - self.target_sample_rate).abs() < 1.0 {
            return input.to_vec();
        }
        
        let ratio = input_sr / self.target_sample_rate;
        let num_output_samples = (input.len() as f32 / ratio).floor() as usize;
        let mut output = Vec::with_capacity(num_output_samples);
        
        for i in 0..num_output_samples {
            let src_idx = i as f32 * ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len() - 1);
            let weight = src_idx - src_idx.floor();
            
            let val = (1.0 - weight) * input[low] + weight * input[high];
            output.push(val);
        }
        output
    }

    /// 生成 Hann 視窗
    fn generate_hann_window(window_size: usize) -> Vec<f32> {
        let mut window = vec![0.0; window_size];
        for i in 0..window_size {
            window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos());
        }
        window
    }

    /// Slaney 刻度轉換：Hz 到 Mel
    fn hz_to_mel(hz: f32) -> f32 {
        if hz < 1000.0 {
            hz / (200.0 / 3.0)
        } else {
            15.0 + (hz / 1000.0).ln() / (1.0_f32 + 0.5 / 8.0).ln()
        }
    }

    /// Slaney 刻度轉換：Mel 到 Hz
    fn mel_to_hz(mel: f32) -> f32 {
        if mel < 15.0 {
            mel * (200.0 / 3.0)
        } else {
            1000.0 * ((mel - 15.0) * (1.0_f32 + 0.5 / 8.0).ln()).exp()
        }
    }

    /// 生成 Librosa/NeMo 對齊的 Slaney Mel-filterbank 矩陣 (n_mels x (fft_size/2 + 1))
    fn generate_mel_filterbank(
        n_mels: usize,
        fft_size: usize,
        sample_rate: f32,
        f_min: f32,
        f_max: f32,
    ) -> Vec<Vec<f32>> {
        let num_bins = fft_size / 2 + 1; // 257
        let min_mel = Self::hz_to_mel(f_min);
        let max_mel = Self::hz_to_mel(f_max);
        
        // 生成包含起點、中點、終點的 mel 等分刻度
        let mut mel_pts = vec![0.0; n_mels + 2];
        let step = (max_mel - min_mel) / (n_mels + 1) as f32;
        for i in 0..(n_mels + 2) {
            mel_pts[i] = min_mel + i as f32 * step;
        }
        
        // 轉回頻率 (Hz)
        let mut hz_pts = vec![0.0; n_mels + 2];
        for i in 0..(n_mels + 2) {
            hz_pts[i] = Self::mel_to_hz(mel_pts[i]);
        }
        
        // 映射到 FFT bin 索引
        let mut bin_pts = vec![0.0; n_mels + 2];
        for i in 0..(n_mels + 2) {
            bin_pts[i] = (fft_size + 1) as f32 * hz_pts[i] / sample_rate;
        }
        
        let mut fb = vec![vec![0.0; num_bins]; n_mels];
        
        for m in 0..n_mels {
            let b_left = bin_pts[m];
            let b_center = bin_pts[m + 1];
            let b_right = bin_pts[m + 2];
            
            // 每個三角濾波器的斜率計算
            for k in 0..num_bins {
                let k_f = k as f32;
                if k_f >= b_left && k_f <= b_center {
                    fb[m][k] = (k_f - b_left) / (b_center - b_left);
                } else if k_f > b_center && k_f <= b_right {
                    fb[m][k] = (b_right - k_f) / (b_right - b_center);
                }
            }
            
            // Slaney 面積歸一化 (Area Normalization)
            let enorm = 2.0 / (hz_pts[m + 2] - hz_pts[m]);
            for k in 0..num_bins {
                fb[m][k] *= enorm;
            }
        }
        fb
    }

    /// 提取 Log-mel Spectrogram 特徵
    /// samples: 輸入的原始音訊幀 (Float32Array)
    /// input_sr: 原始音訊的採樣率
    /// 返回: 一個 2D Vec, 形狀為 [num_new_frames, 80]，代表新增的時間步及其對應的 Mel 特徵
    pub fn process_audio(&mut self, samples: &[f32], input_sr: f32) -> Vec<Vec<f32>> {
        // 1. 重採樣至 16kHz 并拼入緩衝區
        let resampled = self.resample(samples, input_sr);
        self.audio_buffer.extend(resampled);
        
        let mut new_frames = Vec::new();
        let fft = self.fft_planner.plan_fft_forward(self.fft_size);
        
        // 2. 當緩衝區長度充足時，以 window_size (400) 與 hop_size (160) 切分幀
        while self.audio_buffer.len() >= self.window_size {
            // 取出當前幀的音訊樣本
            let frame_samples = &self.audio_buffer[0..self.window_size];
            
            // 3. 預加重 (Pre-emphasis) 與 Hann 窗計算
            // NeMo 公式: y[t] = x[t] - 0.97 * x[t-1]
            let mut windowed = vec![Complex { re: 0.0, im: 0.0 }; self.fft_size];
            
            let mut prev = self.last_sample;
            for t in 0..self.window_size {
                let current = frame_samples[t];
                let emphasized = current - 0.97 * prev;
                windowed[t] = Complex {
                    re: emphasized * self.hann_window[t],
                    im: 0.0,
                };
                prev = current;
            }
            // 更新跨幀狀態（為下一幀保留當前窗 hop_size 的邊界樣本）
            // 注意，NeMo 流式處理中，預加重的 last_sample 是上一個 hop 邊界的最後一個 sample
            // 由於 hop 之後我們會把 buffer 前進 160，下一幀的 window 將從 index 160 開始。
            // 故上一幀的 window 裡 index 159 的 sample，就是下一幀的前一個 sample。
            self.last_sample = frame_samples[self.hop_size - 1];
            
            // 4. 傅立葉變換 (FFT) 
            fft.process(&mut windowed);
            
            // 5. 計算前 257 個點的 Power Spectrogram (模平方)
            let num_bins = self.fft_size / 2 + 1;
            let mut power_spec = vec![0.0; num_bins];
            for k in 0..num_bins {
                power_spec[k] = windowed[k].norm_sqr();
            }
            
            // 6. 點乘 Mel Filterbank，計算 Mel 能量，並取對數
            let mut mel_energy = vec![0.0; self.n_mels];
            for m in 0..self.n_mels {
                let mut sum = 0.0;
                for k in 0..num_bins {
                    sum += power_spec[k] * self.mel_fb[m][k];
                }
                // Log-mel 歸一化
                mel_energy[m] = (sum.max(1e-5)).ln();
            }
            
            new_frames.push(mel_energy);
            
            // 7. 前進 hop_size (160)
            self.audio_buffer.drain(0..self.hop_size);
        }
        
        new_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    // ──────────────────────────────────────────────
    // 1. Hann 窗性質檢查
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_hann_window_properties() {
        let n = 400;
        let win = AudioProcessor::generate_hann_window(n);

        // 長度正確
        assert_eq!(win.len(), n);

        // 首尾應接近 0（Hann 窗的定義：w(0)=0, w(N-1)=0）
        assert!(win[0].abs() < 1e-6, "窗的第一個元素應為 0，實際 = {}", win[0]);
        assert!(win[n - 1].abs() < 1e-6, "窗的最後一個元素應為 0，實際 = {}", win[n - 1]);

        // 峰值應出現在中點附近（index ≈ (N-1)/2），且峰值為 1.0
        let mid = (n - 1) / 2; // 199
        assert!(
            (win[mid] - 1.0).abs() < 0.01,
            "窗的中點值應接近 1.0，實際 = {}",
            win[mid]
        );

        // 所有值介於 [0, 1]
        for (i, &v) in win.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-6, "win[{}] = {} 超出 [0, 1]", i, v);
        }
    }

    // ──────────────────────────────────────────────
    // 2. Hz ↔ Mel 往返一致性（Slaney 刻度）
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_hz_mel_roundtrip() {
        let test_freqs = [0.0_f32, 500.0, 1000.0, 4000.0, 8000.0];
        for &hz in &test_freqs {
            let mel = AudioProcessor::hz_to_mel(hz);
            let hz_back = AudioProcessor::mel_to_hz(mel);
            assert!(
                (hz - hz_back).abs() < 0.5,
                "Hz→Mel→Hz 往返失敗: 原始 Hz={}, 轉回 Hz={}",
                hz,
                hz_back
            );
        }
    }

    // ──────────────────────────────────────────────
    // 3. Mel 濾波器矩陣形狀 [80][257]
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_mel_filterbank_shape() {
        let fb = AudioProcessor::generate_mel_filterbank(80, 512, 16000.0, 0.0, 8000.0);

        assert_eq!(fb.len(), 80, "應有 80 個 Mel 帶");
        for (m, band) in fb.iter().enumerate() {
            assert_eq!(band.len(), 257, "Mel 帶 {} 應有 257 個 bin", m);
        }
    }

    // ──────────────────────────────────────────────
    // 4. 每個 Mel 帶至少有一個非零 bin
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_mel_filterbank_nonzero() {
        let fb = AudioProcessor::generate_mel_filterbank(80, 512, 16000.0, 0.0, 8000.0);

        for (m, band) in fb.iter().enumerate() {
            let has_nonzero = band.iter().any(|&v| v.abs() > 1e-10);
            assert!(has_nonzero, "Mel 帶 {} 全部為 0，不符合預期", m);
        }
    }

    // ──────────────────────────────────────────────
    // 5. 已是 16kHz 的輸入不應被改變
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_resample_identity() {
        let proc = AudioProcessor::new();
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();

        let output = proc.resample(&input, 16000.0);

        assert_eq!(output.len(), input.len(), "相同採樣率應返回相同長度");
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample[{}] 不一致: {} vs {}",
                i, a, b
            );
        }
    }

    // ──────────────────────────────────────────────
    // 6. 48kHz→16kHz 降採樣，輸出長度約為 1/3
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_resample_downsample() {
        let proc = AudioProcessor::new();
        let n_in = 4800;
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.001).sin()).collect();

        let output = proc.resample(&input, 48000.0);

        let expected_len = (n_in as f32 / 3.0).floor() as usize; // 1600
        assert_eq!(
            output.len(),
            expected_len,
            "48kHz→16kHz: 期望 {} 個樣本，實際 {}",
            expected_len,
            output.len()
        );
    }

    // ──────────────────────────────────────────────
    // 7. process_audio 輸出形狀：800 樣本 @ 16kHz → ≥2 幀，每幀 80 Mel
    //    第一幀消耗 400 樣本，第二幀再消耗 hop=160 → 共需 560，
    //    800 夠切出 ≥2 幀（(800-400)/160 + 1 = 3.5 → 3 幀）。
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_process_audio_output_shape() {
        let mut proc = AudioProcessor::new();
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.01).sin()).collect();

        let frames = proc.process_audio(&samples, 16000.0);

        assert!(
            frames.len() >= 2,
            "800 樣本應至少產生 2 幀，實際 {} 幀",
            frames.len()
        );
        for (t, frame) in frames.iter().enumerate() {
            assert_eq!(
                frame.len(),
                80,
                "幀 {} 應有 80 個 Mel 特徵，實際 {}",
                t,
                frame.len()
            );
        }
    }

    // ──────────────────────────────────────────────
    // 8. 串流連續性：分兩次各送 400 樣本，第二次也應產生幀
    //    第一次：400 樣本恰好切出 1 幀（window_size=400），
    //           緩衝區剩餘 400 - 160 = 240。
    //    第二次：240 + 400 = 640 → (640-400)/160 + 1 = 2.5 → 2 幀。
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_process_audio_streaming() {
        let mut proc = AudioProcessor::new();
        let chunk: Vec<f32> = (0..400).map(|i| (i as f32 * 0.01).sin()).collect();

        // 第一次呼叫
        let frames1 = proc.process_audio(&chunk, 16000.0);
        assert!(
            !frames1.is_empty(),
            "第一批 400 樣本應能產生至少 1 幀"
        );

        // 第二次呼叫（測試 buffer 連續性）
        let frames2 = proc.process_audio(&chunk, 16000.0);
        assert!(
            !frames2.is_empty(),
            "第二批 400 樣本（加上殘留 buffer）應能產生至少 1 幀"
        );

        // 每一幀都應該是 80 維
        for frame in frames1.iter().chain(frames2.iter()) {
            assert_eq!(frame.len(), 80);
        }
    }

    // ──────────────────────────────────────────────
    // 9. reset() 應清空緩衝區與狀態
    // ──────────────────────────────────────────────
    #[test]
    #[wasm_bindgen_test]
    fn test_reset_clears_state() {
        let mut proc = AudioProcessor::new();
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.01).sin()).collect();

        // 先處理一些音訊，讓 buffer 與 last_sample 非空/非零
        let _ = proc.process_audio(&samples, 16000.0);
        // 此時 audio_buffer 可能還有殘留樣本

        // 重設
        proc.reset();
        assert!(
            proc.audio_buffer.is_empty(),
            "reset 後 audio_buffer 應為空"
        );
        assert!(
            proc.last_sample.abs() < 1e-10,
            "reset 後 last_sample 應為 0"
        );

        // reset 後再送入相同資料，應得到與全新 processor 一樣的結果
        let mut fresh_proc = AudioProcessor::new();
        let frames_after_reset = proc.process_audio(&samples, 16000.0);
        let frames_fresh = fresh_proc.process_audio(&samples, 16000.0);

        assert_eq!(
            frames_after_reset.len(),
            frames_fresh.len(),
            "reset 後的幀數應與全新 processor 一致"
        );
        for (t, (a, b)) in frames_after_reset.iter().zip(frames_fresh.iter()).enumerate() {
            for (k, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < 1e-4,
                    "reset 後幀[{}][{}] 不一致: {} vs {}",
                    t, k, va, vb
                );
            }
        }
    }
}
