# Nemotron-3.5 ASR Streaming WebGPU 實作計畫

本計畫旨在實現一個基於 WebGPU 的流式語音識別（ASR）npm 模組。我們將使用 Rust + Burn 框架實現底層的 **Cache-Aware FastConformer-RNNT** 推理，並透過 WebAssembly (wasm-bindgen) 將其導出為一個易用的 JavaScript API。

## 使用者審查項目

> [!IMPORTANT]
> - **模型體積與載入時間**：`nemotron-3.5-asr-streaming-0.6b` 參數大小約 600M。若使用 fp16 權重，大小約為 1.2GB；若使用 int4 量化，大小約為 300MB。在瀏覽器端，首次載入可能需要數十秒至數分鐘。SDK 將內建下載進度回呼與 Cache API 緩存機制。
> - **Tokenizer 的 Wasm 編譯**：我們將在 Rust 內部直接使用 `tokenizers` 庫或 `sentencepiece-native` 來解析原始的 `tokenizer.model`，確保解碼行為與 NeMo 官方完全一致。
> - **硬體要求**：預設使用 WebGPU。若瀏覽器不支援 WebGPU 或使用者手動停用，SDK 將降級到 CPU 推理（基於 `burn-flex`，但速度會顯著變慢）。

## 開放性問題

目前無懸而未決的核心問題。我們將根據使用者的反饋，採用基於狀態快取（KV Cache）的嚴格流式推理與內建音訊重採樣器的方案。

---

## 提案變更

### 1. 權重轉換工具

#### [NEW] [convert_nemo_weights.py](file:///home/wsl/nemotron-35-asr-streaming-webgpu/tools/convert_nemo_weights.py)
- 提供一個 Python 腳本，用來載入 NeMo 格式的 `.nemo` 檔案（或從 Hugging Face 下載的權重與 config.yaml）。
- 將權重解壓並導出為標準的 SafeTensors 格式，以便 Rust Burn 可以直接載入。
- 將 `tokenizer.model` 或 `tokenizer.json` 複製到發布目錄。

### 2. 音訊特徵處理模組

#### [NEW] [audio.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/audio.rs)
- **重採樣器 (Resampler)**：實現一個輕量級的線性/三次插值重採樣器，將任意輸入採樣率（如 44.1kHz, 48kHz）轉換為模型要求的 16kHz。
- **Mel-spectrogram 提取器**：基於 `rustfft` 實現 STFT，計算 Hann 視窗、80 維 Mel-filterbank，並進行 Log-mel 特徵提取與 Streaming Normalization。它會維護一個邊緣音訊緩衝區，確保流式輸入時特徵提取的連續性。

### 3. 模型結構實現 (Rust + Burn)

我們將使用 Burn 框架重新構建 Nemotron-3.5 ASR 的 RNNT 結構：

#### [NEW] [subsampling.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/model/subsampling.rs)
- 實現 FastConformer 採用的 Depthwise Separable Convolution 2D 下採樣模組，並維護卷積的歷史狀態緩衝（Conv Cache）。

#### [NEW] [attention.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/model/attention.rs)
- 實現 **Cache-Aware Self-Attention**：在計算 Multi-Head Attention 時，利用輸入的 KV Cache 進行拼接，限制 attention 的 context window（左側 context 與右側 context），並將更新後的 Key 與 Value 作為新 Cache 傳出。
- 支持相對位置偏差 (Relative Position Bias / RoPE)。

#### [NEW] [conformer.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/model/conformer.rs)
- 拼裝 Conformer Block：Feed-Forward Module 1 -> Self-Attention Module -> Convolution Module -> Feed-Forward Module 2。每個 Block 都會接收並返回對應的 Cache 狀態。

#### [NEW] [rnnt.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/model/rnnt.rs)
- **Encoder**：將 Subsampling 與多個 Conformer Blocks 串聯。
- **Stateless Predictor**：基於 Embedding 和 Linear 層實現無狀態預測器，以最近 1-2 個 Token ID 作為歷史輸入。
- **Joint Network**：將 Encoder 的音訊表徵與 Predictor 的文本表徵結合（一般是相加後經過 ReLU 和 Linear），預測詞表機率。

### 4. 解碼與 API 介面

#### [NEW] [decoder.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/decoder.rs)
- 實現 **RNN-T Greedy Search** 算法，在流式時間步中，循環預測 token 直到輸出 blank，並輸出非 blank 的 token 作為識別字元。

#### [NEW] [vocab.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/vocab.rs)
- 封裝 Rust `tokenizers` 庫，載入 `tokenizer.json`，處理 Token ID 與字串之間的相互轉換與 Prompt 建構。

#### [MODIFY] [lib.rs](file:///home/wsl/nemotron-35-asr-streaming-webgpu/src/lib.rs)
- 定義並匯出 `AsrSession` 與 `AsrSessionConfig`。
- 管理狀態生命週期：包含音訊緩衝區、下採樣緩衝區、KV Cache、已解碼 Token 歷史。
- 提供 `feed_audio` 方法供 JavaScript 呼叫。

#### [MODIFY] [index.html](file:///home/wsl/nemotron-35-asr-streaming-webgpu/index.html)
- 提供一個美觀、高質感的調試/演示前端。
- 支援透過瀏覽器 `getUserMedia` 進行實時麥克風錄音。
- 展示模型下載進度條、WebGPU 狀態、以及流式聽寫出的繁體中文/英文內容。

---

## 驗證計畫

### 自動化測試
- 在 Rust 中編寫單元測試以驗證：
  - 重採樣器與 Mel-spectrogram 提取的數值正確性（與 Python scipy/librosa 對比）。
  - Subsampling 與 Attention 的 Cache 機制在流式與非流式（Batch）下推理結果的一致性。
  ```bash
  cargo test
  ```

### 手動驗證
1. 運行 Python 轉換腳本，將 Hugging Face 的 `nvidia/nemotron-3.5-asr-streaming-0.6b` 權重轉為 SafeTensors，並生成 Tokenizer 設定。
2. 啟動一個本地 Web 伺服器載入 `index.html`。
3. 在瀏覽器中點擊「開始錄音」，對著麥克風說話，觀察 WebGPU 的初始化進度，並檢查流式輸出的識別文字是否準確。
