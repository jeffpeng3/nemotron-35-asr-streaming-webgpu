use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
pub struct RelPositionMultiHeadAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub pos_proj: Linear<B>,
    
    // 全局參數偏置，用於相對位置運算
    pub pos_bias_u: Tensor<B, 2>, // [num_heads, head_dim]
    pub pos_bias_v: Tensor<B, 2>, // [num_heads, head_dim]
    
    pub num_heads: usize,
    pub head_dim: usize,
    pub left_context: usize,      // 左側歷史長度，例如 56
}

impl<B: Backend> RelPositionMultiHeadAttention<B> {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        left_context: usize,
        device: &B::Device,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        
        let q_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let k_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let v_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let out_proj = LinearConfig::new(hidden_size, hidden_size).init(device);
        let pos_proj = LinearConfig::new(hidden_size, hidden_size).with_bias(false).init(device);
        
        // 初始化偏置為零，之後會載入 PyTorch 導出的值
        let pos_bias_u = Tensor::zeros([num_heads, head_dim], device);
        let pos_bias_v = Tensor::zeros([num_heads, head_dim], device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            pos_proj,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
            left_context,
        }
    }

    /// 帶有 KV Cache 狀態的流式前向傳播
    /// x: 當前輸入特徵 [Batch, Length_new, hidden_size]
    /// pos_emb: 相對位置正弦編碼 [1, 2 * max_len - 1, hidden_size]
    /// cache_k: 歷史 Key 快取 [Batch, Length_cache, hidden_size] (可選)
    /// cache_v: 歷史 Value 快取 [Batch, Length_cache, hidden_size] (可選)
    /// 返回: (輸出特徵 [Batch, Length_new, hidden_size], 新 Key 快取, 新 Value 快取)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        cache_k: Option<Tensor<B, 3>>,
        cache_v: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [batch_size, len_new, _hidden] = x.shape().dims::<3>();
        
        // 1. Q, K, V 投影
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // 2. 拼接歷史快取
        let (k_total, v_total) = match (cache_k, cache_v) {
            (Some(ck), Some(cv)) => {
                let kt = Tensor::cat(vec![ck, k], 1);
                let vt = Tensor::cat(vec![cv, v], 1);
                (kt, vt)
            }
            _ => (k, v),
        };

        // 3. 裁剪快取以限制左側上下文長度 (left_context)
        let [_, total_len, _] = k_total.shape().dims::<3>();
        let (k_cached, v_cached) = if total_len > self.left_context {
            let start = total_len - self.left_context;
            let kc = k_total.clone().slice([0..batch_size, start..total_len]);
            let vc = v_total.clone().slice([0..batch_size, start..total_len]);
            (kc, vc)
        } else {
            (k_total.clone(), v_total.clone())
        };

        // 4. 重塑為多頭注意力格式 [Batch, num_heads, Length, head_dim]
        let q = self.reshape_heads(q, batch_size, len_new);
        let k_t = self.reshape_heads(k_total.clone(), batch_size, total_len);
        let v_t = self.reshape_heads(v_total.clone(), batch_size, total_len);

        // 5. 計算經典 Attention 部分 AC = Q @ K^T
        // q Shape: [B, H, L_new, D_head]
        // k_t Shape: [B, H, L_total, D_head]
        let k_t_transposed = k_t.clone().swap_dims(2, 3); // -> [B, H, D_head, L_total]
        
        // q_bias_u = q + pos_bias_u.unsqueeze(0, 2) -> [1, H, 1, D_head]
        let pos_bias_u_expanded = self.pos_bias_u.clone()
            .reshape([1, self.num_heads, 1, self.head_dim]);
        let q_u = q.clone() + pos_bias_u_expanded;
        let ac = q_u.matmul(k_t_transposed); // -> [B, H, L_new, L_total]

        // 6. 計算相對位置 Attention 部分 BD = Q @ R^T
        // pos_emb 是相對位置的正弦嵌入，我們將其通過 pos_proj 投影
        let pos_emb_len = pos_emb.shape().dims::<3>()[1];
        let p = self.pos_proj.forward(pos_emb); // -> [1, 2*max_len-1, hidden_size]
        let p = self.reshape_heads(p, 1, pos_emb_len); // -> [1, H, 2*max_len-1, D_head]
        let p_transposed = p.swap_dims(2, 3); // -> [1, H, D_head, 2*max_len-1]
        
        let pos_bias_v_expanded = self.pos_bias_v.clone()
            .reshape([1, self.num_heads, 1, self.head_dim]);
        let q_v = q + pos_bias_v_expanded;
        let bd = q_v.matmul(p_transposed); // -> [B, H, L_new, 2*max_len-1]
        
        // 7. 將相對位置坐標變換（Relative Shift）映射到 [B, H, L_new, L_total]
        // 這裡實現相對偏移 (Relative Shift) 的索引變換
        let bd_shifted = self.relative_shift(bd, len_new, total_len);

        // 8. 融合 AC 與 BD 得分，並縮放
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = (ac + bd_shifted) * scale;

        // 9. Softmax 與 Value 點乘
        let attn_weights = burn::tensor::activation::softmax(scores, 3);
        let context = attn_weights.matmul(v_t); // -> [B, H, L_new, D_head]

        // 10. 還原頭維度並進行 Out 投影
        let context = context.swap_dims(1, 2); // -> [B, L_new, H, D_head]
        let context = context.reshape([batch_size, len_new, self.num_heads * self.head_dim]);
        let output = self.out_proj.forward(context);

        (output, k_cached, v_cached)
    }

    fn reshape_heads(&self, x: Tensor<B, 3>, batch_size: usize, length: usize) -> Tensor<B, 4> {
        let x = x.reshape([batch_size, length, self.num_heads, self.head_dim]);
        x.swap_dims(1, 2) // -> [Batch, num_heads, Length, head_dim]
    }

    /// 相對偏移運算 (Relative Shift)，將 2*L-1 的相對得分映射回長度對齊的對角矩陣
    fn relative_shift(&self, x: Tensor<B, 4>, len_q: usize, len_k: usize) -> Tensor<B, 4> {
        let [batch, heads, _, len_p] = x.shape().dims::<4>();
        
        // NeMo 典型的 relative_shift 實現：
        // 1. Pad 一列零
        // 2. Flatten，reshape，再 slice 取得有效的對角子矩陣
        // 為了避免複雜的 ndarray 操作，我們在 Tensor 級別使用 slice 技巧：
        // 在流式中，相對於當前 Query (長度 len_q)，Key_total (長度 len_k) 的相對偏移位置是固定的。
        // 當前時間步的 index 為 t，與歷史的相對距離範圍是 [t - left_context, t]。
        // 所以我們可以直接根據當前相對坐標，在 p_transposed (即維度 3) 上進行切片。
        // 這裡我們提供一個標準的 NeMo 偏移切片：
        // 由於相對位置 R 的定義中，中間點是 0 偏移，右邊是正，左邊是負。
        // NeMo 中相對位置向量通常是從 -max_len 到 +max_len。
        // 故相對偏置中，Query i 與 Key j 的相對 index 為 (len_k - 1) - j + i。
        // 我們可以直接在維度 3 上取出這一區間：
        // 對於每一個 query step i (0..len_q)，其對應的 key slice 區間為 [(len_k - 1) + i - (len_k - 1) .. (len_k - 1) + i + 1] 即 [i .. i + len_k]。
        // 我們可以在這裡使用循環 slice 拼接，或者使用 tensor 矩陣變換。
        // 由於 len_q 往往很小（流式下為 1 或 2 幀），我們直接按 query step 循環 slice 拼接：
        let mut slices = Vec::with_capacity(len_q);
        let zero_point = len_p / 2; // 對應相對位置 0 (即 query 與 key 對齊的起點)
        
        for i in 0..len_q {
            // 計算當前 query 幀 i 與整個 key_total 的相對偏置區間
            // 對於 key_idx 在 0..len_k，相對位置為 key_idx - (total_len - 1 - len_q + 1 + i) ...
            // 簡化來說，相對 index 的起點是 zero_point - (len_k - 1) + i
            // 我們從 bd 的維度 3 取出長度為 len_k 的子張量
            let start = (zero_point + i).saturating_sub(len_k - 1);
            let end = start + len_k;
            let slice = x.clone().slice([0..batch, 0..heads, i..i+1, start..end]);
            slices.push(slice);
        }
        
        Tensor::cat(slices, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use burn::tensor::Distribution;

    type TestBackend = burn_flex::Flex;

    /// 測試無快取情況下的注意力輸出形狀
    /// hidden_size=64, num_heads=4, left_context=8
    /// 輸入 x=[1,4,64], pos_emb=[1,7,64] (2*4-1=7)
    /// 預期輸出 [1,4,64]
    #[test]
    #[wasm_bindgen_test]
    fn test_attention_output_shape_no_cache() {
        let device = Default::default();
        let attn = RelPositionMultiHeadAttention::<TestBackend>::new(64, 4, 8, &device);
        let x = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, &device);
        let pos_emb = Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Default, &device);

        let (out, cache_k, cache_v) = attn.forward(x, pos_emb, None, None);

        let [ob, ol, oh] = out.shape().dims::<3>();
        assert_eq!([ob, ol, oh], [1, 4, 64], "輸出形狀應為 [1, 4, 64]");
        // 無快取時，cache_k/cache_v 長度應等於 len_new=4（因 4 < left_context=8）
        let [ckb, ckl, ckh] = cache_k.shape().dims::<3>();
        assert_eq!(ckb, 1);
        assert_eq!(ckl, 4);
        assert_eq!(ckh, 64);
        let [cvb, cvl, cvh] = cache_v.shape().dims::<3>();
        assert_eq!([cvb, cvl, cvh], [ckb, ckl, ckh]);
    }

    /// 測試帶有快取的注意力前向傳播
    /// 先執行一次取得 cache_k/cache_v，再用新輸入與快取執行第二次
    #[test]
    #[wasm_bindgen_test]
    fn test_attention_with_cache() {
        let device = Default::default();
        let attn = RelPositionMultiHeadAttention::<TestBackend>::new(64, 4, 8, &device);

        // 第一次前向（無快取）
        let x1 = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, &device);
        let pos1 = Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Default, &device);
        let (out1, ck1, cv1) = attn.forward(x1, pos1, None, None);
        let [o1b, o1l, o1h] = out1.shape().dims::<3>();
        assert_eq!([o1b, o1l, o1h], [1, 4, 64]);

        // 第二次前向（帶快取）
        let x2 = Tensor::<TestBackend, 3>::random([1, 2, 64], Distribution::Default, &device);
        // total_len = cache_len(4) + new_len(2) = 6, pos_emb 需 2*6-1=11
        let pos2 = Tensor::<TestBackend, 3>::random([1, 11, 64], Distribution::Default, &device);
        let (out2, ck2, cv2) = attn.forward(x2, pos2, Some(ck1), Some(cv1));

        // 輸出長度應等於新輸入長度 2
        let [o2b, o2l, o2h] = out2.shape().dims::<3>();
        assert_eq!([o2b, o2l, o2h], [1, 2, 64], "帶快取時輸出應為 [1, 2, 64]");
        // 快取長度應為 min(total_len=6, left_context=8) = 6
        assert_eq!(ck2.shape().dims::<3>()[1], 6);
        assert_eq!(cv2.shape().dims::<3>()[1], 6);
    }

    /// 測試快取裁剪：left_context=4 時，累積快取超過 4 應被裁剪
    #[test]
    #[wasm_bindgen_test]
    fn test_attention_cache_trimming() {
        let device = Default::default();
        let attn = RelPositionMultiHeadAttention::<TestBackend>::new(64, 4, 4, &device);

        // 第一次：len_new=4，total=4，不超過 left_context=4
        let x1 = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, &device);
        let pos1 = Tensor::<TestBackend, 3>::random([1, 7, 64], Distribution::Default, &device);
        let (_out1, ck1, cv1) = attn.forward(x1, pos1, None, None);
        assert_eq!(ck1.shape().dims::<3>()[1], 4);

        // 第二次：len_new=4，total=4+4=8，超過 left_context=4 → 裁剪至 4
        let x2 = Tensor::<TestBackend, 3>::random([1, 4, 64], Distribution::Default, &device);
        let pos2 = Tensor::<TestBackend, 3>::random([1, 15, 64], Distribution::Default, &device);
        let (_out2, ck2, cv2) = attn.forward(x2, pos2, Some(ck1), Some(cv1));

        assert_eq!(
            ck2.shape().dims::<3>()[1], 4,
            "快取應被裁剪至 left_context=4"
        );
        assert_eq!(cv2.shape().dims::<3>()[1], 4);
    }
}
