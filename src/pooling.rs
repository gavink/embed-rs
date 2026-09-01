//! Pooling + L2 normalization for BERT hidden states.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Mask-aware mean pool over the sequence dimension, then L2-normalize.
///
/// `hidden` is `[batch, seq, hidden]`; `mask` is `[seq]` (1 = real token, 0 = padding).
/// Only real-token positions contribute to the sum and the divisor.
pub fn mean_pool_l2_normalize(hidden: &Tensor, mask: &[u32]) -> Result<Vec<f32>> {
    let (batch, seq, _h) = hidden.dims3()?;
    assert_eq!(mask.len(), seq, "mask length must equal sequence length");
    assert_eq!(batch, 1, "this helper only supports batch = 1");

    // [1, seq, 1] mask tensor, broadcast over the hidden dim.
    let mask_t = Tensor::from_vec(
        mask.iter().map(|&m| m as f32).collect::<Vec<_>>(),
        (1, seq, 1),
        hidden.device(),
    )?
    .to_dtype(DType::F32)?;

    // sum over seq of (hidden * mask) -> [1, 1, h]
    let masked = hidden.broadcast_mul(&mask_t)?;
    let sum = masked.sum_keepdim(1usize)?;

    // divisor = mask.sum() -> scalar tensor
    let count: f32 = mask.iter().map(|&m| m as f32).sum();
    let safe_count = count.max(1.0);
    let mean = sum.broadcast_div(&Tensor::from_vec(
        vec![safe_count],
        (1, 1, 1),
        hidden.device(),
    )?)?;

    // [h]
    let mean = mean.squeeze(0)?.squeeze(0)?;
    let vec: Vec<f32> = mean.to_vec1()?;
    l2_normalize(&vec)
}

/// CLS pooling: take the first token's hidden state ([CLS] for BERT), then L2-normalize.
/// This matches BGE's `1_Pooling/config.json` (`pooling_mode_cls_token: true`).
pub fn cls_pool_l2_normalize(hidden: &Tensor) -> Result<Vec<f32>> {
    // hidden: [1, seq, h] -> take index 0 along seq -> [1, h] -> [h]
    let cls = hidden.narrow(1, 0, 1)?; // [1, 1, h]
    let cls = cls.squeeze(0)?.squeeze(0)?; // [h]
    let vec: Vec<f32> = cls.to_vec1()?;
    l2_normalize(&vec)
}

fn l2_normalize(vec: &[f32]) -> Result<Vec<f32>> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let safe_norm = norm.max(1e-8);
    Ok(vec.iter().map(|&x| x / safe_norm).collect())
}
