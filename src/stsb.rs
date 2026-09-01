//! STS-Benchmark (STS-B) evaluation.
//!
//! Loads the MTEB version of the STS-B test set (`mteb/stsbenchmark-sts`,
//! 1379 sentence pairs with gold similarity scores in [0, 5]) and computes
//! the Spearman rank correlation between cosine similarity of the model's
//! embeddings and the gold scores.
//!
//! Reference (official MTEB result for BAAI/bge-small-en-v1.5, test split):
//!   Spearman = 0.8586
//!
//! Data layout (one JSON object per line):
//!   {"score": 2.5, "sentence1": "...", "sentence2": "...", ...}
use std::path::Path;
use anyhow::{Context, Result};
use tokenizers::Tokenizer;
use crate::model::BertForEmbedding;
use crate::pooling::{cls_pool_l2_normalize, mean_pool_l2_normalize};

/// Official reference score (MTEB results DB, BAAI/bge-small-en-v1.5, STSBenchmark test).
pub const REFERENCE: f64 = 0.8586;

struct StsbRow {
    gold: f64,
    s1: String,
    s2: String,
}

fn load(path: &Path) -> Result<Vec<StsbRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read STS-B data: {}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse line {} of {}", i + 1, path.display()))?;
        rows.push(StsbRow {
            gold: v["score"].as_f64().with_context(|| format!("missing score on line {}", i + 1))?,
            s1: v["sentence1"].as_str().context("missing sentence1")?.to_string(),
            s2: v["sentence2"].as_str().context("missing sentence2")?.to_string(),
        });
    }
    Ok(rows)
}

/// Pooling strategy. Matches BGE's `1_Pooling/config.json` when set to Cls.
#[derive(Clone, Copy)]
pub enum Pooling {
    Cls,
    Mean,
}

/// Encode a sentence (no query/passage prefix — symmetric STS protocol).
fn encode(bert: &BertForEmbedding, tok: &Tokenizer, text: &str, pooling: Pooling) -> Result<Vec<f32>> {
    let enc = tok.encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {:?}", e))?;
    let ids: Vec<u32> = enc.get_ids().iter().map(|&t| t as u32).collect();
    let mask: Vec<u32> = enc.get_attention_mask().iter().map(|&t| t as u32).collect();
    let out = bert.forward(&ids, &mask)?;
    match pooling {
        Pooling::Cls => cls_pool_l2_normalize(&out),
        Pooling::Mean => mean_pool_l2_normalize(&out, &mask),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

/// Spearman rank correlation between two slices, using the simple
/// average-rank method for ties (good enough for 1379 points).
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let rx = rank(x);
    let ry = rank(y);
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 { return 0.0; }
    cov / (vx * vy).sqrt()
}

/// Assign ranks (1 = smallest), with ties getting the average of the ranks
/// they would have occupied.
fn rank(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 { j += 1; }
        let avg = ((i + 1 + j) as f64) / 2.0; // ranks i+1 .. j, 1-indexed
        for k in i..j { ranks[idx[k]] = avg; }
        i = j;
    }
    ranks
}

pub fn run(bert: &BertForEmbedding, tok: &Tokenizer, data_path: &Path, pooling: Pooling) -> Result<()> {
    let rows = load(data_path)?;
    let pool_name = match pooling { Pooling::Cls => "cls", Pooling::Mean => "mean" };
    println!();
    println!("=== STS-Benchmark (test, {} pairs, {} pooling) ===", rows.len(), pool_name);
    println!("Encoding sentences (symmetric, {}-pool, L2-norm)...", pool_name);
    let mut preds = Vec::with_capacity(rows.len());
    let mut golds = Vec::with_capacity(rows.len());
    let t0 = std::time::Instant::now();
    for r in &rows {
        let va = encode(bert, tok, &r.s1, pooling)?;
        let vb = encode(bert, tok, &r.s2, pooling)?;
        preds.push(cosine(&va, &vb));
        golds.push(r.gold);
    }
    let sp = spearman(&preds, &golds);
    let secs = t0.elapsed().as_secs_f64();
    println!();
    println!("Spearman:  {:.4}", sp);
    println!("Reference: {:.4}  (BAAI/bge-small-en-v1.5, MTEB official, cls pool)", REFERENCE);
    let delta = sp - REFERENCE;
    println!("Delta:     {:+.4}  ({:+.2}%)", delta, delta / REFERENCE * 100.0);
    println!("Time:      {:.1}s  ({:.1} ms/pair)", secs, secs * 1000.0 / rows.len() as f64);
    Ok(())
}
