//! Simple semantic similarity benchmark using mean pooling.
use anyhow::Result;
use tokenizers::Tokenizer;
use crate::model::BertForEmbedding;
use crate::pooling::mean_pool_l2_normalize;

pub struct Pair {
    pub a: &'static str,
    pub b: &'static str,
    pub expected: &'static str,
}

pub fn run(bert: &BertForEmbedding, tok: &Tokenizer) -> Result<()> {
    let pairs = vec![
        // high — paraphrase / near-synonym rewrites
        Pair { a: "The cat sat on the mat", b: "A feline rested on the rug", expected: "high" },
        Pair { a: "Machine learning is fun", b: "Deep learning is exciting", expected: "high" },
        Pair { a: "The quick brown fox jumps over the lazy dog", b: "A fast brown fox leaps above the sleepy canine", expected: "high" },
        Pair { a: "How do I set the cargo build profile", b: "The build configuration uses cargo profiles for optimization", expected: "high" },
        Pair { a: "The dog ran in the park", b: "A puppy played outside", expected: "high" },
        // medium — same broad domain, different specifics
        Pair { a: "Artificial intelligence will change the world", b: "Technology is evolving rapidly", expected: "medium" },
        // low — unrelated domains
        Pair { a: "The ocean is vast and deep", b: "Rust is a systems programming language", expected: "low" },
        Pair { a: "Quantum computing uses qubits", b: "The weather is sunny today", expected: "low" },
        Pair { a: "Rust programming language is safe", b: "Pizza is delicious food", expected: "low" },
        Pair { a: "Stock market crashed yesterday", b: "Cats love to sleep all day", expected: "low" },
    ];
    println!();
    println!("=== Embedding Similarity Benchmark (BAAI/bge-small-en-v1.5) ===");
    println!("{:<48} {:<48} | {:>7} | {:>7} | {:>6}", "Sentence A", "Sentence B", "Cosine", "Exp", "OK");
    println!("{}", "-".repeat(120));
    let mut pass = 0;
    for p in &pairs {
        let sim = encode_sim(bert, tok, p.a, p.b)?;
        let ok = check(sim, p.expected);
        if ok { pass += 1; }
        println!(
            "{:<48} {:<48} | {:>7.4} | {:>7} | {}",
            short(p.a, 48), short(p.b, 48), sim, p.expected,
            if ok { "OK" } else { "FAIL" }
        );
    }
    println!();
    println!("Score: {}/{} passed", pass, pairs.len());
    Ok(())
}

/// Similarity band check. Bands are tuned to BGE-small's empirical cosine
/// distribution (unrelated pairs sit ~0.43–0.59, same-domain ~0.59–0.72,
/// near-synonyms >= 0.72). These are calibration constants, not hard science.
fn check(sim: f32, exp: &str) -> bool {
    match exp {
        "high" => sim >= 0.72,
        "medium" => sim >= 0.59 && sim < 0.72,
        "low" => sim < 0.59,
        _ => false,
    }
}
fn short(s: &str, n: usize) -> &str {
    if s.len() <= n { s } else { &s[..n] }
}

/// Encode with BGE instruction prefix.
/// Query side uses "query: ", passage side uses "passage: ".
fn encode_with_prefix(tok: &Tokenizer, text: &str, prefix: &str) -> Result<(Vec<u32>, Vec<u32>)> {
    let full = format!("{}{}", prefix, text);
    let enc = tok.encode(full.as_str(), true).map_err(|e| anyhow::anyhow!("tokenize failed: {:?}", e))?;
    let ids: Vec<u32> = enc.get_ids().iter().map(|&t| t as u32).collect();
    let mask: Vec<u32> = enc.get_attention_mask().iter().map(|&t| t as u32).collect();
    Ok((ids, mask))
}

fn encode_sim(bert: &BertForEmbedding, tok: &Tokenizer, a: &str, b: &str) -> Result<f32> {
    // Use query prefix for first sentence, passage prefix for second
    let (ids_a, mask_a) = encode_with_prefix(tok, a, "query: ")?;
    let (ids_b, mask_b) = encode_with_prefix(tok, b, "passage: ")?;
    let out_a = bert.forward(&ids_a, &mask_a)?;
    let out_b = bert.forward(&ids_b, &mask_b)?;
    let va = mean_pool_l2_normalize(&out_a, &mask_a)?;
    let vb = mean_pool_l2_normalize(&out_b, &mask_b)?;
    let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    Ok(dot)
}
