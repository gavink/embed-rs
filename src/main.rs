//! Candle BERT embedding inference — pure Rust.
mod model;
mod pooling;
mod benchmark;
mod stsb;

use std::path::PathBuf;
use anyhow::Result;
use candle_core::Device;
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
    model: String,
    #[arg(long, help = "Force CPU (no GPU backend is selected at runtime yet)")]
    cpu: bool,
    #[arg(long, help = "Run semantic similarity benchmark")]
    bench: bool,
    #[arg(long, help = "Run STS-Benchmark (optionally pass a custom data path)",
         num_args = 0..=1, default_missing_value = "data/sts-test.jsonl")]
    stsb: Option<PathBuf>,
    #[arg(long, default_value = "cls", help = "Pooling for STS-B: cls (BGE default) or mean")]
    pooling_mode: String,
    #[arg(long, default_value = "how do I set the cargo build profile")]
    text: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    // CPU is the only supported backend here; CUDA/Metal need matching cargo features.
    let device = Device::Cpu;
    log::info!("Loading model: {}", cli.model);
    let (config, weight_path, tokenizer_path) = model::download(&cli.model)?;
    let tokenizer: Tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {:?}", e))?;
    log::info!("Building model on {:?}...", device);
    let bert = model::BertForEmbedding::load(&config, &weight_path, &device)?;
    if let Some(path) = &cli.stsb {
        let pooling = match cli.pooling_mode.as_str() {
            "cls" => stsb::Pooling::Cls,
            "mean" => stsb::Pooling::Mean,
            other => return Err(anyhow::anyhow!(
                "unknown --pooling {other:?}; use 'cls' or 'mean'")),
        };
        stsb::run(&bert, &tokenizer, path, pooling)?;
    } else if cli.bench {
        benchmark::run(&bert, &tokenizer)?;
    } else {
        let text = cli.text.as_str();
        log::info!("Input: \"{}\"", text);
        let enc = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {:?}", e))?;
        let ids: Vec<u32> = enc.get_ids().iter().map(|&t| t as u32).collect();
        let mask: Vec<u32> = enc.get_attention_mask().iter().map(|&t| t as u32).collect();
        log::info!("Tokens: {}", ids.len());
        let t0 = std::time::Instant::now();
        let out = bert.forward(&ids, &mask)?;
        let elapsed = t0.elapsed();
        let emb = pooling::mean_pool_l2_normalize(&out, &mask)?;
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        log::info!("Embedding dim: {}", emb.len());
        log::info!("L2 norm: {:.6}", norm);
        log::info!("First 8 dims: {:?}", &emb[..8.min(emb.len())]);
        log::info!("Inference time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    }
    Ok(())
}
