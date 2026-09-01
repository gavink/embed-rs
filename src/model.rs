//! Model loading + HuggingFace download helper.
//!
//! The BERT encoder itself is the upstream `candle_transformers::models::bert::BertModel`,
//! which already handles GELU activation, fused softmax, and the extended attention mask.
//! This file only keeps the weight/tokenizer/config download plumbing.
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

pub struct BertForEmbedding {
    model: BertModel,
    device: Device,
}

impl BertForEmbedding {
    pub fn load(config: &Config, path: &Path, device: &Device) -> Result<Self> {
        log::info!(
            "Config: hidden={} layers={} heads={} vocab={}",
            config.hidden_size, config.num_hidden_layers,
            config.num_attention_heads, config.vocab_size
        );
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        };
        let model = BertModel::load(vb, config)
            .map_err(|e| anyhow::anyhow!("failed to load BertModel: {e:?}"))?;
        Ok(Self { model, device: device.clone() })
    }

    /// Run the encoder and return the last hidden states `[batch, seq, hidden]`.
    pub fn forward(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
    ) -> candle_core::Result<Tensor> {
        let batch = 1;
        let seq = token_ids.len();
        let t_ids = Tensor::from_vec(token_ids.to_vec(), (batch, seq), &self.device)?;
        let t_seg = Tensor::zeros((batch, seq), DType::U32, &self.device)?;
        let t_mask = Tensor::from_vec(attention_mask.to_vec(), (batch, seq), &self.device)?;
        self.model.forward(&t_ids, &t_seg, Some(&t_mask))
    }
}

pub fn download(model_id: &str) -> Result<(Config, PathBuf, PathBuf)> {
    let base = std::env::var("HF_HOME").unwrap_or_else(|_| {
        std::env::var("XDG_CACHE_HOME").ok()
            .unwrap_or_else(|| std::env::var("USERPROFILE").unwrap_or_default() + "\\.cache")
    });
    let model_dir = PathBuf::from(&base)
        .join("huggingface")
        .join("hub")
        .join("models--")
        .join(model_id.replace('/', "--"));
    let snap = model_dir.join("snapshots").join("main");
    std::fs::create_dir_all(&snap).ok();

    let base_url = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());

    fn dl(url: &str, dest: &Path) -> Result<PathBuf> {
        if dest.exists() { return Ok(dest.to_path_buf()); }
        log::info!("Downloading {}", url);
        let bytes = reqwest::blocking::get(url)?.bytes()?;
        std::fs::write(dest, bytes)?;
        Ok(dest.to_path_buf())
    }

    let config_url = format!("{base_url}/{model_id}/resolve/main/config.json");
    let config_path = snap.join("config.json");
    let config_bytes = if config_path.exists() {
        std::fs::read(&config_path)?
    } else {
        reqwest::blocking::get(&config_url)?.bytes()?.to_vec()
    };
    let config: Config = serde_json::from_slice(&config_bytes)
        .with_context(|| "parse config.json")?;
    std::fs::write(&config_path, &config_bytes)?;
    log::info!(
        "Config: hidden={} layers={} heads={} vocab={}",
        config.hidden_size, config.num_hidden_layers,
        config.num_attention_heads, config.vocab_size
    );

    let wt_sf = snap.join("model.safetensors");
    let wt_pt = snap.join("pytorch_model.bin");
    let wt_url = format!("{base_url}/{model_id}/resolve/main/model.safetensors");
    let weight_path = match dl(&wt_url, &wt_sf) {
        Ok(p) => p,
        Err(_) => dl(
            &format!("{base_url}/{model_id}/resolve/main/pytorch_model.bin"),
            &wt_pt,
        )?,
    };
    log::info!("Weights: {}", weight_path.display());

    let tok_path = snap.join("tokenizer.json");
    dl(&format!("{base_url}/{model_id}/resolve/main/tokenizer.json"), &tok_path)?;
    log::info!("Tokenizer: {}", tok_path.display());

    Ok((config, weight_path, tok_path))
}
