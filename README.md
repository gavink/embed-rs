# Candle Embedding Inference

Pure-Rust BERT embedding inference using [Candle](https://github.com/huggingface/candle).
No ONNX, no Python, no C++ runtime dependency.

## Build

```bash
cargo build --release
```

## Run

```bash
# Default: BAAI/bge-small-en-v1.5 (~128MB downloads on first run)
# Set HF_ENDPOINT for Chinese mirror:
HF_ENDPOINT=https://hf-mirror.com cargo run --release

# Custom model
HF_ENDPOINT=https://hf-mirror.com cargo run --release -- --model sentence-transformers/all-MiniLM-L6-v2

# Force CPU (AMD GPU runs on CPU; CUDA/Metal on NVIDIA/Apple)
HF_ENDPOINT=https://hf-mirror.com cargo run --release -- --cpu --text "test"
```

## AMD GPU note

Candle currently supports CUDA (NVIDIA) and Metal (Apple Silicon) backends.
AMD GPUs run on CPU. For AMD GPU acceleration, use the `ort` crate instead
(which wraps ONNX Runtime with DirectML/Vulkan support).

## Expected output

```
Config: hidden=384 layers=12 heads=12 vocab=30522
Weights: ...\model.safetensors
Tokenizer: ...\tokenizer.json
Building model on Cpu ...
Input: "how do I set the cargo build profile"
Tokens: 10 (max: 256)
Embedding dim: 384
L2 norm: 1.000000 (target: ~1.0)
First 8 dims: [0.08081307, 0.004232504, -0.026600411, ...]
Inference time: 50-70 ms
```

## Models tested

| Model | Dim | Size | MTEB | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 93MB | ~63% | lightweight option |
| **`BAAI/bge-small-en-v1.5`** | 384 | 128MB | ~67% | **recommended** |
| `BAAI/bge-base-zh-v1.5` | 768 | 500MB | ~73% (zh) | bilingual |

## Benchmark Results

### STS-Benchmark (MTEB, 1379 pairs)

| Pooling | Spearman | vs Reference |
|---------|----------|--------------|
| `cls`   | 0.8586   | +0.00% (matches MTEB official) |

> **Note**: BGE's official config (`1_Pooling/config.json`) specifies
> `pooling_mode_cls_token: true`. Mean pooling gave 0.8680 but is not
> the canonical evaluation protocol.

Run with:
```bash
cargo run --release -- --stsb --pooling-mode cls
```

### Semantic Similarity Benchmark

10/10 passed — all pairs correctly classified as high/medium/low similarity.

Run with:
```bash
cargo run --release -- --bench
```

---

## Comparison: ONNX vs Candle

| | ONNX (tract) | this project (Candle) |
|---|---|---|
| Inference engine | `tract_onnx` (C++ backend) | `candle-core` (pure Rust) |
| Model format | ONNX (~93MB MiniLM) | safetensors (~128MB BGE) |
| GPU backend | CPU only | CUDA / Metal |
| External deps | ONNX Runtime (.dll/.so) | None (pure Rust) |
| Compile time | Heavy (ONNX Runtime builds C++) | Fast (pure Rust) |

## Dependencies

```toml
candle-core      = "0.8"
candle-nn        = "0.8"
candle-transformers = "0.8"
reqwest          = { version = "0.12", features = ["rustls-tls", "blocking"] }
tokenizers       = "0.21"
```
