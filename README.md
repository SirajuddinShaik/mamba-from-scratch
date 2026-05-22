# Mamba-130M: Trained from Scratch

> A clean, modular PyTorch implementation of the Mamba architecture — trained end-to-end from random initialization on real data (C4).

---

## Overview

This repository contains a **from-scratch** implementation and training pipeline for Mamba-130M (selective state space models). Not a fine-tune. Not a wrapper around a pre-trained checkpoint. Built the architecture, data pipeline, and training loop — and trained it end-to-end.

### What's Included

- **Clean Mamba implementation** — modular blocks, RMS norm, causal conv, selective SSM
- **Streaming data pipeline** — C4 via HuggingFace datasets (no synthetic/toy data)
- **Training scripts** — with LR warmup, cosine decay, gradient clipping, AMP (bfloat16)
- **Evaluation suite** — WikiText-103, WikiText-2, C4 validation
- **W&B integration** — real-time loss curves, checkpoint artifacts
- **Reproducible configs** — seed, hyperparameters, hardware specs documented

---

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train_v2_wandb.py
```

---

## Architecture

| Component | Spec |
|-----------|------|
| Params | 129.1M |
| Layers | 24 |
| d_model | 768 |
| d_state (SSM) | 16 |
| Expand factor | 2 |
| Conv kernel | 4 |
| Vocab size | 50,257 (GPT-2 tokenizer) |

---

## Training Run

| Setting | Value |
|---------|-------|
| Dataset | C4 (Colossal Clean Crawled Corpus) |
| Steps | 20,000 |
| Tokens | ~82M |
| Batch size | 4 |
| Seq len | 1,024 |
| LR | 6e-4 (cosine decay to 6e-5) |
| Warmup | 2,000 steps |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| Precision | bfloat16 + GradScaler |
| Gradient clip | 1.0 |

### Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX 4090 (24 GB) |
| Training time | ~30 min |
| Throughput | ~46K tokens/sec |
| CUDA | 12.9 |

### Loss Curve Highlights

| Step | Loss | Perplexity |
|------|------|-----------|
| 0 | 10.997 | 59,696 |
| 2,000 | 5.816 | 335.5 |
| 10,000 | 4.930 | 138.4 |
| 20,000 | **4.607** | **100.2** |

**Training was stable throughout** — smooth loss decay, zero NaN crashes.

---

## Evaluation Results

| Dataset | Perplexity | Loss | Notes |
|---------|-----------|------|-------|
| **C4 (val)** | **108.7** | 4.69 | In-distribution |
| WikiText-103 | **324.1** | 5.78 | Cross-domain eval |
| WikiText-2 | **341.8** | 5.83 | Cross-domain eval |

> **Context:** Official Mamba-130M was trained on ~300B tokens (3,600× more data). These numbers are expectedly early-stage. The architecture and pipeline are production-ready for scaled runs.

---

## Project Structure

```
mamba/
├── mamba/
│   ├── ops/
│   ├── modules/
│   ├── models/
│   └── utils/
├── scripts/
│   ├── train_v2_wandb.py      # Main training (W&B)
│   ├── train_v2.py             # Training without W&B
│   ├── eval_final.py           # Benchmark evaluation
│   └── test_mamba.py           # Unit tests
├── requirements.txt
└── setup.py
```

---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for minimal training loops.

---

## License

MIT
