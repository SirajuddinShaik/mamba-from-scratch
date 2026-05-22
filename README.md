# Mamba-130M: Trained from Scratch

> A clean, modular PyTorch implementation of the Mamba architecture -- trained end-to-end from random initialization on real data.

---

## Overview

This repository contains a **from-scratch** implementation and training pipeline for Mamba-130M (selective state space models). Not a fine-tune. Not a wrapper around a pre-trained checkpoint. Built the architecture, data pipeline, and training loop -- and trained it end-to-end.

### What's Included

- **Clean Mamba implementation** -- modular blocks, RMS norm, causal conv, selective SSM
- **Streaming data pipeline** -- C4, FineWeb, FineWeb-Edu via HuggingFace (no synthetic data)
- **Training scripts** -- LR warmup, cosine decay, gradient clipping, AMP (bfloat16)
- **Extended training** -- gradient accumulation, checkpoint resume, periodic eval
- **Evaluation suite** -- WikiText-103, WikiText-2, C4 validation
- **W&B integration** -- real-time loss curves, checkpoint artifacts
- **Reproducible configs** -- seed, hyperparameters, hardware specs documented

---

## Quick Start



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

## Training Runs

### Run 1: Baseline (C4, 82M tokens)

| Setting | Value |
|---------|-------|
| Dataset | C4 |
| Steps | 20,000 |
| Tokens | ~82M |
| Batch | 4 x 1024 |
| LR | 6e-4 |
| Time | ~30 min |
| GPU | RTX 4090 (24GB) |

**Results:**

| Dataset | Perplexity | Loss |
|---------|-----------|------|
| C4 (val) | **108.7** | 4.69 |
| WikiText-103 | **324.1** | 5.78 |
| WikiText-2 | **341.8** | 5.83 |

### Run 2: Extended (FineWeb, 205M tokens)

| Setting | Value |
|---------|-------|
| Dataset | FineWeb (or mixed C4+FineWeb) |
| Steps | 50,000 |
| Tokens | ~205M |
| Batch | 4 x 2 grad accum = eff 8 |
| LR | 6e-4 |
| Eval every | 5,000 steps |

*Run this with: *

---

## Project Structure



---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) -- minimal training loop inspiration
- FineWeb dataset: [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

---

## License

MIT
