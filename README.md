# Mamba-130M: Trained from Scratch

> A clean, modular PyTorch implementation of the Mamba architecture -- trained end-to-end from random initialization on real data.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## Overview

This repository contains a **from-scratch** implementation and training pipeline for **Mamba-130M** (selective state space models). Not a fine-tune. Not a wrapper around a pre-trained checkpoint. Built the architecture, data pipeline, and training loop -- and trained it end-to-end on real streaming data.

### What is Included

- **Clean Mamba implementation** -- modular blocks, RMS norm, causal conv, selective SSM
- **Streaming data pipeline** -- C4, FineWeb, FineWeb-Edu via HuggingFace (no synthetic data)
- **Training scripts** -- LR warmup, cosine decay, gradient clipping, AMP (bfloat16)
- **Extended training** -- gradient accumulation, checkpoint resume, periodic eval
- **Evaluation suite** -- WikiText-103, WikiText-2, C4 validation
- **W&B integration** -- real-time loss curves, checkpoint artifacts
- **Reproducible configs** -- seed, hyperparameters, hardware specs documented

---

## Quick Start

```bash
# Clone
git clone https://github.com/SirajuddinShaik/mamba-from-scratch.git
cd mamba-from-scratch

# Create env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train baseline (C4, ~82M tokens, ~15 min on RTX 4090)
python scripts/train_extended.py --dataset c4 --max_steps 10000 --batch_size 2 --grad_accum 4

# Or train fixed version (20K steps)
python scripts/train_fixed.py

# Evaluate
python scripts/evaluate.py --checkpoint outputs_extended/final_model.pt
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

Full architecture details in the code:
- `mamba/models/mamba_lm.py` -- Language model head + backbone
- `mamba/modules/mamba_block.py` -- Core selective SSM block
- `mamba/modules/rms_norm.py` -- RMSNorm layer
- `mamba/ops/selective_scan.py` -- Selective scan operation
- `mamba/ops/triton/selective_scan_triton.py` -- Triton-accelerated variant

---

## Training Results

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

| Dataset | Perplexity | Loss |
|---------|-----------|------|
| C4 (val) | **108.7** | 4.69 |
| WikiText-103 | **324.1** | 5.78 |
| WikiText-2 | **341.8** | 5.83 |

### Run 2: Extended C4 (Resumed, 82M tokens, 5K->10K)

| Setting | Value |
|---------|-------|
| Dataset | C4 |
| Steps | 10,000 (resumed from 5,000) |
| Effective batch | 2 x 4 grad accum = 8 |
| Seq len | 1,024 |
| LR | 6e-4 |
| Time | ~15.8 min |
| GPU | RTX 4090 (24GB) |

| Dataset | Perplexity | Loss |
|---------|-----------|------|
| C4 (val) | **94.6** | 4.55 |
| WikiText-103 | **294.4** | 5.68 |
| WikiText-2 | **294.4** | 5.68 |
| **Best Train** | **72.4** | **4.28** |

> **W&B Dashboard:** https://wandb.ai/sirajuddin-shaik-007/mamba-from-scratch/runs/ynw4lu6u

### Performance Notes

These are early-stage training results on ~82M tokens (0.03% of the ~300B tokens the official Mamba was trained on). Perplexity improves significantly with longer training:

| Target | Expected WikiText-2 PPL |
|--------|------------------------|
| 500M tokens | ~80-150 |
| 1-10B tokens | ~40-60 |
| 100B+ tokens | ~15-25 (approaching official) |

---

## Project Structure

```
mamba-from-scratch/
  mamba/
    models/
      config.py
      mamba_lm.py
    modules/
      mamba_block.py
      rms_norm.py
    ops/
      selective_scan.py
      triton/
        selective_scan_triton.py
    utils/
      torch_utils.py
  scripts/
    train_extended.py
    train_fixed.py
    train_v2.py
    evaluate.py
    eval_final.py
    eval_official.py
    example_usage.py
    test_mamba.py
  requirements.txt
  setup.py
  README.md
```

---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) -- minimal training loop inspiration
- FineWeb dataset: [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

---

## License

MIT
