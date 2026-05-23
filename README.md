# Mamba-130M: Trained from Scratch

> Clean PyTorch implementation of Mamba — trained end-to-end from random weights on real data.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/W%26B-Integrated-yellow.svg" alt="Weights & Biases">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## What This Is

A from-scratch PyTorch implementation of Mamba-130M. Not a wrapper around `mamba_ssm`. Not loading pretrained weights. Built the selective SSM, causal convolution, RMSNorm, and training loop — and trained it on real C4 data.

Supports multiple model sizes: **130M, 370M, 790M, 1.4B, 2.8B** (configs ready, training tested on 130M).

---

## Quick Start

```bash
git clone https://github.com/SirajuddinShaik/mamba-from-scratch.git
cd mamba-from-scratch

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train from scratch
python scripts/train_extended.py \
    --dataset c4 --max_steps 50000 \
    --batch_size 2 --grad_accum 4 --seq_len 1024

# Resume training
python scripts/train_extended.py \
    --dataset c4 --max_steps 100000 \
    --batch_size 2 --grad_accum 4 --seq_len 1024 \
    --resume outputs_extended/final_model.pt

# Evaluate
python scripts/evaluate.py --checkpoint outputs_extended/final_model.pt
```

---

## Architecture

| Size | Params | d_model | Layers | Status |
|------|--------|---------|--------|--------|
| **130M** | 129M | 768 | 24 | ✅ Tested on RTX 4090 |
| **370M** | ~370M | 1024 | 48 | 🔄 Config ready |
| **790M** | ~790M | 1536 | 48 | 🔄 Config ready |
| **1.4B** | ~1.4B | 2048 | 48 | 🔄 Config ready |
| **2.8B** | ~2.8B | 2560 | 64 | 🔄 Config ready |

Vocab: 50,257 (GPT-2 tokenizer)

---

## Training Results (RTX 4090)

### Full Run: 50K Steps, ~410M Tokens

| Metric | Value |
|--------|-------|
| Dataset | C4 |
| Steps | 50,000 |
| Tokens | ~410M |
| Batch | 2 × 4 grad accum = 8 |
| Time | **124 min** |
| Throughput | 54,940 tok/s |
| Best Train Loss | **3.83** |

**Evaluation:**

| Split | Perplexity |
|-------|-----------|
| C4 (val) | **53.8** |
| WikiText-103 | **132.3** |
| WikiText-2 | **132.3** |

**W&B Dashboard:** https://wandb.ai/sirajuddin-shaik-007/mamba-from-scratch/runs/a0i2sey4

### Learning Curve

| Check-in | Steps | C4 PPL | WikiText PPL |
|----------|-------|--------|-------------|
| Baseline | 20K | 108.7 | 324.1 |
| Extended v1 | 10K | 94.6 | 294.4 |
| **Full Run** | **50K** | **53.8** | **132.3** |

---

## What's Inside

- **Full Mamba implementation** — selective SSM, causal conv, RMSNorm
- **Streaming data pipeline** — C4 via HuggingFace (no toy data)
- **Training loop** — AdamW, warmup + cosine LR, gradient clipping, bfloat16 AMP
- **Checkpoint resume** — save/load model + optimizer state
- **Evaluation** — WikiText-103, WikiText-2, C4 validation
- **W&B logging** — real-time loss curves, artifact tracking
- **Multi-size configs** — 130M to 2.8B ready to go

---

## Hardware Used

- **GPU:** NVIDIA RTX 4090 (24GB)
- **RAM:** 62GB
- **OS:** Ubuntu 22.04
- **Driver:** CUDA 12.9

---

## What's Next

- [ ] Scale up to **370M** on RTX 4090 (need ZeRO-offload for larger batches)
- [ ] Train **790M / 1.4B / 2.8B** on H200 / DGX Spark clusters
- [ ] Add MoE routing to Mamba blocks
- [ ] Experiment with different optimizers (AdamW vs Lion vs Muon)
- [ ] Mixture of datasets (C4 + FineWeb + code)
- [ ] Long-context training (seq_len 4096+)

---

## Project Structure

```
mamba-from-scratch/
├── mamba/
│   ├── models/           # Configs + MambaLM
│   ├── modules/          # MambaBlock, RMSNorm
│   ├── ops/              # Selective scan (naive + Triton)
│   └── utils/
├── scripts/
│   ├── train_extended.py # Main training script
│   ├── evaluate.py       # Eval suite
│   ├── eval_final.py     # Benchmarking
│   └── test_mamba.py     # Unit tests
├── requirements.txt
└── setup.py
```

---

## Why From Scratch?

Most "from scratch" repos are thin wrappers around `mamba_ssm`. This one implements the core ops in PyTorch — selective scan, discretization, causal convolution. You can trace every operation.

---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## License

MIT
