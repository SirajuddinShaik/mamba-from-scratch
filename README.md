# Mamba-130M: Trained from Scratch

> Clean PyTorch implementation of the Mamba architecture — trained end-to-end from random initialization on real streaming data.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/W%26B-Integrated-yellow.svg" alt="Weights & Biases">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## What's Inside

- **Full Mamba implementation** from scratch — selective SSM, causal convolution, RMSNorm
- **Streaming data pipeline** — C4 via HuggingFace `datasets` (no synthetic/toy data)
- **Complete training loop** — AdamW, warmup + cosine LR decay, gradient clipping, bfloat16 AMP
- **Checkpoint resume** — save/load model + optimizer state mid-training
- **Evaluation suite** — WikiText-103, WikiText-2, C4 validation perplexity
- **W&B integration** — real-time loss curves, artifact tracking

---

## Quick Start

```bash
git clone https://github.com/SirajuddinShaik/mamba-from-scratch.git
cd mamba-from-scratch

# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train from scratch (50K steps, ~2 hours on RTX 4090)
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

| Component | Spec |
|-----------|------|
| Parameters | **129.1M** |
| Layers | 24 |
| d_model | 768 |
| d_state (SSM) | 16 |
| Expand factor | 2 |
| Conv kernel | 4 |
| Vocab size | 50,257 (GPT-2 tokenizer) |

---

## Training Results

### Run 3: Full Run (50K steps, ~410M tokens, 2h on RTX 4090) ✅

| Metric | Value |
|--------|-------|
| Dataset | C4 |
| Steps | 50,000 |
| Tokens | ~410M |
| Batch | 2 × 4 grad accum = eff 8 |
| LR | 6e-4 → 1e-5 (cosine) |
| Time | **124 min** |
| Throughput | 54,940 tok/s |
| Best Train Loss | **3.83** |
| Best Train PPL | **~46** |

**Evaluation:**

| Split | Loss | Perplexity |
|-------|------|-----------|
| C4 (val) | 3.99 | **53.8** |
| WikiText-103 | 4.89 | **132.3** |
| WikiText-2 | 4.89 | **132.3** |

**W&B Dashboard:** https://wandb.ai/sirajuddin-shaik-007/mamba-from-scratch/runs/a0i2sey4

### Historical Runs

| Run | Steps | Tokens | C4 PPL | WikiText PPL | Notes |
|-----|-------|--------|--------|-------------|-------|
| Baseline | 20K | ~82M | 108.7 | 324.1 | Initial implementation |
| Extended v1 | 10K | ~82M | 94.6 | 294.4 | Added grad accumulation |
| **Full v2** | **50K** | **~410M** | **53.8** | **132.3** | Fresh optimizer on resume |

---

## Project Structure

```
mamba-from-scratch/
├── mamba/
│   ├── models/
│   │   ├── config.py           # Model configs (130M, 370M, 790M)
│   │   └── mamba_lm.py         # Full MambaLM with generation
│   ├── modules/
│   │   ├── mamba_block.py      # Core selective SSM block
│   │   └── rms_norm.py         # RMSNorm
│   ├── ops/
│   │   ├── selective_scan.py   # Naive selective scan
│   │   └── triton/
│   │       └── selective_scan_triton.py  # Triton kernel
│   └── utils/
│       └── torch_utils.py
├── scripts/
│   ├── train_extended.py       # Main training script
│   ├── evaluate.py             # Eval suite
│   ├── eval_final.py           # Final benchmarking
│   ├── eval_official.py        # Compare with HF Mamba
│   ├── example_usage.py        # Inference demo
│   └── test_mamba.py           # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## Why Build From Scratch?

Most "from scratch" implementations are wrappers around `mamba_ssm` or `causal-conv1d`. This repo implements the core operations natively in PyTorch — the selective scan, the discretization, the causal convolution. You can trace every operation.

---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## License

MIT
