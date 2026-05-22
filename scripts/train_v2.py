#!/usr/bin/env python3
"""
Mamba-130M: Pretrain from scratch on The Pile subset + evaluate.

For LinkedIn portfolio: Train Mamba-130M from scratch and compare with
official pretrained model on standard benchmarks.

Strategy:
- Phase 1: Pretrain on C4 for 500M tokens (~10K steps, ~35 min on RTX 4090)
- Phase 2: Evaluate on WikiText-103, Penn Treebank, LAMBADA
- Compare: Our model vs Official mamba-130m-hf
"""

import os, sys, math, time, json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, seq_len=2048, split="train"):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if dataset_name == "c4":
            self.ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        elif dataset_name == "wikitext103":
            self.ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        elif dataset_name == "ptb":
            self.ds = load_dataset("ptb", split=split, streaming=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __iter__(self):
        buffer = []
        for item in self.ds:
            text = item.get("text", "")
            if len(text) < 20:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[: self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]


def evaluate(model, dataset_name, tokenizer, device, seq_len=512, max_batches=200):
    """Evaluate model on a dataset, return loss and PPL."""
    model.eval()
    ds = StreamingDataset(dataset_name, tokenizer, seq_len, split="validation")
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= max_batches:
                break
            batch = batch.to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            if not torch.isnan(loss):
                total_loss += loss.item()
                count += 1

    model.train()
    if count == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / count
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def train():
    # ── Config ──────────────────────────────────────────────────────────
    MODEL = "130M"
    BATCH = 4
    SEQ = 1024             # Longer sequences for better training
    LR = 6e-4              # Standard Mamba LR
    WARMUP = 2000
    CLIP = 1.0
    LOG_EVERY = 200
    SAVE_EVERY = 5000
    MAX_STEPS = 20000      # ~82M tokens
    DATASET = "c4"
    OUTPUT_DIR = Path.home() / "workspace/mamba/outputs_v2"
    SEED = 42

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # ── Model ───────────────────────────────────────────────────────────
    config = create_config(MODEL)
    config.ssm_cfg = {"d_state": 16, "expand": 2, "d_conv": 4}
    model = MambaLM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {MODEL} | Params: {params / 1e6:.1f}M")
    print(f"Seq len: {SEQ} | Dataset: {DATASET}")
    print(f"Total tokens: {MAX_STEPS * BATCH * SEQ / 1e6:.0f}M")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading {DATASET} (streaming)...")
    train_ds = StreamingDataset(DATASET, tokenizer, SEQ, split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=0)

    # ── Optimizer ───────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )

    best_loss = float("inf")
    global_step = 0
    running_loss = 0.0
    running_count = 0
    start = time.time()
    log_entries = []

    print(f"\n{'=' * 80}")
    print(f"Training Mamba-{MODEL} from scratch on {DATASET}")
    print(f"LR: {LR} | Warmup: {WARMUP} | Max steps: {MAX_STEPS}")
    print(f"{'=' * 80}\n")

    model.train()
    for batch in train_loader:
        if global_step >= MAX_STEPS:
            break

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # ── LR schedule ─────────────────────────────────────────────────
        if global_step < WARMUP:
            lr = LR * (global_step + 1) / WARMUP
        else:
            progress = (global_step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
            lr = LR * 0.1 + (LR - LR * 0.1) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ── Forward ─────────────────────────────────────────────────────
        opt.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠ NaN at step {global_step} — skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()

        running_loss += loss.item()
        running_count += 1

        # ── Logging ─────────────────────────────────────────────────────
        if global_step % LOG_EVERY == 0 and running_count > 0:
            avg = running_loss / running_count
            running_loss = 0.0
            running_count = 0
            elapsed = time.time() - start
            tok_s = global_step * BATCH * SEQ / max(elapsed, 1)
            ppl = math.exp(min(avg, 20))
            prog = global_step / MAX_STEPS * 100
            total_tok = global_step * BATCH * SEQ

            entry = {
                "step": global_step, "loss": avg, "ppl": ppl,
                "lr": lr, "tok_s": tok_s, "total_tokens_M": total_tok / 1e6,
                "time_s": elapsed, "progress": prog
            }
            log_entries.append(entry)

            print(
                f"S{global_step:5d} ({prog:5.1f}%) | "
                f"Loss: {avg:.4f} | PPL: {ppl:.1f} | "
                f"LR: {lr:.2e} | Tok: {total_tok/1e6:.0f}M | "
                f"Tok/s: {tok_s:.0f} | T: {elapsed:.0f}s"
            )

            if avg < best_loss:
                best_loss = avg

        # ── Checkpoint + eval ───────────────────────────────────────────
        if global_step > 0 and global_step % SAVE_EVERY == 0:
            ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}.pt"
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_loss": best_loss,
            }, ckpt_path)
            print(f"  💾 Saved: {ckpt_path}")

            # Quick eval
            val_loss, val_ppl = evaluate(model, "wikitext103", tokenizer, device, max_batches=50)
            print(f"  📊 WikiText-103 val: Loss={val_loss:.4f} PPL={val_ppl:.1f}")

        global_step += 1

    # ── Final save ──────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
    }, final_path)

    # ── Full evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    results = {}
    for ds_name in ["wikitext103", "ptb"]:
        loss, ppl = evaluate(model, ds_name, tokenizer, device, max_batches=200)
        results[ds_name] = {"loss": loss, "ppl": ppl}
        print(f"  {ds_name}: Loss={loss:.4f} PPL={ppl:.2f}")

    # ── Summary ─────────────────────────────────────────────────────────
    total = time.time() - start
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"  Model: Mamba-{MODEL} ({params/1e6:.1f}M params)")
    print(f"  Dataset: {DATASET}")
    print(f"  Tokens trained: {global_step * BATCH * SEQ / 1e6:.0f}M")
    print(f"  Steps: {global_step}")
    print(f"  Best train loss: {best_loss:.4f}")
    print(f"  Time: {total / 60:.1f} min")
    print(f"  Throughput: {global_step * BATCH * SEQ / total:.0f} tok/s")
    for ds_name, r in results.items():
        print(f"  {ds_name} PPL: {r['ppl']:.2f}")
    print(f"{'=' * 80}")
    print(f"\n  Reference: Official mamba-130m-hf gets PPL ~25 on WikiText-103")
    print(f"  (trained on 300B tokens = {300000/82:.0f}x more data)")

    # Save log
    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": MODEL,
            "dataset": DATASET,
            "params": params,
            "total_tokens_M": global_step * BATCH * SEQ / 1e6,
            "total_steps": global_step,
            "best_train_loss": best_loss,
            "total_time_min": total / 60,
            "avg_tok_s": global_step * BATCH * SEQ / total,
            "eval_results": results,
            "loss_curve": log_entries,
        }, f, indent=2)
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    train()
