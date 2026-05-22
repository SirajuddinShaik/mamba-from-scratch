import sys; sys.stdout.reconfigure(line_buffering=True)
#!/usr/bin/env python3
"""Fixed Mamba-130M training script for LinkedIn portfolio.

Key fixes from original:
1. Pre-norm (RMSNorm) before each Mamba layer — original was missing this
2. Removed GradScaler (not needed for bf16)
3. Lower LR (3e-4) with longer warmup (5000 steps)
4. OpenWebText dataset (larger, higher quality than WikiText-103)
5. NaN loss detection and skip
6. Checkpoint saving
7. Proper logging with W&B-style console output
"""

import os, sys, math, time, json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, seq_len=512, split="train", max_tokens=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        if dataset_name == "c4":
            self.ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        elif dataset_name == "wikitext103":
            self.ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __iter__(self):
        buffer = []
        total_tokens = 0
        for item in self.ds:
            text = item.get("text", "")
            if len(text) < 20:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[: self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]
                total_tokens += self.seq_len
                if self.max_tokens and total_tokens >= self.max_tokens:
                    return


def train():
    # ── Config ──────────────────────────────────────────────────────────
    MODEL = "130M"
    BATCH = 8
    SEQ = 512
    EPOCHS = 1
    LR = 3e-4              # Lowered from 6e-4
    WARMUP = 5000           # Increased from 2000
    CLIP = 1.0
    LOG_EVERY = 100
    SAVE_EVERY = 2000
    MAX_STEPS = 20000
    DATASET = "c4"  # C4 is larger and cleaner than WikiText-103
    OUTPUT_DIR = Path.home() / "workspace/mamba/outputs_fixed"
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

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading {DATASET} (streaming)...")
    train_ds = StreamingDataset(DATASET, tokenizer, SEQ, split="train")
    val_ds = StreamingDataset("wikitext103", tokenizer, SEQ, split="validation")

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0)

    # ── Optimizer (no GradScaler for bf16) ──────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )

    best_loss = float("inf")
    global_step = 0
    running_loss = 0.0
    running_count = 0
    nan_count = 0
    start = time.time()
    log_entries = []

    print(f"\n{'=' * 80}")
    print(f"Training Mamba-{MODEL} on {DATASET}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH} | Seq: {SEQ} | LR: {LR}")
    print(f"Warmup: {WARMUP} | Max steps: {MAX_STEPS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            batch = batch.to(device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            # ── Learning rate schedule ───────────────────────────────────
            if global_step < WARMUP:
                lr = LR * (global_step + 1) / WARMUP
            else:
                progress = (global_step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
                lr = LR * 0.1 + (LR - LR * 0.1) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            for pg in opt.param_groups:
                pg["lr"] = lr

            # ── Forward (bf16 autocast, no GradScaler) ──────────────────
            opt.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )

            # ── NaN detection ────────────────────────────────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  ⚠ NaN/Inf loss at step {global_step} (total: {nan_count}) — skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

            loss_val = loss.item()
            running_loss += loss_val
            running_count += 1

            # ── Logging ──────────────────────────────────────────────────
            if global_step % LOG_EVERY == 0 and running_count > 0:
                avg = running_loss / running_count
                running_loss = 0.0
                running_count = 0
                elapsed = time.time() - start
                tok_s = global_step * BATCH * SEQ / max(elapsed, 1)
                ppl = math.exp(min(avg, 20))
                prog = global_step / MAX_STEPS * 100

                entry = {
                    "step": global_step, "loss": avg, "ppl": ppl,
                    "lr": lr, "tok_s": tok_s, "time_s": elapsed, "progress": prog
                }
                log_entries.append(entry)

                print(
                    f"E{epoch + 1} S{global_step:5d} ({prog:5.1f}%) | "
                    f"Loss: {avg:.4f} | PPL: {ppl:.2f} | "
                    f"LR: {lr:.2e} | Tok/s: {tok_s:.0f} | T: {elapsed:.0f}s"
                )

                if avg < best_loss:
                    best_loss = avg

            # ── Checkpoint ───────────────────────────────────────────────
            if global_step > 0 and global_step % SAVE_EVERY == 0:
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_loss": best_loss,
                    "config": str(config),
                }, ckpt_path)
                print(f"  💾 Saved checkpoint: {ckpt_path}")

            global_step += 1

        print(f">>> Epoch {epoch + 1} done")

    # ── Final save ──────────────────────────────────────────────────────
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
        "config": str(config),
    }, final_path)

    # ── Validation ──────────────────────────────────────────────────────
    print("\n--- Running validation on WikiText-103 ---")
    model.eval()
    val_losses = []
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            if val_steps >= 200:
                break
            batch = batch.to(device)
            with autocast(dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            if not torch.isnan(loss):
                val_losses.append(loss.item())
                val_steps += 1

    val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
    val_ppl = math.exp(min(val_loss, 20))

    # ── Summary ─────────────────────────────────────────────────────────
    total = time.time() - start
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"  Steps: {global_step}")
    print(f"  Best train loss: {best_loss:.4f}")
    print(f"  Best train PPL: {math.exp(min(best_loss, 20)):.2f}")
    print(f"  Val loss (WikiText-103): {val_loss:.4f}")
    print(f"  Val PPL (WikiText-103): {val_ppl:.2f}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Time: {total / 60:.1f} min")
    print(f"  Throughput: {global_step * BATCH * SEQ / total:.0f} tok/s avg")
    print(f"  Model saved: {final_path}")
    print(f"{'=' * 80}")

    # Save log
    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": MODEL,
            "dataset": DATASET,
            "params": params,
            "best_train_loss": best_loss,
            "best_train_ppl": math.exp(min(best_loss, 20)),
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "nan_skips": nan_count,
            "total_steps": global_step,
            "total_time_min": total / 60,
            "avg_tok_s": global_step * BATCH * SEQ / total,
            "loss_curve": log_entries,
        }, f, indent=2)
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    train()
