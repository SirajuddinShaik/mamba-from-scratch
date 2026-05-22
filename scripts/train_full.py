#!/usr/bin/env python3
import os, sys, math, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class StreamingWikiText(IterableDataset):
    def __init__(self, split, tokenizer, seq_len=512):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
        )

    def __iter__(self):
        buffer = []
        for item in self.ds:
            text = item.get("text", "")
            if len(text) < 20:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                yield torch.tensor(buffer[: self.seq_len], dtype=torch.long)
                buffer = buffer[self.seq_len :]


def train():
    MODEL = "130M"
    BATCH = 8
    SEQ = 512
    EPOCHS = 1
    LR = 6e-4
    WARMUP = 2000
    CLIP = 1.0
    LOG = 100
    MAX_STEPS = 50000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(42)

    config = create_config(MODEL)
    config.ssm_cfg = {"d_state": 16, "expand": 2, "d_conv": 4}
    model = MambaLM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {MODEL} | Params: {params / 1e6:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading WikiText-103 (streaming)...")
    train_ds = StreamingWikiText("train", tokenizer, SEQ)
    val_ds = StreamingWikiText("validation", tokenizer, SEQ)

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0)

    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = GradScaler()

    best_loss = float("inf")
    global_step = 0
    running_loss = 0.0
    start = time.time()

    print(f"\n{'=' * 70}")
    print(f"Training {MODEL} on WikiText-103")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH} | Seq: {SEQ} | LR: {LR}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"{'=' * 70}\n")

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            batch = batch.to(device)

            if global_step < WARMUP:
                lr = LR * (global_step + 1) / WARMUP
            else:
                progress = (global_step - WARMUP) / max(MAX_STEPS - WARMUP, 1)
                lr = LR * 0.1 + (LR - LR * 0.1) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()

            loss_val = loss.item()
            running_loss += loss_val

            if global_step % LOG == 0:
                avg = running_loss / LOG if global_step > 0 else loss_val
                running_loss = 0.0
                elapsed = time.time() - start
                tok_s = global_step * BATCH * SEQ / max(elapsed, 1)
                ppl = math.exp(min(avg, 10))
                prog = global_step / MAX_STEPS * 100
                print(
                    f"E{epoch + 1} S{global_step:5d} ({prog:5.1f}%) | "
                    f"Loss: {avg:.4f} | PPL: {ppl:.2f} | "
                    f"LR: {lr:.2e} | Tok/s: {tok_s:.0f} | T: {elapsed:.0f}s"
                )

                if avg < best_loss:
                    best_loss = avg

            global_step += 1

        print(f">>> Epoch {epoch + 1} done")

    total = time.time() - start
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"  Steps: {global_step}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final PPL: {math.exp(min(best_loss, 10)):.2f}")
    print(f"  Time: {total / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    train()
