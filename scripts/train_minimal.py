#!/usr/bin/env python3
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class TextDataset(IterableDataset):
    def __init__(self, dataset_name, config_name, split, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset(
            dataset_name, config_name, split=split, streaming=True
        )

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample.get("text", sample.get("content", ""))
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)
            while len(buffer) >= self.max_length:
                yield torch.tensor(buffer[: self.max_length], dtype=torch.long)
                buffer = buffer[self.max_length :]


def main():
    model_size = "130M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    batch_size = 2
    max_length = 256
    max_steps = 5000
    target_loss = 2.5
    log_interval = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    config = create_config(model_size)
    config.ssm_cfg = {"d_state": 16, "expand": 2, "d_conv": 4}

    model = MambaLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(
        dataset_name, dataset_config, "train", tokenizer, max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1
    )
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    best_loss = float("inf")
    step = 0
    start_time = time.time()

    print(f"\nTraining {model_size} model...")
    print(f"Batch size: {batch_size}, Sequence length: {max_length}")
    print(f"Target loss: {target_loss}")
    print("=" * 60)

    for batch in train_loader:
        if step >= max_steps:
            break

        batch = batch.to(device)

        optimizer.zero_grad()
        with autocast(dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % log_interval == 0:
            ppl = math.exp(min(loss.item(), 10))
            elapsed = time.time() - start_time
            print(
                f"Step {step:5d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | Time: {elapsed:.1f}s"
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                if loss.item() <= target_loss:
                    print(f"\nTARGET LOSS {target_loss} REACHED AT STEP {step}!")
                    break

        step += 1

    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"  Model: {model_size}")
    print(f"  Parameters: {num_params / 1e6:.1f}M")
    print(f"  Total steps: {step}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final perplexity: {math.exp(best_loss):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
