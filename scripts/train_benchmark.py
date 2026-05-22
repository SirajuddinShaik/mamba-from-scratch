#!/usr/bin/env python3
"""Train Mamba on standard benchmark dataset (OpenWebText/C4)."""

import os
import sys
import math
import time
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class TokenizedDataset(IterableDataset):
    """Tokenizes text on-the-fly for streaming datasets."""

    def __init__(self, dataset_stream, tokenizer, max_length=512):
        self.dataset = dataset_stream
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample.get("text", "")
            if not text or len(text) < 20:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.max_length:
                yield torch.tensor(buffer[: self.max_length], dtype=torch.long)
                buffer = buffer[self.max_length :]


def count_tokens_in_dataset(dataset_name, config_name, split, tokenizer):
    """Count total tokens for progress estimation."""
    print(f"Counting tokens in {dataset_name}...")
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    total_tokens = 0
    for i, sample in enumerate(islice(ds, 10000)):
        text = sample.get("text", "")
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        total_tokens += tokens
        if i % 1000 == 0:
            print(f"  Sampled {i}/10000...", end="\r")
    avg_tokens = total_tokens / 10000
    print(f"\nAverage tokens per sample: {avg_tokens:.0f}")
    return avg_tokens


def main():
    # Config
    model_size = "130M"
    dataset_name = "openwebtext"
    dataset_config = None
    split = "train"
    batch_size = 4
    max_length = 512
    num_epochs = 1
    grad_clip = 1.0
    learning_rate = 6e-4
    warmup_steps = 1000
    log_interval = 50
    save_interval = 5000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # Model
    config = create_config(model_size)
    config.ssm_cfg = {"d_state": 16, "expand": 2, "d_conv": 4}
    model = MambaLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_size}")
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print(f"\nLoading dataset: {dataset_name}")
    try:
        raw_ds = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        print("Trying fallback: Skylion007/openwebtext...")
        dataset_name = "Skylion007/openwebtext"
        raw_ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

    train_dataset = TokenizedDataset(raw_ds, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    # Estimate dataset size
    print("Estimating dataset size...")
    sample_count = 0
    token_count = 0
    for batch in islice(train_loader, 100):
        sample_count += batch.size(0)
        token_count += batch.numel()
    tokens_per_batch = token_count / max(sample_count, 1)
    print(f"Tokens per batch: {tokens_per_batch:.0f}")
    print(f"Batch size: {batch_size}, Seq len: {max_length}")
    print(f"Estimated samples/batch: {tokens_per_batch / max_length:.0f}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = GradScaler()

    # Training state
    step = 0
    epoch = 0
    tokens_processed = 0
    best_loss = float("inf")
    running_loss = 0.0
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {max_length}")
    print(f"  Gradient clipping: {grad_clip}")
    print(f"{'=' * 60}\n")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            tokens_processed += batch.numel()

            # LR warmup
            if step < warmup_steps:
                lr = learning_rate * (step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Forward + backward
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Logging
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval if step > 0 else loss.item()
                running_loss = 0.0
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed
                ppl = math.exp(min(avg_loss, 10))
                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Step {step:6d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {ppl:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # Checkpointing
            if step > 0 and step % save_interval == 0:
                ckpt_path = f"/tmp/mamba_checkpoint_step{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    ckpt_path,
                )
                print(f"  -> Saved checkpoint to {ckpt_path}")

            step += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Total steps: {step}")
    print(f"  Total epochs: {epoch + 1}")
    print(f"  Tokens processed: {tokens_processed / 1e9:.2f}B")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final perplexity: {math.exp(best_loss):.2f}")
    print(f"  Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
