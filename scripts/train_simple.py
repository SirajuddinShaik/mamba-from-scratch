#!/usr/bin/env python3
"""Simple training script for Mamba model with early stopping based on loss."""

import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class TextDataset(IterableDataset):
    def __init__(
        self, dataset_name, split, tokenizer_name="gpt2", max_length=512, streaming=True
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)

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


def train_until_convergence(
    model,
    train_loader,
    device,
    target_loss=2.5,
    max_steps=10000,
    eval_every=100,
    patience=10,
):
    """Train until loss converges below target."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))

    best_loss = float("inf")
    steps_without_improvement = 0
    step = 0

    print(
        f"Training {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameter model"
    )
    print(f"Target loss: {target_loss}")
    print(f"Max steps: {max_steps}")
    print("-" * 60)

    start_time = time.time()

    for batch in train_loader:
        if step >= max_steps:
            break

        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % eval_every == 0:
            perplexity = math.exp(min(loss.item(), 10))
            elapsed = time.time() - start_time
            print(
                f"Step {step:5d} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Check convergence
            if loss.item() < best_loss:
                best_loss = loss.item()
                steps_without_improvement = 0

                # Save best checkpoint
                os.makedirs("outputs", exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "loss": loss.item(),
                    },
                    "outputs/checkpoint_best.pt",
                )

                if loss.item() <= target_loss:
                    print(f"\n🎉 Target loss {target_loss} reached at step {step}!")
                    print(f"Best loss: {best_loss:.4f}")
                    return step, best_loss
            else:
                steps_without_improvement += 1

            if steps_without_improvement >= patience:
                print(
                    f"\n⚠️  Early stopping at step {step} (no improvement for {patience} evaluations)"
                )
                print(f"Best loss achieved: {best_loss:.4f}")
                return step, best_loss

        step += 1

    print(f"\nTraining complete. Steps: {step}, Best loss: {best_loss:.4f}")
    return step, best_loss


def main():
    # Configuration
    model_size = "130M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    batch_size = 4
    max_length = 512
    target_loss = 2.5
    max_steps = 5000

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    config = create_config(model_size)
    model = MambaLM(config).to(device)

    # Create dataset
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset_name}/{dataset_config}...")
    train_dataset = TextDataset(
        dataset_name, "train", tokenizer_name="gpt2", max_length=max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Train
    final_step, best_loss = train_until_convergence(
        model,
        train_loader,
        device,
        target_loss=target_loss,
        max_steps=max_steps,
        eval_every=50,
        patience=15,
    )

    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"  Model size: {model_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Final step: {final_step}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Perplexity: {math.exp(best_loss):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
