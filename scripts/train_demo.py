#!/usr/bin/env python3
import sys
from pathlib import Path
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


def train_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)

    # Small model for demo
    config = create_config("130M")
    config.ssm_cfg = {
        "d_state": 16,
        "expand": 2,
        "d_conv": 4,
    }

    print("Initializing Mamba-130M model...")
    model = MambaLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")
    print()

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Small dataset for demo - wikitext is small and fast
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

    # Prepare data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
        )

    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids"])

    # DataLoader
    train_loader = DataLoader(tokenized, batch_size=4, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Training
    num_epochs = 3
    model.train()

    print("=" * 60)
    print("Starting training...")
    print(f"Samples: {len(tokenized)}, Epochs: {num_epochs}, Batch size: 4")
    print("=" * 60)

    start_time = time.time()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)

            # Skip empty sequences
            if input_ids.shape[1] < 2:
                continue

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids[:, :-1])

            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id else -100,
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Stats
            batch_loss = loss.item()
            batch_tokens = input_ids.numel()
            epoch_loss += batch_loss * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1

            # Update progress bar
            current_ppl = (
                math.exp(epoch_loss / epoch_tokens)
                if epoch_tokens > 0
                else float("inf")
            )
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "ppl": f"{current_ppl:.2f}",
                }
            )

        # Epoch summary
        avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        epoch_ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {epoch_ppl:.2f}")
        print(f"  Tokens processed: {epoch_tokens:,}")
        print()

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Training complete! Time: {elapsed / 60:.2f} minutes")
    print(f"Final perplexity: {epoch_ppl:.2f}")
    print("=" * 60)

    # Save model
    save_path = Path(__file__).parent.parent / "outputs"
    save_path.mkdir(exist_ok=True)
    checkpoint_file = save_path / "demo_checkpoint.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "tokenizer": tokenizer.name_or_path,
            "final_loss": avg_loss,
            "final_ppl": epoch_ppl,
        },
        checkpoint_file,
    )

    print(f"Checkpoint saved to: {checkpoint_file}")

    # Quick generation test
    print("\n" + "=" * 60)
    print("Generation Test:")
    print("=" * 60)

    model.eval()
    prompt_text = "The quick brown fox"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=30, temperature=0.8)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt_text}")
    print(f"Generated: {generated_text}")

    return {
        "final_loss": avg_loss,
        "final_perplexity": epoch_ppl,
        "training_time_minutes": elapsed / 60,
        "checkpoint_path": str(checkpoint_file),
    }


if __name__ == "__main__":
    try:
        results = train_demo()
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print("=" * 60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
