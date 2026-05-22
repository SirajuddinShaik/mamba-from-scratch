#!/usr/bin/env python3
import os
import sys
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


class TextDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        split,
        tokenizer,
        max_length=512,
        streaming=True,
        config_name=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if config_name:
            self.dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=streaming,
            )
        else:
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


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_epoch(
    model, dataloader, optimizer, scaler, device, epoch, global_step, config, args
):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)

        lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        with autocast(enabled=args.fp16):
            logits = model(batch[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.size(0)
        total_tokens += batch.numel()
        global_step += 1

        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{lr:.2e}",
                }
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/perplexity": math.exp(loss.item()),
                        "train/lr": lr,
                        "train/global_step": global_step,
                    }
                )

        if global_step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, global_step, epoch, args)

        if global_step >= args.max_steps:
            break

    return global_step


def save_checkpoint(model, optimizer, scaler, global_step, epoch, args):
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        default="130M",
        choices=["130M", "370M", "790M", "1.4B", "2.8B"],
    )
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)

    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--streaming", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="mamba-training")

    # System args
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = create_config(args.model_size)
    config.ssm_cfg = {
        "d_state": args.d_state,
        "expand": args.expand,
        "d_conv": args.d_conv,
    }

    model = MambaLM(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(
        args.dataset,
        "train",
        tokenizer,
        args.max_length,
        args.streaming,
        args.dataset_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=args.fp16)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model)

    global_step = 0
    epoch = 0
    while global_step < args.max_steps:
        global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            global_step,
            config,
            args,
        )
        epoch += 1

    save_checkpoint(model, optimizer, scaler, global_step, epoch, args)
    print("Training complete!")


if __name__ == "__main__":
    main()
