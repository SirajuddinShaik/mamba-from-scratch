#!/usr/bin/env python3
"""
Mamba-130M Extended Training Script

Trains on larger corpora with:
- W&B real-time logging (loss, PPL, LR, throughput)
- Checkpoint resume support
- Periodic evaluation on WikiText-103
- Gradient accumulation for larger effective batch
- FineWeb, FineWeb-Edu, C4, or mixed dataset support

Usage:
    python scripts/train_extended.py --dataset fineweb --max_steps 50000
"""

import os, sys, math, time, json, argparse
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("WARNING: wandb not installed")


class StreamingDataset(IterableDataset):
    """Universal streaming dataset supporting C4, FineWeb, and mixed."""

    SUPPORTED = {
        "c4": ("allenai/c4", "en"),
        "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT"),
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    }

    def __init__(self, name, tokenizer, seq_len=1024, split="train"):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.buffer = []
        self.name = name

        if name == "mixed":
            self.ds_c4 = load_dataset("allenai/c4", "en", split=split, streaming=True)
            self.ds_fw = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split=split, streaming=True)
            self.ds = self._mixed_iter()
        elif name in self.SUPPORTED:
            ds_name, config = self.SUPPORTED[name]
            self.ds = load_dataset(ds_name, config, split=split, streaming=True)
        else:
            valid = list(self.SUPPORTED.keys()) + ["mixed"]
            raise ValueError(f"Unknown dataset: {name}. Choose from {valid}")

    def _mixed_iter(self):
        """Interleave C4 and FineWeb 50/50."""
        c4_iter = iter(self.ds_c4)
        fw_iter = iter(self.ds_fw)
        toggle = True
        while True:
            try:
                yield next(c4_iter) if toggle else next(fw_iter)
                toggle = not toggle
            except StopIteration:
                break

    def __iter__(self):
        for item in self.ds:
            text = item.get("text", item.get("content", ""))
            if len(text) < 20:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.buffer.extend(tokens)
            while len(self.buffer) >= self.seq_len + 1:
                yield torch.tensor(self.buffer[: self.seq_len + 1], dtype=torch.long)
                self.buffer = self.buffer[self.seq_len:]


@torch.no_grad()
def evaluate(model, dataset_name, tokenizer, device, seq_len=512, max_batches=100):
    """Evaluate model, return loss and PPL."""
    model.eval()
    ds = StreamingDataset(dataset_name, tokenizer, seq_len, split="validation")
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    total_loss = 0
    count = 0
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
    avg = total_loss / count
    return avg, math.exp(min(avg, 20))


def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMamba-130M Extended Training")
    print(f"   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    config = create_config(args.model)
    config.ssm_cfg = {"d_state": 16, "expand": 2, "d_conv": 4}
    model = MambaLM(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    eff_batch = args.batch_size * args.grad_accum
    total_tokens_M = args.max_steps * eff_batch * args.seq_len / 1e6
    print(f"\n   Model: {args.model} | {params/1e6:.1f}M params")
    print(f"   Dataset: {args.dataset}")
    print(f"   Steps: {args.max_steps} | Tokens: ~{total_tokens_M:.0f}M")
    print(f"   Batch: {args.batch_size} x accum {args.grad_accum} = eff {eff_batch}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print(f"\n   Loading {args.dataset}...")
    train_ds = StreamingDataset(args.dataset, tokenizer, args.seq_len, split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Resume?
    start_step = 0
    best_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"\n   Resumed from step {start_step}")

    # W&B
    if HAS_WANDB:
        wandb.init(
            project="mamba-from-scratch",
            name=f"{args.model}-{args.dataset}-{args.max_steps}steps-extended",
            config=vars(args),
            resume="allow" if args.resume else None,
        )
        print(f"   W&B: {wandb.run.url if wandb.run else 'N/A'}")

    global_step = start_step
    running_loss = 0.0
    accum_loss = 0.0
    running_count = 0
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"   TRAINING START")
    print(f"{'='*70}\n")

    model.train()
    opt.zero_grad()

    for batch in train_loader:
        if global_step >= args.max_steps:
            break

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            ) / args.grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"   NaN at step {global_step}")
            if HAS_WANDB:
                wandb.log({"train/nan_skip": 1}, step=global_step)
            continue

        loss.backward()
        accum_loss += loss.item() * args.grad_accum
        running_loss += loss.item() * args.grad_accum
        running_count += 1

        # Gradient step
        if running_count % args.grad_accum == 0:
            lr = get_lr(global_step, args.warmup, args.max_steps, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad()

            # Logging
            if global_step % args.log_every == 0 and running_count > 0:
                avg = running_loss / running_count
                running_loss = 0.0
                running_count = 0
                elapsed = time.time() - start_time
                tok_s = global_step * eff_batch * args.seq_len / max(elapsed, 1)
                ppl = math.exp(min(avg, 20))
                prog = global_step / args.max_steps * 100
                total_tok = global_step * eff_batch * args.seq_len

                print(
                    f"S{global_step:5d} ({prog:5.1f}%) | "
                    f"Loss: {avg:.4f} | PPL: {ppl:.1f} | "
                    f"LR: {lr:.2e} | Tok: {total_tok/1e6:.0f}M | "
                    f"Tok/s: {tok_s:.0f} | T: {elapsed/60:.1f}min"
                )

                if HAS_WANDB:
                    wandb.log({
                        "train/loss": avg,
                        "train/perplexity": ppl,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tok_s,
                        "train/total_tokens_M": total_tok / 1e6,
                    }, step=global_step)

                if avg < best_loss:
                    best_loss = avg

            # Checkpoint
            if global_step > 0 and global_step % args.save_every == 0:
                ckpt_path = OUTPUT_DIR / f"checkpoint-{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_loss": best_loss,
                }, ckpt_path)
                print(f"   Checkpoint: {ckpt_path}")

                if HAS_WANDB:
                    artifact = wandb.Artifact(f"ckpt-{global_step}", type="model")
                    artifact.add_file(str(ckpt_path))
                    wandb.log_artifact(artifact)

            # Eval
            if global_step > 0 and global_step % args.eval_every == 0:
                val_loss, val_ppl = evaluate(model, "wikitext103", tokenizer, device, max_batches=args.eval_batches)
                print(f"   WikiText-103: Loss={val_loss:.4f} PPL={val_ppl:.1f}")
                if HAS_WANDB:
                    wandb.log({
                        "val/wikitext103_loss": val_loss,
                        "val/wikitext103_ppl": val_ppl,
                    }, step=global_step)

            global_step += 1

    # Final
    total = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"   TRAINING COMPLETE")
    print(f"   Steps: {global_step}")
    print(f"   Best train loss: {best_loss:.4f}")
    print(f"   Total time: {total/60:.1f} min")
    print(f"   Throughput: {global_step * eff_batch * args.seq_len / total:.0f} tok/s")
    print(f"{'='*70}")

    # Final eval
    print(f"\n   Final Evaluation:")
    final_results = {}
    for ds_name in ["wikitext103", "wikitext2", "c4"]:
        try:
            loss, ppl = evaluate(model, ds_name, tokenizer, device, max_batches=200)
            final_results[ds_name] = {"loss": loss, "ppl": ppl}
            print(f"   {ds_name}: Loss={loss:.4f} PPL={ppl:.1f}")
        except Exception as e:
            print(f"   {ds_name}: ERROR {str(e)[:40]}")

    # Save
    final_path = OUTPUT_DIR / "final_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "best_loss": best_loss,
    }, final_path)

    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump({
            "config": vars(args),
            "results": final_results,
            "best_loss": best_loss,
            "total_time_min": total / 60,
        }, f, indent=2)

    if HAS_WANDB:
        for ds, r in final_results.items():
            wandb.summary[f"final/{ds}_ppl"] = r["ppl"]
            wandb.summary[f"final/{ds}_loss"] = r["loss"]
        wandb.summary["total_time_min"] = total / 60
        wandb.finish()
        print(f"\n   W&B run complete")

    print(f"\n   Done! Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended Mamba Training")
    parser.add_argument("--model", default="130M")
    parser.add_argument("--dataset", default="fineweb", choices=["c4", "fineweb", "fineweb-edu", "mixed"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--output_dir", default="outputs_extended")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
