#!/usr/bin/env python3
"""
Mamba-130M: Pretrain from scratch on C4 with Weights & Biases logging.

Logs real-time metrics to wandb for LinkedIn portfolio proof.
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

# ── WANDB ───────────────────────────────────────────────────────────
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print(WARNING: wandb not installed. Install with: pip install wandb)


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer, seq_len=1024, split="train"):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        if dataset_name == "c4":
            self.ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        elif dataset_name == "wikitext103":
            self.ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
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
    SEQ = 1024
    LR = 6e-4
    WARMUP = 2000
    CLIP = 1.0
    LOG_EVERY = 200
    SAVE_EVERY = 5000
    MAX_STEPS = 20000
    DATASET = "c4"
    OUTPUT_DIR = Path.home() / "workspace/mamba/outputs_wandb"
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
    total_tokens_M = MAX_STEPS * BATCH * SEQ / 1e6
    print(f"\nModel: {MODEL} | Params: {params / 1e6:.1f}M")
    print(f"Seq len: {SEQ} | Dataset: {DATASET}")
    print(f"Total tokens: {total_tokens_M:.0f}M")

    # ── WANDB INIT ──────────────────────────────────────────────────────
    if HAS_WANDB:
        wandb.init(
            project="mamba-from-scratch",
            name=f"mamba-{MODEL}-c4-{MAX_STEPS}steps",
            config={
                "model": MODEL,
                "params_M": params / 1e6,
                "dataset": DATASET,
                "batch_size": BATCH,
                "seq_len": SEQ,
                "max_steps": MAX_STEPS,
                "total_tokens_M": total_tokens_M,
                "lr": LR,
                "warmup": WARMUP,
                "grad_clip": CLIP,
                "seed": SEED,
                "implementation": "from_scratch",
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            }
        )
        print("📊 W&B initialized: ", wandb.run.url if wandb.run else "N/A")

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
            if HAS_WANDB:
                wandb.log({"train/nan_skip": 1}, step=global_step)
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

            # ── WANDB LOG ─────────────────────────────────────────────
            if HAS_WANDB:
                wandb.log({
                    "train/loss": avg,
                    "train/perplexity": ppl,
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": tok_s,
                    "train/total_tokens_M": total_tok / 1e6,
                    "train/progress_pct": prog,
                }, step=global_step)

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

            # ── WANDB VAL LOG ─────────────────────────────────────────
            if HAS_WANDB:
                wandb.log({
                    "val/wikitext103_loss": val_loss,
                    "val/wikitext103_ppl": val_ppl,
                    "checkpoint/step": global_step,
                }, step=global_step)

                # Save checkpoint as artifact
                artifact = wandb.Artifact(f"checkpoint-{global_step}", type="model")
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

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
    for ds_name in ["wikitext103", "wikitext2", "c4", "ptb", "lambada", "penn", "pile", "pg19", "scielo", "text8", "enwik8", "openwebtext", "bookcorpus", "commoncrawl", "reddit", "stories", "arxiv", "pubmed", "freelaw", "hackernews", "philpapers", "ubuntu_irc", "youtube_subtitles", "gutenberg", "nih_exporter", "uspto", "pubmed_central", "pubmed_abstracts", "freelaw", "hackernews", "philpapers", "ubuntu_irc", "youtube_subtitles", "gutenberg", "nih_exporter", "uspto", "pubmed_central", "pubmed_abstracts" ]:
        try:
            loss, ppl = evaluate(model, ds_name, tokenizer, device, max_batches=200)
            results[ds_name] = {"loss": loss, "ppl": ppl}
            print(f"  {ds_name}: Loss={loss:.4f} PPL={ppl:.2f}")
        except Exception as e:
            print(f"  {ds_name}: ERROR: {str(e)[:40]}")

    # ── Summary ─────────────────────────────────────────────────────────
    total = time.time() - start
    summary = {
        "model": MODEL,
        "params_M": params / 1e6,
        "dataset": DATASET,
        "total_tokens_M": global_step * BATCH * SEQ / 1e6,
        "steps": global_step,
        "best_train_loss": best_loss,
        "total_time_min": total / 60,
        "avg_tok_s": global_step * BATCH * SEQ / total,
        "eval_results": results,
    }

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
    print(f"  (trained on 300B tokens = {300000/(global_step * BATCH * SEQ / 1e6):.0f}x more data)")

    # Save log
    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({**summary, "loss_curve": log_entries}, f, indent=2)
    print(f"  Log saved: {log_path}")

    # ── WANDB FINAL LOG ───────────────────────────────────────────────
    if HAS_WANDB:
        wandb.log({
            "final/best_train_loss": best_loss,
            "final/total_time_min": total / 60,
            "final/avg_throughput_tok_s": global_step * BATCH * SEQ / total,
            "final/total_tokens_M": global_step * BATCH * SEQ / 1e6,
        })

        # Log eval results
        for ds_name, r in results.items():
            wandb.log({
                f"final/{ds_name}_loss": r["loss"],
                f"final/{ds_name}_ppl": r["ppl"],
            })

        # Save final model artifact
        artifact = wandb.Artifact("final-model", type="model")
        artifact.add_file(str(final_path))
        wandb.log_artifact(artifact)

        # Finish
        wandb.finish()
        print(f"\n✅ W&B run complete: {wandb.run.url if wandb.run else 'N/A'}")


if __name__ == "__main__":
    train()
