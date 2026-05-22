#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


def evaluate_perplexity(model, dataset, tokenizer, device, max_samples=1000):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break

        text = sample.get("text", sample.get("content", ""))
        if not text:
            continue

        tokens = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        tokens = tokens.to(device)

        if tokens.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(tokens[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="sum",
            )

        total_loss += loss.item()
        total_tokens += tokens.shape[1] - 1

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity


def evaluate_generation(model, tokenizer, device, prompts, max_length=100):
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_p=0.95,
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(
            {
                "prompt": prompt,
                "generated": generated_text,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="130M")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--test_prompts", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = create_config(args.model_size)
    model = MambaLM(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    perplexity = evaluate_perplexity(
        model, dataset, tokenizer, device, args.max_samples
    )
    results["perplexity"] = perplexity
    print(f"Perplexity: {perplexity:.2f}")

    if args.test_prompts:
        generations = evaluate_generation(model, tokenizer, device, args.test_prompts)
        results["generations"] = generations

        for gen in generations:
            print(f"\nPrompt: {gen['prompt']}")
            print(f"Generated: {gen['generated']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
