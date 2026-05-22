#!/usr/bin/env python3
"""Evaluate official mamba-130m-hf on WikiText-103 for reference PPL."""
import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print("Loading official mamba-130m-hf...")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").cuda().eval()

print("Loading WikiText-103 validation...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
text = "\n\n".join([x["text"] for x in ds if len(x["text"]) > 20])
enc = tokenizer(text, return_tensors="pt")
input_ids = enc["input_ids"][0]
print(f"Total tokens: {input_ids.shape[0]}")

# Sliding window perplexity
seq_len = 512
stride = 512
total_loss = 0
count = 0
with torch.no_grad():
    for i in range(0, input_ids.shape[0] - seq_len, stride):
        batch = input_ids[i:i+seq_len].unsqueeze(0).cuda()
        out = model(batch)
        shift = out.logits[:, :-1, :]
        labels = batch[:, 1:]
        loss = F.cross_entropy(shift.reshape(-1, shift.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        count += 1
        if count >= 100:
            break

avg_loss = total_loss / count
ppl = math.exp(avg_loss)
print(f"\nOfficial mamba-130m-hf on WikiText-103 val:")
print(f"  Loss: {avg_loss:.4f}")
print(f"  PPL:  {ppl:.2f}")
print(f"  Windows evaluated: {count}")
