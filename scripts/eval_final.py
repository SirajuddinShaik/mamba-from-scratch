#!/usr/bin/env python3
"""Final evaluation script for trained Mamba model."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from mamba.models.config import create_config
from mamba.models.mamba_lm import MambaLM
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F


class StreamDS(IterableDataset):
    def __init__(self, name, split, seq_len=512):
        self.seq_len = seq_len
        if name == 'wikitext103':
            self.ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split, streaming=True)
        elif name == 'wikitext2':
            self.ds = load_dataset('wikitext', 'wikitext-2-v1', split=split, streaming=True)
        elif name == 'penn':
            self.ds = load_dataset('penn-treebank', 'text', split=split, streaming=True)
        elif name == 'c4':
            self.ds = load_dataset('allenai/c4', 'en', split=split, streaming=True)
        self.name = name

    def __iter__(self):
        buf = []
        for item in self.ds:
            text = item.get('text', '')
            if len(text) < 20:
                continue
            toks = self._tokenize(text)
            buf.extend(toks)
            while len(buf) >= self.seq_len + 1:
                yield torch.tensor(buf[: self.seq_len + 1], dtype=torch.long)
                buf = buf[self.seq_len :]

    def _tokenize(self, text):
        # Simple tokenization fallback
        return self._tokenizer.encode(text, add_special_tokens=False)

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer


def evaluate(model, name, tokenizer, seq_len=512, max_batches=300):
    ds = StreamDS(name, 'validation', seq_len)
    ds.set_tokenizer(tokenizer)
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = batch.cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )
            if not torch.isnan(loss):
                losses.append(loss.item())
    avg = sum(losses) / len(losses) if losses else float('inf')
    return avg, math.exp(min(avg, 20))


def main():
    device = torch.device('cuda')
    config = create_config('130M')
    model = MambaLM(config).to(device)
    
    # Load the best model
    ckpt_path = 'outputs_v2/final_model.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f'Model: Mamba-130M (from scratch)')
    print(f'Params: {params:,} ({params/1e6:.1f}M)')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Total tokens trained: ~82M')
    print()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print('=' * 65)
    print(f'{"Dataset":<15} {"Loss":>8} {"PPL":>10}')
    print('=' * 65)
    
    all_results = {}
    for ds_name in ['wikitext103', 'wikitext2', 'penn', 'c4']:
        try:
            loss, ppl = evaluate(model, ds_name, tokenizer, max_batches=300)
            all_results[ds_name] = (loss, ppl)
            print(f'{ds_name:<15} {loss:>8.4f} {ppl:>10.1f}')
        except Exception as e:
            print(f'{ds_name:<15} ERROR: {str(e)[:40]}')
    
    print('=' * 65)
    print()
    print('--- Reference Comparison ---')
    print(f'{"Model":<30} {"Dataset":<12} {"PPL":>8}')
    print('-' * 50)
    print(f'{"Our Mamba-130M (82M toks)":<30} {"wikitext103":<12} {all_results.get("wikitext103", (0,0))[1]:>8.1f}')
    print(f'{"Our Mamba-130M (82M toks)":<30} {"wikitext2":<12} {all_results.get("wikitext2", (0,0))[1]:>8.1f}')
    print(f'{"Our Mamba-130M (82M toks)":<30} {"c4":<12} {all_results.get("c4", (0,0))[1]:>8.1f}')
    print('-' * 50)
    print(f'{"Official mamba-130m-hf (300B)":<30} {"wikitext103":<12} {"~25":>8}')
    print(f'{"Official mamba-130m-hf (300B)":<30} {"wikitext2":<12} {"~15":>8}')
    print(f'{"Official mamba-130m-hf (300B)":<30} {"ptb":<12} {"~20":>8}')
    print()
    print('Note: We trained on ~82M tokens (0.03% of official 300B).')
    print('PPL would converge toward reference with more training.')


if __name__ == '__main__':
    main()