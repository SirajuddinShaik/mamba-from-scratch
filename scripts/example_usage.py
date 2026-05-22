#!/usr/bin/env python3
import torch

from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


def example_basic_usage():
    print("Example 1: Basic Mamba LM Usage")
    print("-" * 40)

    config = create_config("130M")
    model = MambaLM(config)

    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print()


def example_custom_config():
    print("Example 2: Custom Configuration")
    print("-" * 40)

    from mamba.models.config import MambaConfig

    config = MambaConfig(
        d_model=512,
        n_layer=12,
        vocab_size=32000,
        ssm_cfg={
            "d_state": 16,
            "expand": 2,
            "d_conv": 4,
        },
    )

    model = MambaLM(config)
    print(
        f"Custom model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    print()


def example_generation():
    print("Example 3: Text Generation")
    print("-" * 40)

    config = create_config("130M")
    model = MambaLM(config)
    model.eval()

    prompt_length = 10
    max_new_tokens = 50

    input_ids = torch.randint(0, config.vocab_size, (1, prompt_length))

    print(f"Generating {max_new_tokens} tokens from random prompt...")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_length + max_new_tokens,
            temperature=0.8,
            top_p=0.95,
        )

    print(f"Input:  {prompt_length} tokens")
    print(f"Output: {output.shape[1]} tokens")
    print(f"Generated: {output.shape[1] - prompt_length} new tokens")
    print()


def example_using_mamba_block():
    print("Example 4: Using Individual Mamba Block")
    print("-" * 40)

    from mamba.modules.mamba_block import MambaBlock

    d_model = 512
    batch_size = 4
    seq_len = 256

    mamba_block = MambaBlock(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape:  {x.shape}")

    with torch.no_grad():
        output = mamba_block(x)

    print(f"Output shape: {output.shape}")
    print(
        f"Block parameters: {sum(p.numel() for p in mamba_block.parameters()) / 1e6:.1f}M"
    )
    print()


def main():
    print("=" * 60)
    print("Mamba Examples")
    print("=" * 60)
    print()

    example_basic_usage()
    example_custom_config()
    example_generation()
    example_using_mamba_block()

    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
