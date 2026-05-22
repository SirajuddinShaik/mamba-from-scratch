#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mamba.models.mamba_lm import MambaLM
from mamba.models.config import create_config


def test_forward_pass():
    print("Testing Mamba forward pass...")

    config = create_config("130M")
    model = MambaLM(config)

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Expected {expected_shape}, got {logits.shape}"
    )

    print(f"✓ Forward pass successful. Output shape: {logits.shape}")
    print(
        f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    return True


def test_generation():
    print("\nTesting text generation...")

    config = create_config("130M")
    model = MambaLM(config)
    model.eval()

    batch_size = 1
    prompt_length = 10
    max_new_tokens = 20

    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_length))

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=prompt_length + max_new_tokens,
            temperature=1.0,
        )

    expected_length = prompt_length + max_new_tokens
    assert output.shape[1] == expected_length, (
        f"Expected length {expected_length}, got {output.shape[1]}"
    )

    print(f"✓ Generation successful. Generated {max_new_tokens} new tokens")
    print(f"  Input length: {prompt_length}, Output length: {output.shape[1]}")

    return True


def test_mamba_block():
    print("\nTesting Mamba block...")

    from mamba.modules.mamba_block import MambaBlock

    d_model = 768
    batch_size = 2
    seq_len = 64

    block = MambaBlock(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = block(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    print(f"✓ Mamba block successful. Input/output shape: {output.shape}")
    print(
        f"  Block parameters: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M"
    )

    return True


def test_all_model_sizes():
    print("\nTesting all model sizes...")

    sizes = ["130M", "370M", "790M", "1.4B", "2.8B"]
    batch_size = 1
    seq_len = 32

    for size in sizes:
        config = create_config(size)
        model = MambaLM(config)

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ {size:>4}: {param_count:>6.1f}M params, output shape {logits.shape}")

    return True


def main():
    print("=" * 60)
    print("Mamba Implementation Tests")
    print("=" * 60)

    tests = [
        ("Mamba Block", test_mamba_block),
        ("Forward Pass", test_forward_pass),
        ("Generation", test_generation),
        ("Model Sizes", test_all_model_sizes),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
