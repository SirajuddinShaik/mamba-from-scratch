from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    fused_add_norm: bool = False
    residual_in_fp32: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = True


def create_config(model_size: str):
    configs = {
        "130M": MambaConfig(d_model=768, n_layer=24),
        "370M": MambaConfig(d_model=1024, n_layer=48),
        "790M": MambaConfig(d_model=1536, n_layer=48),
        "1.4B": MambaConfig(d_model=2048, n_layer=48),
        "2.8B": MambaConfig(d_model=2560, n_layer=64),
    }
    return configs.get(model_size)
