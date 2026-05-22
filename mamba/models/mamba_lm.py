import torch
import torch.nn as nn
from mamba.modules.mamba_block import MambaBlock
from mamba.modules.rms_norm import RMSNorm
from mamba.models.config import MambaConfig, create_config


class MambaLM(nn.Module):
    def __init__(self, config: MambaConfig, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )

        self.embeddings = nn.Embedding(vocab_size, config.d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=config.d_model,
                    layer_idx=i,
                    **config.ssm_cfg,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        # Pre-norm for each Mamba layer (was missing in original)
        self.norms = nn.ModuleList(
            [
                RMSNorm(config.d_model, eps=1e-6)
                if config.rms_norm
                else nn.LayerNorm(config.d_model, eps=1e-6)
                for _ in range(config.n_layer)
            ]
        )

        self.norm_f = (
            RMSNorm(config.d_model, eps=1e-6)
            if config.rms_norm
            else nn.LayerNorm(config.d_model, eps=1e-6)
        )

        self.lm_head = nn.Linear(
            config.d_model, vocab_size, bias=False, **factory_kwargs
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        hidden_states = self.embeddings(input_ids)

        for norm, layer in zip(self.norms, self.layers):
            hidden_states = hidden_states + layer(
                norm(hidden_states), inference_params=inference_params
            )

        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=0, top_p=1.0):
        self.eval()
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]

        for _ in range(max_length - current_length):
            with torch.no_grad():
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
