"""A compact decoder-only Transformer using modern pre-norm and RoPE."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tiny_llm.architectures.base import CausalLM, CausalLMOutput
from tiny_llm.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE injects relative position without a learned position table."""

    def __init__(self, head_dim: int, max_length: int) -> None:
        super().__init__()
        inverse = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_length).float()
        angles = torch.outer(positions, inverse)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-2]
        cos = self.cos[:length][None, None, :, :]
        sin = self.sin[:length][None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def reshape(t: Tensor) -> Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = self.rope(reshape(q)), self.rope(reshape(k)), reshape(v)
        # PyTorch selects its best available scaled-dot-product kernel automatically.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.output(out.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int(config.d_model * config.ff_multiplier * 2 / 3)
        hidden = 64 * math.ceil(hidden / 64)  # friendly dimensions for matrix kernels
        self.gate_up = nn.Linear(config.d_model, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class TransformerLM(CausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying saves many parameters
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return CausalLMOutput(logits=logits, loss=loss)
