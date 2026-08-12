"""Readable Mamba-1-style selective state-space language model.

This is a portable reference implementation, not the fused CUDA implementation from
state-spaces/mamba. Its explicit time scan is excellent for understanding the algorithm
and works on macOS CPU, but it is slower than specialized kernels.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tiny_llm.architectures.base import CausalLM, CausalLMOutput
from tiny_llm.architectures.transformer.model import RMSNorm
from tiny_llm.config import ModelConfig


class SelectiveSSM(nn.Module):
    """Input-dependent discretized state-space recurrence (the heart of Mamba)."""

    def __init__(self, inner: int, state: int) -> None:
        super().__init__()
        self.state = state
        self.delta_bc = nn.Linear(inner, inner + state * 2, bias=False)
        # Negative A produces stable decay. Store log(-A) for easy optimization.
        base = torch.arange(1, state + 1, dtype=torch.float32).repeat(inner, 1)
        self.a_log = nn.Parameter(base.log())
        self.d = nn.Parameter(torch.ones(inner))

    def forward(self, x: Tensor) -> Tensor:
        delta, b, c = torch.split(self.delta_bc(x), [x.shape[-1], self.state, self.state], -1)
        delta = F.softplus(delta)
        a = -self.a_log.exp()
        hidden = x.new_zeros(x.shape[0], x.shape[-1], self.state)
        outputs: list[Tensor] = []
        # Each token chooses how quickly state changes (delta) and what to read/write.
        for step in range(x.shape[1]):
            dt = delta[:, step, :, None]
            decay = torch.exp(dt * a[None, :, :])
            drive = dt * b[:, step, None, :] * x[:, step, :, None]
            hidden = decay * hidden + drive
            readout = (hidden * c[:, step, None, :]).sum(-1)
            outputs.append(readout + self.d * x[:, step])
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        inner = config.d_model * config.expand
        self.norm = RMSNorm(config.d_model)
        self.input_projection = nn.Linear(config.d_model, inner * 2, bias=False)
        # Depthwise causal convolution learns short-range patterns before the SSM.
        self.conv = nn.Conv1d(inner, inner, config.d_conv, groups=inner, padding=config.d_conv - 1)
        self.ssm = SelectiveSSM(inner, config.d_state)
        self.output_projection = nn.Linear(inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, residual: Tensor) -> Tensor:
        x, gate = self.input_projection(self.norm(residual)).chunk(2, dim=-1)
        length = x.shape[1]
        x = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = self.ssm(F.silu(x)) * F.silu(gate)
        return residual + self.dropout(self.output_projection(x))


class MambaLM(CausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(MambaBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None if labels is None else F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return CausalLMOutput(logits=logits, loss=loss)
