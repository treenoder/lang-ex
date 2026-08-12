"""Common language-model result and small helper functions."""

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None


class CausalLM(nn.Module):
    """Minimal protocol implemented by every architecture in this project."""

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        raise NotImplementedError


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (
        (p for p in model.parameters() if p.requires_grad) if trainable_only else model.parameters()
    )
    return sum(p.numel() for p in parameters)
