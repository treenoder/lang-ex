import pytest
import torch

from tiny_llm.architectures import build_model, parameter_count
from tiny_llm.config import ModelConfig


@pytest.mark.parametrize("architecture", ["transformer", "mamba"])
def test_model_forward_and_backward(architecture: str) -> None:
    config = ModelConfig(
        architecture=architecture,
        vocab_size=256,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_state=4,
        expand=1,
        dropout=0,
    )
    model = build_model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 12))
    output = model(tokens, tokens)
    assert output.logits.shape == (2, 12, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert parameter_count(model) > 0


def test_transformer_is_causal() -> None:
    config = ModelConfig(
        architecture="transformer",
        vocab_size=256,
        context_length=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0,
    )
    model = build_model(config).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 99, 100]])
    with torch.no_grad():
        a, b = model(first).logits, model(second).logits
    torch.testing.assert_close(a[:, :2], b[:, :2])
