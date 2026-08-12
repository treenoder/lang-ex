"""Architecture factory: the rest of the application never imports concrete models."""

from tiny_llm.architectures.base import CausalLM, parameter_count
from tiny_llm.config import ModelConfig


def build_model(config: ModelConfig) -> CausalLM:
    # Local imports keep architecture dependencies isolated as new packages are added.
    if config.architecture == "transformer":
        from tiny_llm.architectures.transformer import TransformerLM

        return TransformerLM(config)
    if config.architecture == "mamba":
        from tiny_llm.architectures.mamba import MambaLM

        return MambaLM(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


__all__ = ["build_model", "parameter_count"]
