"""Autoregressive text generation shared by the CLI and web UI."""

from dataclasses import dataclass

import torch

from tiny_llm.architectures.base import CausalLM
from tiny_llm.data.tokenizer import BPETokenizer


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.1


@torch.inference_mode()
def generate(
    model: CausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    config: GenerationConfig,
    device: str = "cpu",
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.bos_id]
    context_length = model.config.context_length  # type: ignore[attr-defined]
    generated = list(ids)
    for _ in range(config.max_new_tokens):
        window = torch.tensor([generated[-context_length:]], device=device)
        logits = model(window).logits[0, -1].float()
        # Penalize tokens already emitted to reduce short loops in tiny models.
        unique = torch.tensor(list(set(generated)), device=device)
        logits[unique] = torch.where(
            logits[unique] < 0,
            logits[unique] * config.repetition_penalty,
            logits[unique] / config.repetition_penalty,
        )
        if config.temperature <= 0:
            next_id = int(logits.argmax())
        else:
            logits /= config.temperature
            k = min(max(1, config.top_k), logits.numel())
            values, indices = torch.topk(logits, k)
            next_id = int(indices[torch.multinomial(values.softmax(-1), 1)])
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
    return tokenizer.decode(generated)
