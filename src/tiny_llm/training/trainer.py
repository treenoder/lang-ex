"""A complete but intentionally compact next-token training loop."""

import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tiny_llm.architectures import build_model, parameter_count
from tiny_llm.catalog import ModelCatalog
from tiny_llm.config import PROJECT_ROOT, ExperimentConfig
from tiny_llm.data.pipeline import iter_corpus, prepare_corpus
from tiny_llm.data.tokenizer import BPETokenizer, train_tokenizer
from tiny_llm.training.dataset import PackedTokenDataset, prepare_tokens

ProgressCallback = Callable[[dict[str, float | int | str]], None]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, batches: int) -> float:
    model.eval()
    losses = []
    for index, (inputs, labels) in enumerate(loader):
        if index >= batches:
            break
        output = model(inputs.to(device), labels.to(device))
        losses.append(float(output.loss))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config: ExperimentConfig, callback: ProgressCallback | None = None) -> Path:
    """Fetch data, train tokenizer/model, evaluate, and publish a named checkpoint."""
    report = callback or (lambda event: print(event, flush=True))
    _seed_everything(config.training.seed)
    report({"stage": "data", "message": "Fetching or opening cached training data"})
    corpus = prepare_corpus(config.data)
    experiment_cache = corpus.parent
    tokenizer_path = experiment_cache / (
        f"tokenizer-v{config.tokenizer.vocab_size}-f{config.tokenizer.min_frequency}.json"
    )
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        report({"stage": "tokenizer", "message": "Training BPE tokenizer"})
        tokenizer = train_tokenizer(
            iter_corpus(corpus),
            tokenizer_path,
            config.tokenizer.vocab_size,
            config.tokenizer.min_frequency,
        )

    # A small corpus can produce fewer symbols than requested; persist the exact value.
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"vocab_size": tokenizer.vocab_size})}
    )
    tokens = prepare_tokens(corpus, tokenizer, experiment_cache / tokenizer_path.stem)
    train_data = PackedTokenDataset(tokens, config.model.context_length)
    validation_data = PackedTokenDataset(tokens, config.model.context_length, validation=True)
    if not len(train_data) or not len(validation_data):
        raise ValueError("Corpus is too small for the selected context length")

    train_loader = DataLoader(
        train_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )
    validation_loader = DataLoader(validation_data, batch_size=config.training.batch_size)
    device = resolve_device(config.training.device)
    model = build_model(config.model).to(device)
    count = parameter_count(model)
    report({"stage": "training", "device": str(device), "parameters": count})
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    def learning_rate(step: int) -> float:
        if step < config.training.warmup_steps:
            return (step + 1) / max(1, config.training.warmup_steps)
        progress = (step - config.training.warmup_steps) / max(
            1, config.training.max_steps - config.training.warmup_steps
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    iterator = iter(train_loader)
    model.train()
    started = time.monotonic()
    for step in range(1, config.training.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.training.gradient_accumulation):
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)
            output = model(inputs.to(device), labels.to(device))
            loss = output.loss / config.training.gradient_accumulation
            loss.backward()
            accumulated_loss += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % config.training.eval_interval == 0:
            validation_loss = evaluate(
                model, validation_loader, device, config.training.eval_batches
            )
            report(
                {
                    "stage": "training",
                    "step": step,
                    "train_loss": accumulated_loss,
                    "validation_loss": validation_loss,
                    "perplexity": math.exp(min(20, validation_loss)),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            )

    output = ModelCatalog(PROJECT_ROOT / "artifacts" / "models").save(
        model, config, tokenizer_path, count
    )
    report({"stage": "complete", "path": str(output)})
    return output
