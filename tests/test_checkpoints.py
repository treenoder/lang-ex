import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from tiny_llm.architectures import build_model
from tiny_llm.config import make_experiment
from tiny_llm.training.trainer import _load_checkpoint, _save_checkpoint


def test_checkpoint_restores_training_state(tmp_path: Path) -> None:
    config = make_experiment("resume-test", "transformer", "micro")
    config.model.vocab_size = 256
    config.model.d_model = 32
    config.model.n_layers = 1
    config.model.n_heads = 4
    model = build_model(config.model)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)

    tokens = torch.randint(0, 256, (2, 12))
    model(tokens, tokens).loss.backward()  # type: ignore[union-attr]
    optimizer.step()
    scheduler.step()
    expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint = tmp_path / "latest.pt"
    _save_checkpoint(checkpoint, model, optimizer, scheduler, config, step=23)
    expected_random = (random.random(), float(np.random.random()), float(torch.rand(1)))

    for parameter in model.parameters():
        parameter.data.zero_()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restored_step = _load_checkpoint(
        checkpoint, model, optimizer, scheduler, config, torch.device("cpu")
    )

    assert restored_step == 23
    assert scheduler.last_epoch == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name])
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected_random


def test_checkpoint_rejects_incompatible_model(tmp_path: Path) -> None:
    config = make_experiment("resume-test", "transformer", "micro")
    model = build_model(config.model)
    optimizer = AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    checkpoint = tmp_path / "latest.pt"
    _save_checkpoint(checkpoint, model, optimizer, scheduler, config, step=1)

    changed = config.model_copy(
        update={"model": config.model.model_copy(update={"n_layers": config.model.n_layers + 1})}
    )
    changed_model = build_model(changed.model)
    changed_optimizer = AdamW(changed_model.parameters())
    changed_scheduler = torch.optim.lr_scheduler.LambdaLR(changed_optimizer, lambda step: 1.0)

    try:
        _load_checkpoint(
            checkpoint,
            changed_model,
            changed_optimizer,
            changed_scheduler,
            changed,
            torch.device("cpu"),
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("An incompatible checkpoint was accepted")
