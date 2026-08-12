from pathlib import Path

from tiny_llm.architectures import build_model, parameter_count
from tiny_llm.catalog import ModelCatalog
from tiny_llm.config import make_experiment
from tiny_llm.data.tokenizer import train_tokenizer


def test_catalog_round_trip(tmp_path: Path) -> None:
    tokenizer = train_tokenizer(
        ["small models make useful tests"] * 100, tmp_path / "tokenizer.json", 256, 1
    )
    experiment = make_experiment("test-model", "transformer", "micro")
    experiment.model.vocab_size = tokenizer.vocab_size
    experiment.model.d_model = 32
    experiment.model.n_layers = 1
    experiment.model.n_heads = 4
    model = build_model(experiment.model)
    catalog = ModelCatalog(tmp_path / "models")
    catalog.save(model, experiment, tmp_path / "tokenizer.json", parameter_count(model))
    loaded, loaded_tokenizer, loaded_config = catalog.load("test-model")
    assert loaded_config.training.model_name == "test-model"
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded.embedding.weight.shape == model.embedding.weight.shape
    assert catalog.list()[0].parameters == parameter_count(model)
