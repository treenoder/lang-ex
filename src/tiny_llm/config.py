"""Validated experiment configuration shared by CLI, trainer, and saved models."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataSourceConfig(BaseModel):
    """One normalized text source in a training-data mixture."""

    name: str
    provider: Literal["huggingface", "kaggle"] = "huggingface"
    dataset: str
    subset: str | None = None
    split: str = "train"
    text_column: str = "text"
    format: Literal["text", "question_answer"] = "text"
    # A finite sample makes runs reproducible and protects disk/RAM on a laptop.
    max_documents: int = Field(20_000, ge=100)
    streaming: bool = True


def _default_data_sources() -> list[DataSourceConfig]:
    return [
        DataSourceConfig(
            name="wikipedia",
            dataset="Salesforce/wikitext",
            subset="wikitext-103-raw-v1",
        ),
        DataSourceConfig(
            name="question-answering",
            dataset="rajpurkar/squad",
            format="question_answer",
        ),
        DataSourceConfig(name="books", dataset="jxie/bookcorpus"),
    ]


class DataConfig(BaseModel):
    """A balanced collection of sources materialized for one experiment."""

    sources: list[DataSourceConfig] = Field(default_factory=_default_data_sources, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_source(cls, value: Any) -> Any:
        """Keep configurations created before mixed datasets were introduced working."""
        if not isinstance(value, dict) or "sources" in value or "dataset" not in value:
            return value
        legacy = dict(value)
        legacy.setdefault("name", "primary")
        return {"sources": [legacy]}


class TokenizerConfig(BaseModel):
    vocab_size: int = Field(8_000, ge=256)
    min_frequency: int = Field(2, ge=1)


class ModelConfig(BaseModel):
    """Architecture-neutral shape. Architecture packages interpret extra fields."""

    architecture: Literal["transformer", "mamba"]
    # Real experiments use thousands; the lower bound also permits miniature tests.
    vocab_size: int = Field(8_000, ge=4)
    context_length: int = Field(256, ge=32)
    d_model: int = Field(256, ge=32)
    n_layers: int = Field(6, ge=1)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    # Transformer fields.
    n_heads: int = Field(8, ge=1)
    ff_multiplier: float = Field(4.0, ge=1.0)
    # Mamba fields. A deliberately small state keeps the reference scan practical.
    d_state: int = Field(16, ge=4)
    d_conv: int = Field(4, ge=2)
    expand: int = Field(2, ge=1)

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        if self.architecture == "transformer" and self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class TrainConfig(BaseModel):
    model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    size: Literal["micro", "tiny", "small", "medium", "large", "xlarge"] = "tiny"
    batch_size: int = Field(8, ge=1)
    gradient_accumulation: int = Field(4, ge=1)
    max_steps: int = Field(2_000, ge=1)
    learning_rate: float = Field(3e-4, gt=0)
    weight_decay: float = Field(0.1, ge=0)
    warmup_steps: int = Field(100, ge=0)
    eval_interval: int = Field(100, ge=1)
    eval_batches: int = Field(10, ge=1)
    save_interval: int = Field(500, ge=1)
    seed: int = 42
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    num_workers: int = Field(0, ge=0)


class ExperimentConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig
    training: TrainConfig


# Exact parameter counts are calculated after construction and written to metadata;
# names are intentionally approximate, not marketing labels.
SIZE_PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "transformer": {
        "micro": {"d_model": 128, "n_layers": 4, "n_heads": 4},
        "tiny": {"d_model": 256, "n_layers": 6, "n_heads": 8},
        "small": {"d_model": 384, "n_layers": 8, "n_heads": 8},
        "medium": {"d_model": 512, "n_layers": 12, "n_heads": 8},
        "large": {"d_model": 768, "n_layers": 16, "n_heads": 12},
        "xlarge": {"d_model": 1024, "n_layers": 24, "n_heads": 16},
    },
    "mamba": {
        "micro": {"d_model": 96, "n_layers": 4},
        "tiny": {"d_model": 192, "n_layers": 6},
        "small": {"d_model": 320, "n_layers": 8},
        "medium": {"d_model": 512, "n_layers": 12},
        "large": {"d_model": 768, "n_layers": 16},
        "xlarge": {"d_model": 1024, "n_layers": 24},
    },
}


def make_experiment(name: str, architecture: str, size: str) -> ExperimentConfig:
    """Create a safe preset while keeping every field editable in the saved JSON."""
    if architecture not in SIZE_PRESETS or size not in SIZE_PRESETS[architecture]:
        raise ValueError(f"Unknown architecture/size: {architecture}/{size}")
    shape = SIZE_PRESETS[architecture][size]
    model = ModelConfig(architecture=architecture, **shape)
    return ExperimentConfig(model=model, training=TrainConfig(model_name=name, size=size))
