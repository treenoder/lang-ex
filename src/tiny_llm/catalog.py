"""Atomic checkpoint storage and discovery for both CLI and web application."""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file, save_file

from tiny_llm.architectures import build_model
from tiny_llm.architectures.base import CausalLM
from tiny_llm.config import PROJECT_ROOT, ExperimentConfig
from tiny_llm.data.tokenizer import BPETokenizer


@dataclass(slots=True)
class ModelRecord:
    name: str
    architecture: str
    size: str
    parameters: int
    updated_at: str
    path: Path

    @property
    def formatted_parameters(self) -> str:
        if self.parameters >= 1_000_000:
            return f"{self.parameters / 1_000_000:.2f}M"
        return f"{self.parameters / 1_000:.1f}K"


class ModelCatalog:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "artifacts" / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.root / name

    def list(self) -> list[ModelRecord]:
        records = []
        for metadata_path in self.root.glob("*/metadata.json"):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                records.append(ModelRecord(path=metadata_path.parent, **data))
            except (OSError, ValueError, TypeError):
                continue  # An interrupted or manually edited model is not loadable.
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def save(
        self,
        model: CausalLM,
        experiment: ExperimentConfig,
        tokenizer_path: Path,
        parameters: int,
    ) -> Path:
        target = self.path_for(experiment.training.model_name)
        target.mkdir(parents=True, exist_ok=True)
        # Safetensors rejects tied storage, so save one canonical copy per shared tensor.
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        state.pop("lm_head.weight", None)
        save_file(state, target / "model.safetensors")
        (target / "config.json").write_text(experiment.model_dump_json(indent=2), encoding="utf-8")
        shutil.copy2(tokenizer_path, target / "tokenizer.json")
        metadata = {
            "name": experiment.training.model_name,
            "architecture": experiment.model.architecture,
            "size": experiment.training.size,
            "parameters": parameters,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return target

    def load(
        self, name: str, device: str = "cpu"
    ) -> tuple[CausalLM, BPETokenizer, ExperimentConfig]:
        path = self.path_for(name)
        experiment = ExperimentConfig.model_validate_json((path / "config.json").read_text())
        model = build_model(experiment.model)
        state = load_file(path / "model.safetensors", device=device)
        # strict=False is intentional: lm_head is tied to embedding and stored only once.
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model, BPETokenizer.load(path / "tokenizer.json"), experiment
