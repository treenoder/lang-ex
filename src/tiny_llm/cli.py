"""Command-line entry points. Run ``uv run tiny-llm --help`` for discovery."""

import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from tiny_llm.catalog import ModelCatalog
from tiny_llm.config import ExperimentConfig, make_experiment
from tiny_llm.inference import GenerationConfig, generate
from tiny_llm.training import train

app = typer.Typer(no_args_is_help=True, help="Train and chat with educational tiny LLMs.")


@app.command("init-config")
def init_config(
    name: Annotated[str, typer.Option(help="Unique model/checkpoint name")],
    architecture: Annotated[str, typer.Option(help="transformer or mamba")] = "transformer",
    size: Annotated[str, typer.Option(help="micro, tiny, or small preset")] = "tiny",
    output: Annotated[Path, typer.Option(help="JSON configuration to create")] = Path(
        "experiment.json"
    ),
) -> None:
    """Create an editable experiment file without beginning a download or training run."""
    experiment = make_experiment(name, architecture, size)
    output.write_text(experiment.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Wrote {output}. Review it, then run: tiny-llm train {output}")


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Argument(exists=True, readable=True)],
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Continue from the latest checkpoint")
    ] = True,
) -> None:
    """Train from an experiment JSON file and save the named model."""
    experiment = ExperimentConfig.model_validate_json(config.read_text(encoding="utf-8"))

    def display(event: dict) -> None:
        typer.echo(json.dumps(event, ensure_ascii=False))

    path = train(experiment, callback=display, resume=resume)
    typer.secho(f"Saved model to {path}", fg=typer.colors.GREEN)


@app.command("models")
def list_models() -> None:
    """List checkpoints available to the web UI."""
    records = ModelCatalog().list()
    if not records:
        typer.echo("No trained models yet.")
    for item in records:
        typer.echo(f"{item.name:24} {item.architecture:12} {item.formatted_parameters:>10}")


@app.command("chat")
def chat_command(
    model_name: Annotated[str, typer.Argument(help="Name shown by the models command")],
    prompt: Annotated[str, typer.Option(prompt=True)],
    max_new_tokens: Annotated[int, typer.Option(min=1, max=512)] = 100,
    temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.8,
) -> None:
    """Generate once in the terminal (useful before launching the server)."""
    model, tokenizer, _ = ModelCatalog().load(model_name)
    typer.echo(generate(model, tokenizer, prompt, GenerationConfig(max_new_tokens, temperature)))


@app.command("web")
def web(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Start the local chat web application."""
    uvicorn.run("tiny_llm.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
