# Tiny LLM Lab

Tiny LLM Lab is a deliberately readable, local-first project for training decoder-only
language models and chatting with saved checkpoints. It starts with two independent
architecture packages:

- **Transformer:** RMSNorm, rotary position embeddings (RoPE), causal scaled-dot-product
  attention, SwiGLU, and tied input/output embeddings.
- **Mamba:** a portable Mamba-1-style selective state-space model with causal depthwise
  convolution and an explicit recurrence that is easy to study.

The project is small enough to understand end to end, but includes real experiment
configuration, external dataset caching, BPE training, memory-mapped token storage,
validation/perplexity, cosine learning-rate decay, gradient accumulation and clipping,
safe tensor checkpoints, a model catalog, CLI inference, and a FastAPI web UI.

## Hardware acceleration

Training defaults to `device: "auto"`: CUDA is preferred when available, followed by Apple
MPS and then CPU. Windows installs use the project's CUDA 12.1 PyTorch wheel, so a supported
NVIDIA GPU is selected automatically. Confirm the environment before a long run:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Set `device` to `"cuda"` in the experiment configuration if CUDA should be required instead
of allowing a CPU fallback. Intel Macs with an AMD GPU remain CPU-only: MPS is unavailable
for that hardware, and PyTorch 2.2.2 is pinned because it is the final CPython 3.12 x86_64
macOS release. The pure-PyTorch Mamba reference scan works on CUDA but remains slower than
specialized fused kernels.

## Set up

```bash
uv sync
uv run tiny-llm --help
```

`uv` creates `.venv` and uses the committed lock file. Data is cached beneath
`data/cache/`; named checkpoints go beneath `artifacts/models/`. Both are ignored by Git.

## Train a model

Create a configuration with a chosen name, architecture, and approximate size preset:

```bash
uv run tiny-llm init-config --name my-story-model --architecture transformer --size micro
```
```bash
uv run tiny-llm init-config --name trans-small --architecture transformer --size xlarge
```

Open `experiment.json` to adjust dataset size, context, batch size, or training steps, then:

```bash
uv run tiny-llm train experiment.json
```

Training writes a resumable checkpoint every `save_interval` steps under
`artifacts/checkpoints/<model-name>/latest.pt`. Running the same command again resumes
automatically. Use `--no-resume` to deliberately start over; the first new save interval
then replaces the old checkpoint. Copy both `artifacts/checkpoints/` and `data/cache/`
when moving an interrupted run between machines.

For a quick pipeline smoke test, reduce each source's `max_documents` to `100`, `max_steps` to `2`,
`eval_interval` to `1`, and `eval_batches` to `1`. This verifies plumbing, not model quality.
Useful text generation requires far more tokens and steps. The included
`configs/transformer-micro.json` is a full starter experiment.

Every training invocation calls the data preparation layer. It downloads the requested
Hugging Face streams when absent and otherwise verifies/reuses its local snapshot. The
default is a round-robin mixture of 20,000 documents each from WikiText-103, SQuAD, and
BookCorpus. SQuAD records are rendered as context/question/answer text. Adjust each
source's `max_documents` or edit the `sources` list to control the size and composition.

To use Kaggle, install the optional dependency and configure credentials as documented by
Kaggle, then add a source with `provider` set to `kaggle` and `dataset` set to an
`owner/dataset` handle:

```bash
uv sync --extra kaggle
```

## Inspect and chat

```bash
uv run tiny-llm models
uv run tiny-llm chat my-story-model --prompt "Once upon a time"
```
```bash
uv run tiny-llm web
```

Open <http://127.0.0.1:8000>. The selector discovers all valid named checkpoints and shows
the architecture, preset, and exact parameter count. Only one model stays loaded in RAM.

## Choosing size

Presets select width/depth, while the exact parameter count depends on vocabulary and
architecture and is calculated after construction. Typical defaults are intentionally in
the low single-digit to low tens-of-millions range. `micro` is the sensible first run on
this Intel laptop. `tiny` is a longer experiment; `small` is mainly for a faster machine or
patience. You can edit the generated config for full control.

## Project map

```text
src/tiny_llm/
├── architectures/       # common interface plus one package per architecture
│   ├── transformer/
│   └── mamba/
├── data/                # external providers, snapshots, BPE tokenizer
├── training/            # packed token dataset and trainer
├── inference/           # sampling
├── web/                 # FastAPI API and dependency-free browser UI
├── catalog.py           # checkpoint persistence/discovery
├── config.py            # validated experiment schema and presets
└── cli.py               # user-facing commands
```

## Learning path and limitations

Read `config.py`, then `data/pipeline.py`, one architecture's `model.py`, and finally
`training/trainer.py`. The comments focus on why each important choice exists.

This is pretraining, not an instruction-tuned assistant: the UI continues text rather than
understanding chat roles. There is no KV cache yet, and the reference Mamba recomputes its
prompt for each generated token. Those are good next learning extensions. Models trained
on small corpora can memorize, hallucinate, and reproduce undesirable text; inspect the
dataset and outputs before sharing them.

## Development

```bash
uv run ruff check .
uv run pytest
```
