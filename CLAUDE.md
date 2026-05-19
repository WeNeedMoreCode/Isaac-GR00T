# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run all CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run all GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Run a single test file
python -m pytest tests/gr00t/model/test_model_forward.py -v --timeout=300

# Run a single test function
python -m pytest tests/gr00t/model/test_model_forward.py::test_model_forward -v --timeout=300

# Run tests matching a keyword
python -m pytest tests/ -k "policy" -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

## Architecture

### Model registration pipeline

The training entry point (`gr00t/experiment/experiment.py`) resolves a model class to its training pipeline via two registries:

1. **Config registry** (`gr00t/configs/model/__init__.py`): Uses `register_model_config(shortname, configtype)` to collect all model config classes. It auto-discovers `.py` files in `gr00t/configs/model/` and imports them dynamically. `create_model_union_type()` builds a `typing.Union` of all registered configs for tyro CLI subcommands.

2. **Pipeline registry** (`gr00t/model/registry.py`): A simple `MODEL_REGISTRY: dict[model_cfg_class, pipeline_class]`. `gr00t/model/gr00t_n1d7/setup.py` registers `Gr00tN1d7Pipeline` against `Gr00tN1d7Config` at import time.

3. **Base config** (`gr00t/configs/base_config.py`): `Config` dataclass has a `model` field typed as `ModelUnionType`. At runtime `config.model._class_` identifies which pipeline class to instantiate from `MODEL_REGISTRY`.

To add a new model variant, create a config class in `gr00t/configs/model/`, register it with `@register_model_config`, and create a pipeline class that calls `register_model(config_cls, pipeline_cls)`.

### Embodiment tags and modality configs

`EmbodimentTag` (`gr00t/data/embodiment_tags.py`) is an enum that maps robot types to string values. Tags are **case-insensitive** and can be resolved by name or value via `EmbodimentTag.resolve()`.

- **Pretrain tags** (e.g. `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`, `XDOF`, `REAL_G1`) are baked into the base model and support zero-shot inference.
- **Posttrain tags** (e.g. `LIBERO_PANDA`, `SIMPLER_ENV_GOOGLE`, `UNITREE_G1_SONIC`) require a finetuned checkpoint.
- **`NEW_EMBODIMENT`** is the generic tag for custom robots. It requires a `--modality-config-path` during finetuning. Only one `NEW_EMBODIMENT` modality config may be registered per Python process.

The embodiment tag determines which `modality.json` is used (state/action keys, normalization, video keys). Modality configs for known tags live in `gr00t/configs/data/embodiment_configs.py`.

### Data format

GR00T uses a LeRobot v2 dataset format with an extra `meta/modality.json` file:

```
dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json        # state/action/video key mapping
    statistics.json      # normalization stats
  data/chunk-000/        # parquet files
  videos/chunk-000/      # mp4 files
```

Demo datasets are included under `demo_data/` for quick testing.

## Platform notes and known issues

- **`flash-attn` re-validates on every `uv run`:** This is expected uv behavior with URL-pinned wheel sources — it is not rebuilding from source. The wheel is cached locally and the check takes 2-3 seconds. Only affects x86_64.
- **`CUDA_HOME is unset` during fine-tuning:** Run `bash scripts/deployment/dgpu/install_deps.sh` once, or manually `export CUDA_HOME=/usr/local/cuda`.
- **CUDA 13.x (Thor, Spark):** PyTorch 2.7 pins Triton 3.3.1, which does not recognize CUDA 13. Run `uv run bash scripts/patch_triton_cuda13.sh` to fix.
- **GB300 (sm_103):** Triton 3.3.1 does not support this architecture. `torch.compile` will fail; use eager mode or TensorRT instead.
- **aarch64 video backend:** Only `torchcodec` is supported. `decord` and `pyav` are not supported on aarch64.
