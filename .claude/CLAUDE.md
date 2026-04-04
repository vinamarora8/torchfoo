# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

torchfoo is a Python utility library for PyTorch (>=2.7.0), providing reusable helpers focused on distributed training operations. Requires Python >=3.10.

## Build & Install

```bash
# Install in development mode (uses venv at ./venv)
source venv/bin/activate
pip install -e ".[dev]"
```

Build system: setuptools via pyproject.toml. Formatter: `black` (24.2.0). Test runner: `pytest`.

## Architecture

- `torchfoo/` — single-package library, importable as `torchfoo` or `tfoo`
- `torchfoo/distributed.py` — distributed training utilities: collective ops (`all_reduce_`, `all_reduce_sum_`, `all_concat`, `all_concat_jagged`, `all_equal`) with custom `torch.autograd.Function` subclasses for backprop-safe in-place operations; process group lifecycle (`setup_distributed`, `cleanup_distributed`); the `parallelize` decorator for multi-GPU function execution; helpers (`is_distributed`, `get_world_size`, `get_rank`, `rank_zero_only`)
- `torchfoo/ddp.py` — DDP module wrapping (`make_module_ddp`: SyncBatchNorm conversion + DistributedDataParallel)
- `torchfoo/__init__.py` exports `distributed` (also aliased as `dist`) and `ddp`
- Public API is defined via `__all__` in each module
- In-place operations follow PyTorch's trailing underscore convention (e.g. `all_reduce_`)
