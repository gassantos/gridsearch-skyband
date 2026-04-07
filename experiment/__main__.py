"""python -m experiment — CLI para execução de experimentos BERT-PLI.

Uso::

    uv run python -m experiment <config_path> [gpu_id ...]
    uv run python -m experiment config/experiments/BertPLI.config 0
    uv run python -m experiment config/experiments/BertPLI.config 0 1
    uv run python -m experiment config/experiments/BertPLI.config  # CPU
"""

import sys

from experiment import execute_experiment

if len(sys.argv) < 2:
    print("Uso: uv run python -m experiment <config_path> [gpu_id ...]")
    print("  Ex. (single GPU): uv run python -m experiment config/experiments/BertPLI.config 0")
    print("  Ex. (multi-GPU):  uv run python -m experiment config/experiments/BertPLI.config 0 1")
    print("  Ex. (CPU):        uv run python -m experiment config/experiments/BertPLI.config")
    sys.exit(1)

_gpus = [int(g) for g in sys.argv[2:]] if len(sys.argv) > 2 else None
execute_experiment(sys.argv[1], gpu_list=_gpus)
