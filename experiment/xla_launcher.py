"""Launcher PJRT para executar um experimento em múltiplos cores TPU."""

from __future__ import annotations

import importlib
from typing import Any


def _run_xla_worker(rank: int, world_size: int, experiment_kwargs: dict[str, Any]) -> None:
    from .runner import execute_experiment

    execute_experiment(
        **experiment_kwargs,
        xla_rank=rank,
        xla_world_size=world_size,
    )


def launch_experiment(*, tpu_cores: int = 1, **experiment_kwargs: Any) -> dict[str, Any] | None:
    """Executa diretamente ou cria um worker PJRT para cada core TPU."""
    if tpu_cores < 1:
        raise ValueError("tpu_cores deve ser maior ou igual a 1")

    if tpu_cores == 1:
        from .runner import execute_experiment

        return execute_experiment(**experiment_kwargs)

    try:
        xmp = importlib.import_module("torch_xla.distributed.xla_multiprocessing")
    except ImportError as exc:
        raise RuntimeError(
            "Execução TPU multicore requer torch_xla. Instale com 'uv sync --extra tpu'."
        ) from exc

    xmp.spawn(
        _run_xla_worker,
        args=(tpu_cores, experiment_kwargs),
        nprocs=tpu_cores,
        start_method="spawn",
    )