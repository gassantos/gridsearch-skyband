"""
Protocolos de domínio — Contratos estruturais (DIP)
====================================================

Define ``typing.Protocol`` para as dependências externas consumidas
pelo pipeline de execução (``run_experiment.py``).  Permite injetar
implementações alternativas (mocks, stubs, adaptadores) sem modificar
os módulos concretos em ``tools/``.

Uso::

    from gridsearch.protocols import InitFn, TrainFn, MetricsComputeFn

Autor: Gustavo Alexandre
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


# ============================================================================
# Protocolos de inicialização e treino
# ============================================================================


@runtime_checkable
class InitFn(Protocol):
    """Contrato para ``tools.init_tool.init_all``."""

    def __call__(
        self,
        config: Any,
        gpu_list: List[int],
        checkpoint: Any,
        mode: str,
    ) -> Any: ...


@runtime_checkable
class TrainFn(Protocol):
    """Contrato para ``tools.train_tool.train``."""

    def __call__(
        self,
        parameters: Any,
        config: Any,
        gpu_list: List[int],
    ) -> None: ...


# ============================================================================
# Protocolos de avaliação
# ============================================================================


@runtime_checkable
class ConvertTestResultsFn(Protocol):
    """Contrato para ``tools.eval_tool.convert_test_results_to_task1``."""

    def __call__(self, result_path: str) -> Dict[str, Any]: ...


@runtime_checkable
class ComputeMetricsFn(Protocol):
    """Contrato para ``tools.eval_tool.compute_metrics``."""

    def __call__(self, labels_path: str, predictions_path: str) -> Dict[str, Any]: ...
