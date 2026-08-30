"""Estimativas de recursos de workflow baseadas em execucoes historicas."""

from __future__ import annotations

from statistics import median
from typing import Any

from .workflow import ExperimentDefinition, ExperimentRun, TaskDefinition, TaskStatus

_ADDITIVE_METRICS = ("task_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd")
_PEAK_METRICS = ("peak_ram_mb", "rss_mb", "peak_vram_mb")
_USABLE_STATUSES = {TaskStatus.SUCCEEDED, TaskStatus.CACHED}


def estimate_workflow_resources(
    definition: ExperimentDefinition,
    history: list[ExperimentRun],
) -> dict[str, Any]:
    """Estima recursos de cada tarefa pela mediana de tentativas concluidas.

    Tarefas com perfil declardo usam apenas historico com a mesma configuracao
    e assinaturas de entrada. Sem perfil, a busca prioriza o ``task_id`` e
    usa o ``task_type`` como fallback explicito.
    Tentativas falhas nao entram na baseline de uma execucao bem-sucedida.
    """
    task_estimates = [
        _estimate_task_resources(task, history)
        for task in definition.tasks
    ]
    return {
        "definition_name": definition.name,
        "tasks": task_estimates,
        "resources": _aggregate_task_estimates(task_estimates),
    }


def _estimate_task_resources(
    task: TaskDefinition,
    history: list[ExperimentRun],
) -> dict[str, Any]:
    attempts, match_level = _select_matching_attempts(task, history)
    resources = [
        attempt.metrics.get("resources", {})
        for attempt in attempts
        if isinstance(attempt.metrics.get("resources", {}), dict)
    ]
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "match_level": match_level if attempts else "none",
        "sample_count": len(resources),
        "resources": {
            metric: _median_numeric(resources, metric)
            for metric in (*_ADDITIVE_METRICS, *_PEAK_METRICS)
        },
    }


def _select_matching_attempts(
    definition: TaskDefinition,
    history: list[ExperimentRun],
) -> tuple[list[Any], str]:
    if definition.config or definition.input_signatures:
        exact = _matching_attempts(
            history, task_id=definition.task_id, task_type=definition.task_type, definition=definition
        )
        same_profile = _matching_attempts(history, task_type=definition.task_type, definition=definition)
        return (exact, "task_id_and_profile") if exact else (same_profile, "task_profile")

    exact = _matching_attempts(history, task_id=definition.task_id)
    same_type = _matching_attempts(history, task_type=definition.task_type)
    return (exact, "task_id") if exact else (same_type, "task_type")


def _matching_attempts(
    history: list[ExperimentRun],
    *,
    task_id: str | None = None,
    task_type: str | None = None,
    definition: TaskDefinition | None = None,
) -> list[Any]:
    attempts = []
    for workflow in history:
        for task in workflow.tasks:
            matches = task.task_id == task_id if task_id is not None else task.task_type == task_type
            matches = matches and (task_type is None or task.task_type == task_type)
            matches = matches and (
                definition is None
                or (task.config == definition.config and task.input_signatures == definition.input_signatures)
            )
            if matches:
                attempts.extend(attempt for attempt in task.attempts if attempt.status in _USABLE_STATUSES)
    return attempts


def _aggregate_task_estimates(task_estimates: list[dict[str, Any]]) -> dict[str, float | None]:
    resources = [estimate["resources"] for estimate in task_estimates]
    summary = {metric: _sum_metric(resources, metric) for metric in _ADDITIVE_METRICS}
    summary["peak_ram_mb"] = _max_metric(resources, "peak_ram_mb", "rss_mb")
    summary["peak_vram_mb"] = _max_metric(resources, "peak_vram_mb")
    return summary


def _median_numeric(resources: list[dict[str, Any]], metric: str) -> float | None:
    values = [_numeric(resource.get(metric)) for resource in resources]
    present = [value for value in values if value is not None]
    return float(median(present)) if present else None


def _sum_metric(resources: list[dict[str, Any]], metric: str) -> float | None:
    values = [_numeric(resource.get(metric)) for resource in resources]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_metric(resources: list[dict[str, Any]], *metrics: str) -> float | None:
    values = [_numeric(resource.get(metric)) for resource in resources for metric in metrics]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None