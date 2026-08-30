"""Agregação científica de métricas observadas em workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from .workflow import ExperimentDefinition, ExperimentRun, TaskStatus
from .workflow_planner import WorkflowPlanner

_ADDITIVE_METRICS = ("task_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd")
_PEAK_METRICS = ("peak_ram_mb", "rss_mb", "peak_vram_mb")


class MetricAggregation(StrEnum):
    """Políticas suportadas para uma métrica de qualidade."""

    LAST = "last"
    MIN = "min"
    MAX = "max"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass(frozen=True)
class MetricAggregationPolicy:
    """Regra declarativa de agregação para uma métrica de avaliação."""

    aggregation: MetricAggregation = MetricAggregation.LAST
    task_ids: tuple[str, ...] = ()
    weight_metric: str | None = None

    def __post_init__(self) -> None:
        if self.aggregation is MetricAggregation.WEIGHTED_AVERAGE and not self.weight_metric:
            raise ValueError("weighted_average requer weight_metric.")


def aggregate_workflow_run(
    workflow: ExperimentRun,
    definition: ExperimentDefinition | None = None,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
) -> dict[str, Any]:
    """Agrega consumo real, picos e duração crítica de um workflow executado.

    Todas as tentativas são incluídas nas métricas aditivas, inclusive as que
    falharam, pois representam consumo efetivamente observado.
    """
    attempts = [attempt for task in workflow.tasks for attempt in task.attempts]
    resources = [_resources(attempt.metrics) for attempt in attempts]
    summary = {
        metric: _sum_numeric(resources, metric)
        for metric in _ADDITIVE_METRICS
    }
    summary["peak_ram_mb"] = _max_numeric(resources, "peak_ram_mb", "rss_mb")
    summary["peak_vram_mb"] = _max_numeric(resources, "peak_vram_mb")

    critical_path = _critical_path(workflow, definition)
    return {
        "experiment_run_id": workflow.experiment_run_id,
        "status": workflow.status,
        "attempts": dict(Counter(attempt.status.value for attempt in attempts)),
        "resources": summary,
        "evaluation": aggregate_evaluation_metrics(workflow, evaluation_policies),
        "makespan_sec": _makespan_sec(attempts),
        "critical_path": critical_path,
    }


def aggregate_evaluation_metrics(
    workflow: ExperimentRun,
    policies: dict[str, MetricAggregationPolicy] | None = None,
) -> dict[str, float | None]:
    """Agrega métricas de qualidade de tentativas concluídas ou cacheadas.

    Métricas sem política usam ``last``, isto é, o último valor observado na
    ordem determinística das tarefas e tentativas do workflow.
    """
    policies = policies or {}
    metric_names = set(policies)
    points = _evaluation_points(workflow)
    metric_names.update(metric for _, values in points for metric in values)
    return {
        metric: _aggregate_evaluation_metric(points, policies.get(metric, MetricAggregationPolicy()), metric)
        for metric in sorted(metric_names)
    }


def _evaluation_points(workflow: ExperimentRun) -> list[tuple[str, dict[str, Any]]]:
    points: list[tuple[str, dict[str, Any]]] = []
    for task in workflow.tasks:
        for attempt in task.attempts:
            if attempt.status not in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}:
                continue
            evaluation = attempt.metrics.get("evaluation", {})
            if isinstance(evaluation, dict):
                points.append((task.task_id, evaluation))
    return points


def _aggregate_evaluation_metric(
    points: list[tuple[str, dict[str, Any]]],
    policy: MetricAggregationPolicy,
    metric: str,
) -> float | None:
    selected = [
        values for task_id, values in points
        if not policy.task_ids or task_id in policy.task_ids
    ]
    values = [_numeric(point.get(metric)) for point in selected]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    if policy.aggregation is MetricAggregation.LAST:
        return numeric_values[-1]
    if policy.aggregation is MetricAggregation.MIN:
        return min(numeric_values)
    if policy.aggregation is MetricAggregation.MAX:
        return max(numeric_values)
    if policy.aggregation is MetricAggregation.AVERAGE:
        return sum(numeric_values) / len(numeric_values)

    weights = [_numeric(point.get(policy.weight_metric)) for point in selected]
    pairs = [(value, weight) for value, weight in zip(values, weights) if value is not None and weight is not None]
    total_weight = sum(weight for _, weight in pairs)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def _resources(metrics: dict[str, Any]) -> dict[str, Any]:
    resources = metrics.get("resources", {})
    return resources if isinstance(resources, dict) else {}


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_numeric(resources: list[dict[str, Any]], metric: str) -> float | None:
    values = [_numeric(resource.get(metric)) for resource in resources]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _max_numeric(resources: list[dict[str, Any]], *metrics: str) -> float | None:
    values = [_numeric(resource.get(metric)) for resource in resources for metric in metrics]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _makespan_sec(attempts: list[Any]) -> float | None:
    starts = [_parse_timestamp(attempt.started_at) for attempt in attempts]
    ends = [_parse_timestamp(attempt.completed_at) for attempt in attempts]
    valid_starts = [timestamp for timestamp in starts if timestamp is not None]
    valid_ends = [timestamp for timestamp in ends if timestamp is not None]
    if not valid_starts or not valid_ends:
        return None
    return (max(valid_ends) - min(valid_starts)).total_seconds()


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _critical_path(
    workflow: ExperimentRun,
    definition: ExperimentDefinition | None,
) -> dict[str, Any] | None:
    if definition is None:
        return None

    duration_by_task = {
        task.task_id: _sum_numeric([_resources(attempt.metrics) for attempt in task.attempts], "task_time_sec")
        or 0.0
        for task in workflow.tasks
    }
    longest: dict[str, tuple[float, list[str]]] = {}
    for task in WorkflowPlanner().plan(definition):
        predecessors = [longest[dependency] for dependency in task.depends_on]
        previous_duration, previous_path = max(predecessors, default=(0.0, []), key=lambda item: item[0])
        longest[task.task_id] = (
            previous_duration + duration_by_task.get(task.task_id, 0.0),
            [*previous_path, task.task_id],
        )

    duration, task_ids = max(longest.values(), key=lambda item: item[0])
    return {"task_ids": task_ids, "duration_sec": duration}