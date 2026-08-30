"""Relatorios textuais e timeline estruturada de workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .aggregation import MetricAggregationPolicy, aggregate_workflow_run
from .workflow import ExperimentDefinition, ExperimentRun


def workflow_timeline(workflow: ExperimentRun) -> list[dict[str, Any]]:
    """Retorna tentativas ordenadas cronologicamente para visualizacao de timeline.

    Tentativas sem timestamp sao mantidas ao fim da linha do tempo, na ordem
    deterministica de tarefa e numero de tentativa.
    """
    events: list[dict[str, Any]] = []
    for task_index, task in enumerate(workflow.tasks):
        for attempt in task.attempts:
            start = _parse_timestamp(attempt.started_at)
            end = _parse_timestamp(attempt.completed_at)
            resource_duration = _numeric(attempt.metrics.get("resources", {}).get("task_time_sec"))
            duration = (end - start).total_seconds() if start and end else resource_duration
            events.append({
                "task_id": task.task_id,
                "task_name": task.name,
                "task_type": task.task_type,
                "attempt_id": attempt.attempt_id,
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
                "started_at": attempt.started_at,
                "completed_at": attempt.completed_at,
                "duration_sec": duration,
                "error_type": attempt.error_type,
                "error": attempt.error,
                "_sort_key": (start is None, start or datetime.max, task_index, attempt.attempt_number),  # noqa: DTZ901
            })

    events.sort(key=lambda event: event.pop("_sort_key"))
    origin = _parse_timestamp(events[0]["started_at"]) if events else None
    for event in events:
        start = _parse_timestamp(event["started_at"])
        event["relative_start_sec"] = (start - origin).total_seconds() if start and origin else None
    return events


def workflow_report(
    workflow: ExperimentRun,
    definition: ExperimentDefinition | None = None,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
) -> str:
    """Gera relatorio textual com resumo cientifico e historico de tentativas."""
    summary = aggregate_workflow_run(workflow, definition, evaluation_policies)
    lines = [
        "=" * 72,
        "RELATORIO DE WORKFLOW",
        "=" * 72,
        f"Execucao: {workflow.experiment_run_id}",
        f"Definicao: {workflow.definition_name}",
        f"Status: {workflow.status}",
        f"Tentativas: {summary['attempts']}",
        "",
        "RECURSOS:",
    ]
    lines.extend(_metric_lines(summary["resources"]))
    lines.append(f"  makespan_sec: {_format_value(summary['makespan_sec'])}")

    if summary["evaluation"]:
        lines.extend(["", "QUALIDADE:"])
        lines.extend(_metric_lines(summary["evaluation"]))

    critical_path = summary["critical_path"]
    if critical_path:
        lines.extend([
            "",
            f"CAMINHO CRITICO: {' -> '.join(critical_path['task_ids'])}",
            f"  duration_sec: {_format_value(critical_path['duration_sec'])}",
        ])

    lines.extend(["", "TIMELINE:"])
    for event in workflow_timeline(workflow):
        duration = _format_seconds(event["duration_sec"])
        relative_start = _format_seconds(event["relative_start_sec"])
        lines.append(
            f"  {event['task_id']}#{event['attempt_number']} | {event['status']} "
            f"| inicio+{relative_start} | duracao={duration}"
        )
        if event["error"]:
            lines.append(f"    erro[{event['error_type'] or 'unknown'}]: {event['error']}")

    lines.append("=" * 72)
    return "\n".join(lines)


def _metric_lines(metrics: dict[str, float | None]) -> list[str]:
    return [f"  {name}: {_format_value(value)}" for name, value in metrics.items()]


def _format_value(value: float | None) -> str:
    return "indisponivel" if value is None else f"{value:.6g}"


def _format_seconds(value: float | None) -> str:
    return "indisponivel" if value is None else f"{value:.6g}s"


def _numeric(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None