"""Artefato versionado que prepara workflows para planejamento posterior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .aggregation import MetricAggregationPolicy, aggregate_workflow_run
from .estimation import estimate_workflow_resources
from .workflow import ExperimentDefinition, ExperimentRun

PLANNING_ARTIFACT_SCHEMA_VERSION = "1.0"


def build_planning_artifact(
    definition: ExperimentDefinition,
    history: Iterable[ExperimentRun] = (),
    *,
    workflow: ExperimentRun | None = None,
    constraints: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
) -> dict[str, Any]:
    """Consolida definicao, estimativas, observacoes e parametros de decisao.

    Restricoes e preferencias sao preservadas como dados declarativos. Sua
    interpretacao e intencionalmente deixada para a formulacao MILP posterior.
    """
    estimate = estimate_workflow_resources(definition, list(history))
    estimates_by_id = {task["task_id"]: task for task in estimate["tasks"]}
    return {
        "schema_version": PLANNING_ARTIFACT_SCHEMA_VERSION,
        "definition": {
            "name": definition.name,
            "experiment_type": definition.experiment_type,
            "schema_version": definition.schema_version,
        },
        "tasks": [
            _task_artifact(task, estimates_by_id[task.task_id])
            for task in definition.tasks
        ],
        "estimated_workflow_resources": estimate["resources"],
        "observed_workflow": (
            aggregate_workflow_run(workflow, definition, evaluation_policies)
            if workflow is not None else None
        ),
        "constraints": dict(constraints or {}),
        "preferences": dict(preferences or {}),
    }


def write_planning_artifact(artifact: dict[str, Any], path: Path) -> Path:
    """Escreve o artefato de planejamento como JSON UTF-8 formatado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)
    return path


def _task_artifact(task: Any, estimate: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "name": task.name,
        "task_type": task.task_type,
        "depends_on": list(task.depends_on),
        "required": task.required,
        "retry_policy": {
            "max_attempts": task.retry_policy.max_attempts,
            "retryable_error_types": list(task.retry_policy.retryable_error_types),
        },
        "config": dict(task.config),
        "input_signatures": dict(task.input_signatures),
        "estimated_resources": estimate["resources"],
        "estimation_evidence": {
            "match_level": estimate["match_level"],
            "sample_count": estimate["sample_count"],
        },
    }