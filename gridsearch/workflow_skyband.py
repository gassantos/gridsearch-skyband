"""Consultas Skyband para workflows e tarefas monitoradas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from experiment.aggregation import MetricAggregationPolicy, aggregate_workflow_run
from experiment.workflow import ExperimentDefinition, ExperimentRun, TaskRun, TaskStatus

from .dominance import skyband_query


def workflow_skyband_query(
    workflows: Sequence[ExperimentRun],
    *,
    definitions: Mapping[str, ExperimentDefinition] | None = None,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
    **skyband_options: Any,
) -> list[dict[str, Any]]:
    """Executa Skyband entre resumos científicos de workflows completos."""
    points = [
        workflow_to_skyband_point(
            workflow,
            definition=definitions.get(workflow.definition_name) if definitions else None,
            evaluation_policies=evaluation_policies,
        )
        for workflow in workflows
    ]
    return skyband_query(points, **skyband_options)


def task_skyband_query(
    workflows: Sequence[ExperimentRun],
    task_id: str,
    *,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
    **skyband_options: Any,
) -> list[dict[str, Any]]:
    """Executa Skyband entre ocorrências da mesma tarefa em vários workflows.

    Uma tarefa cacheada é comparável a uma bem-sucedida. Tarefas ausentes ou
    não concluídas não participam da fronteira.
    """
    points = [
        task_to_skyband_point(workflow, task, evaluation_policies)
        for workflow in workflows
        for task in workflow.tasks
        if task.task_id == task_id
        and task.status in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
    ]
    return skyband_query(points, **skyband_options)


def workflow_to_skyband_point(
    workflow: ExperimentRun,
    *,
    definition: ExperimentDefinition | None = None,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
) -> dict[str, Any]:
    """Projeta a agregação científica de um workflow no contrato do Skyband."""
    summary = aggregate_workflow_run(workflow, definition, evaluation_policies)
    resources = {**summary["resources"], "makespan_sec": summary["makespan_sec"]}
    return {
        "status": workflow.status,
        "experiment_run_id": workflow.experiment_run_id,
        "definition_name": workflow.definition_name,
        "resources": resources,
        "evaluation": summary["evaluation"],
        "workflow_summary": summary,
    }


def task_to_skyband_point(
    workflow: ExperimentRun,
    task: TaskRun,
    evaluation_policies: dict[str, MetricAggregationPolicy] | None = None,
) -> dict[str, Any]:
    """Projeta uma tarefa e suas tentativas como ponto comparável pelo Skyband."""
    task_workflow = ExperimentRun(
        experiment_run_id=workflow.experiment_run_id,
        definition_name=workflow.definition_name,
        status="success" if task.status in {TaskStatus.SUCCEEDED, TaskStatus.CACHED} else "failed",
        tasks=[task],
    )
    point = workflow_to_skyband_point(
        task_workflow, evaluation_policies=evaluation_policies
    )
    point.update({"task_id": task.task_id, "task_name": task.name, "task_type": task.task_type})
    return point