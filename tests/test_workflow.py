"""Testes dos contratos P0 para workflows orientados a tarefas."""

import pytest

from experiment.task_executor import SequentialWorkflowExecutor
from experiment.workflow import (
    ExperimentDefinition,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskStatus,
    legacy_task_run,
)


def _legacy_result(status: str = "success") -> dict:
    return {
        "experiment": {
            "id": "experiment-1",
            "config_name": "result.json",
            "status": status,
            "timestamp_start": "2026-08-29T10:00:00",
            "timestamp_end": "2026-08-29T10:01:00",
        },
        "resources": {"train_time_sec": "60.00"},
        "evaluation": {"f1_score": 0.9},
        "logs": {"stderr_tail": ""},
    }


def test_experiment_definition_requires_a_task():
    with pytest.raises(ValueError, match="ao menos uma tarefa"):
        ExperimentDefinition(name="empty", tasks=())


def test_task_attempt_rejects_invalid_transition():
    attempt = TaskExecutionAttempt(attempt_id="attempt-1", attempt_number=1)

    with pytest.raises(ValueError, match="Transição inválida"):
        attempt.transition_to(TaskStatus.SUCCEEDED)


def test_legacy_result_becomes_a_single_task_workflow():
    workflow = legacy_task_run(_legacy_result())

    assert workflow.experiment_run_id == "experiment-1"
    assert len(workflow.tasks) == 1
    assert workflow.tasks[0].task_id == "legacy-main-task"
    assert workflow.tasks[0].attempts[0].status is TaskStatus.SUCCEEDED


def test_definition_rejects_duplicated_task_ids():
    task = TaskDefinition(task_id="train", name="Treino")
    with pytest.raises(ValueError, match="únicos"):
        ExperimentDefinition(name="duplicated", tasks=(task, task))


def test_sequential_executor_records_task_metrics_and_artifacts():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(TaskDefinition(task_id="prepare", name="Preparar dados"),),
    )
    executor = SequentialWorkflowExecutor(
        {"prepare": lambda: {"artifacts": {"dataset": "data.json"}}}
    )

    workflow = executor.execute(definition)

    attempt = workflow.tasks[0].attempts[0]
    assert workflow.status == "success"
    assert attempt.status is TaskStatus.SUCCEEDED
    assert attempt.metrics["resources"]["task_time_sec"] >= 0
    assert attempt.artifacts == {"dataset": "data.json"}


def test_sequential_executor_skips_task_with_failed_dependency():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="first", name="Primeira"),
            TaskDefinition(task_id="next", name="Seguinte", depends_on=("first",)),
        ),
    )
    executor = SequentialWorkflowExecutor(
        {"first": lambda: (_ for _ in ()).throw(RuntimeError("falhou")), "next": lambda: {}}
    )

    workflow = executor.execute(definition)

    assert workflow.status == "failed"
    assert workflow.tasks[0].status is TaskStatus.FAILED
    assert workflow.tasks[1].status is TaskStatus.SKIPPED