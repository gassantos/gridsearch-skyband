"""Testes da timeline e do relatorio de workflows."""

from experiment.workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)
from experiment.workflow_reporting import workflow_report, workflow_timeline


def _workflow() -> ExperimentRun:
    return ExperimentRun(
        "run-1", "workflow", "failed",
        [
            TaskRun("train", "Treinar", "train", TaskStatus.FAILED, [
                TaskExecutionAttempt(
                    "train-1", 1, TaskStatus.FAILED,
                    "2026-01-01T10:02:00+00:00", "2026-01-01T10:04:00+00:00",
                    {"resources": {"task_time_sec": 120, "cost_usd": 1.5}},
                    error="temporarily unavailable", error_type="TimeoutError",
                ),
            ]),
            TaskRun("prepare", "Preparar", "prepare", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt(
                    "prepare-1", 1, TaskStatus.SUCCEEDED,
                    "2026-01-01T10:00:00+00:00", "2026-01-01T10:02:00+00:00",
                    {"resources": {"task_time_sec": 120}, "evaluation": {"accuracy": 0.8}},
                ),
            ]),
            TaskRun("publish", "Publicar", "publish", TaskStatus.SKIPPED, [
                TaskExecutionAttempt("publish-1", 1, TaskStatus.SKIPPED),
            ]),
        ],
    )


def test_workflow_timeline_orders_attempts_and_calculates_relative_time():
    timeline = workflow_timeline(_workflow())

    assert [event["task_id"] for event in timeline] == ["prepare", "train", "publish"]
    assert timeline[0]["relative_start_sec"] == 0.0
    assert timeline[1]["relative_start_sec"] == 120.0
    assert timeline[1]["duration_sec"] == 120.0
    assert timeline[2]["relative_start_sec"] is None


def test_workflow_report_includes_summary_critical_path_and_failure_details():
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("prepare", "Preparar"),
            TaskDefinition("train", "Treinar", depends_on=("prepare",)),
            TaskDefinition("publish", "Publicar", depends_on=("train",)),
        ),
    )

    report = workflow_report(_workflow(), definition)

    assert "RELATORIO DE WORKFLOW" in report
    assert "cost_usd: 1.5" in report
    assert "accuracy: 0.8" in report
    assert "CAMINHO CRITICO: prepare -> train" in report
    assert "erro[TimeoutError]: temporarily unavailable" in report