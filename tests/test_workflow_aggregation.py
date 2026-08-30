"""Testes da agregação científica de métricas por workflow."""

import pytest

from experiment.aggregation import (
    MetricAggregation,
    MetricAggregationPolicy,
    aggregate_evaluation_metrics,
    aggregate_workflow_run,
)
from experiment.workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)


def _attempt(attempt_id, start, end, **resources):
    return TaskExecutionAttempt(
        attempt_id=attempt_id,
        attempt_number=1,
        status=TaskStatus.SUCCEEDED,
        started_at=start,
        completed_at=end,
        metrics={"resources": resources},
    )


def test_aggregate_workflow_sums_attempts_and_calculates_makespan_and_critical_path():
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("prepare", "Preparar"),
            TaskDefinition("train", "Treinar", depends_on=("prepare",)),
            TaskDefinition("export", "Exportar", depends_on=("prepare",)),
        ),
    )
    workflow = ExperimentRun(
        "run-1", "workflow", "success",
        [
            TaskRun("prepare", "Preparar", "prepare", TaskStatus.SUCCEEDED, [
                _attempt("prepare-1", "2026-01-01T10:00:00+00:00", "2026-01-01T10:02:00+00:00",
                         task_time_sec=2, energy_kwh=0.2, cost_usd=1, rss_mb=100),
            ]),
            TaskRun("train", "Treinar", "train", TaskStatus.SUCCEEDED, [
                _attempt("train-1", "2026-01-01T10:02:00+00:00", "2026-01-01T10:06:00+00:00",
                         task_time_sec=3, energy_kwh=0.3, emissions_kg_co2=0.05, peak_ram_mb=200),
            ]),
            TaskRun("export", "Exportar", "export", TaskStatus.SUCCEEDED, [
                _attempt("export-1", "2026-01-01T10:02:00+00:00", "2026-01-01T10:10:00+00:00",
                         task_time_sec=7, energy_kwh=0.4, cost_usd=2, peak_vram_mb=300),
            ]),
        ],
    )

    summary = aggregate_workflow_run(workflow, definition)

    assert summary["resources"] == {
        "task_time_sec": 12.0,
        "energy_kwh": 0.9,
        "emissions_kg_co2": 0.05,
        "cost_usd": 3.0,
        "peak_ram_mb": 200.0,
        "peak_vram_mb": 300.0,
    }
    assert summary["makespan_sec"] == 600.0
    assert summary["critical_path"] == {"task_ids": ["prepare", "export"], "duration_sec": 9.0}


def test_aggregate_workflow_includes_failed_attempts_and_accepts_legacy_resource_strings():
    workflow = ExperimentRun(
        "run-1", "workflow", "failed",
        [
            TaskRun("train", "Treinar", "train", TaskStatus.FAILED, [
                _attempt("train-1", None, None, task_time_sec="2.5", energy_kwh="0.1"),
            ]),
        ],
    )
    workflow.tasks[0].attempts[0].status = TaskStatus.FAILED

    summary = aggregate_workflow_run(workflow)

    assert summary["attempts"] == {"failed": 1}
    assert summary["resources"]["task_time_sec"] == 2.5
    assert summary["resources"]["energy_kwh"] == 0.1
    assert summary["makespan_sec"] is None
    assert summary["critical_path"] is None


def test_evaluation_aggregation_applies_declarative_policies_and_ignores_failed_attempts():
    workflow = ExperimentRun(
        "run-1", "workflow", "success",
        [
            TaskRun("train", "Treinar", "train", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt("train-1", 1, TaskStatus.SUCCEEDED, metrics={"evaluation": {
                    "f1_score": 0.7, "accuracy": 0.8, "loss": 0.5, "calibration": 0.3, "examples": 100,
                }}),
                TaskExecutionAttempt("train-2", 2, TaskStatus.FAILED, metrics={"evaluation": {
                    "f1_score": 0.1, "accuracy": 0.1, "loss": 9.0, "calibration": 0.1, "examples": 10,
                }}),
            ]),
            TaskRun("evaluate", "Avaliar", "evaluate", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt("evaluate-1", 1, TaskStatus.SUCCEEDED, metrics={"evaluation": {
                    "f1_score": 0.9, "accuracy": 0.6, "loss": 0.2, "calibration": 0.7, "examples": 300,
                }}),
            ]),
        ],
    )

    summary = aggregate_evaluation_metrics(
        workflow,
        {
            "f1_score": MetricAggregationPolicy(MetricAggregation.LAST),
            "accuracy": MetricAggregationPolicy(MetricAggregation.AVERAGE),
            "loss": MetricAggregationPolicy(MetricAggregation.MIN),
            "calibration": MetricAggregationPolicy(
                MetricAggregation.MAX, task_ids=("train",)
            ),
        },
    )

    assert summary["f1_score"] == 0.9
    assert summary["accuracy"] == pytest.approx(0.7)
    assert summary["loss"] == 0.2
    assert summary["calibration"] == 0.3


def test_weighted_evaluation_policy_uses_requested_metric_name_and_requires_weight():
    workflow = ExperimentRun(
        "run-1", "workflow", "success",
        [TaskRun("evaluate", "Avaliar", "evaluate", TaskStatus.SUCCEEDED, [
            TaskExecutionAttempt("evaluate-1", 1, TaskStatus.SUCCEEDED, metrics={"evaluation": {
                "f1_score": 0.5, "examples": 100,
            }}),
            TaskExecutionAttempt("evaluate-2", 2, TaskStatus.CACHED, metrics={"evaluation": {
                "f1_score": 0.9, "examples": 300,
            }}),
        ])],
    )

    summary = aggregate_evaluation_metrics(
        workflow,
        {"f1_score": MetricAggregationPolicy(MetricAggregation.WEIGHTED_AVERAGE, weight_metric="examples")},
    )

    assert summary["f1_score"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="weight_metric"):
        MetricAggregationPolicy(MetricAggregation.WEIGHTED_AVERAGE)