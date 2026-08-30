"""Testes de estimativas historicas por tarefa de workflow."""

from experiment.estimation import estimate_workflow_resources
from experiment.workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)


def _workflow(run_id, train_time, train_status=TaskStatus.SUCCEEDED, train_config=None):
    return ExperimentRun(
        run_id, "historico", "success",
        [
            TaskRun("train", "Treinar", "train", train_status, [
                TaskExecutionAttempt("train-1", 1, train_status, metrics={"resources": {
                    "task_time_sec": train_time,
                    "energy_kwh": train_time / 100,
                    "cost_usd": train_time / 10,
                    "rss_mb": 100 + train_time,
                }}),
            ], config=train_config or {}),
            TaskRun("validate", "Validar", "evaluate", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt("validate-1", 1, TaskStatus.SUCCEEDED, metrics={"resources": {
                    "task_time_sec": 4, "peak_vram_mb": 300,
                }}),
            ]),
        ],
    )


def test_estimate_uses_median_exact_history_and_aggregates_resources():
    definition = ExperimentDefinition(
        "target",
        (TaskDefinition("train", "Treinar", "train"), TaskDefinition("validate", "Validar", "evaluate")),
    )

    estimate = estimate_workflow_resources(definition, [_workflow("run-1", 10), _workflow("run-2", 30)])

    train, validate = estimate["tasks"]
    assert train["match_level"] == "task_id"
    assert train["sample_count"] == 2
    assert train["resources"]["task_time_sec"] == 20.0
    assert train["resources"]["rss_mb"] == 120.0
    assert validate["resources"]["task_time_sec"] == 4.0
    assert estimate["resources"] == {
        "task_time_sec": 24.0,
        "energy_kwh": 0.2,
        "emissions_kg_co2": None,
        "cost_usd": 2.0,
        "peak_ram_mb": 120.0,
        "peak_vram_mb": 300.0,
    }


def test_estimate_falls_back_to_task_type_and_ignores_failed_attempts():
    definition = ExperimentDefinition("target", (TaskDefinition("adapt", "Adaptar", "train"),))

    estimate = estimate_workflow_resources(
        definition,
        [_workflow("successful", 10), _workflow("failed", 100, TaskStatus.FAILED)],
    )

    task = estimate["tasks"][0]
    assert task["match_level"] == "task_type"
    assert task["sample_count"] == 1
    assert task["resources"]["task_time_sec"] == 10.0


def test_estimate_uses_only_history_with_matching_task_profile():
    definition = ExperimentDefinition(
        "target", (TaskDefinition("train", "Treinar", "train", config={"batch_size": 32}),)
    )

    estimate = estimate_workflow_resources(
        definition,
        [
            _workflow("different-profile", 10, train_config={"batch_size": 16}),
            _workflow("matching-profile", 30, train_config={"batch_size": 32}),
        ],
    )

    task = estimate["tasks"][0]
    assert task["match_level"] == "task_id_and_profile"
    assert task["sample_count"] == 1
    assert task["resources"]["task_time_sec"] == 30.0