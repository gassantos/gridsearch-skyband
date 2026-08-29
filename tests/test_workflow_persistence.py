"""Testes de persistência dos manifestos P1 de workflow."""

import json

from experiment import persistence
from experiment.workflow import ExperimentRun, TaskExecutionAttempt, TaskRun, TaskStatus


def test_write_workflow_run_writes_manifest_and_task(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = ExperimentRun(
        experiment_run_id="run-1",
        definition_name="workflow",
        status="success",
        tasks=[TaskRun("train", "Treino", "train", TaskStatus.SUCCEEDED)],
    )

    run_dir = persistence.write_workflow_run(workflow)

    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        assert json.load(f)["experiment_run_id"] == "run-1"
    with open(run_dir / "tasks" / "train.json", encoding="utf-8") as f:
        assert json.load(f)["status"] == "succeeded"


def test_load_workflow_run_restores_attempt_history(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = ExperimentRun(
        experiment_run_id="run-1",
        definition_name="workflow",
        status="failed",
        tasks=[
            TaskRun(
                "train",
                "Treino",
                "train",
                TaskStatus.FAILED,
                [
                    TaskExecutionAttempt(
                        "attempt-1", 1, TaskStatus.FAILED, error="timeout", error_type="TimeoutError"
                    )
                ],
            )
        ],
    )

    restored = persistence.load_workflow_run(persistence.write_workflow_run(workflow))

    assert restored.experiment_run_id == "run-1"
    assert restored.tasks[0].attempts[0].error_type == "TimeoutError"