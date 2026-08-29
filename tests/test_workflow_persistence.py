"""Testes de persistência dos manifestos P1 de workflow."""

import json

from experiment import persistence
from experiment.workflow import ExperimentRun, TaskRun, TaskStatus


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