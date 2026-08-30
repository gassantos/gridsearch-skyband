"""Testes do workflow generico para ML, DL, NLP e LLM."""

import json

import pytest

from experiment.generic_workflow import (
    GenericTaskSpec,
    GenericWorkflowSpec,
    build_generic_task_functions,
    build_generic_workflow,
    load_generic_workflow_spec,
)
from experiment.task_executor import SequentialWorkflowExecutor


@pytest.mark.parametrize("experiment_type", ["ml_classic", "deep_learning", "nlp", "llm"])
def test_generic_workflow_supports_all_target_domains(experiment_type):
    spec = GenericWorkflowSpec(
        "pipeline", experiment_type,
        (
            GenericTaskSpec("prepare", "Preparar", ("prepare",), task_type="prepare"),
            GenericTaskSpec("train", "Treinar", ("train",), depends_on=("prepare",)),
            GenericTaskSpec("evaluate", "Avaliar", ("evaluate",), task_type="evaluate", depends_on=("train",)),
        ),
    )
    commands: list[list[str]] = []

    result = SequentialWorkflowExecutor(
        build_generic_task_functions(spec, command_runner=commands.append)
    ).execute(build_generic_workflow(spec))

    assert result.status == "success"
    assert result.definition_name == "pipeline"
    assert commands == [["prepare"], ["train"], ["evaluate"]]


def test_generic_workflow_loads_external_metrics_and_preserves_profiles(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"evaluation": {"f1_score": 0.9}}), encoding="utf-8")
    spec_file = tmp_path / "workflow.json"
    spec_file.write_text(json.dumps({
        "name": "hf-text-classification",
        "experiment_type": "nlp",
        "monitoring": {"enable_emissions": True, "environment_cost_per_hour_usd": 2.0},
        "tasks": [{
            "task_id": "fine_tune", "name": "Fine-tune", "command": ["hf-train"],
            "config": {"model": "bert-base-uncased"},
            "input_signatures": {"dataset": "glue-mrpc-v1"},
            "metrics_file": str(metrics_file), "artifacts": {"model": "model/"},
        }],
    }), encoding="utf-8")

    spec = load_generic_workflow_spec(spec_file)
    result = SequentialWorkflowExecutor(
        build_generic_task_functions(spec, command_runner=lambda _command: None)
    ).execute(build_generic_workflow(spec))

    task = result.tasks[0]
    assert task.config == {"model": "bert-base-uncased"}
    assert task.input_signatures == {"dataset": "glue-mrpc-v1"}
    assert task.attempts[0].metrics["evaluation"] == {"f1_score": 0.9}
    assert task.attempts[0].artifacts == {"model": "model/"}


def test_generic_workflow_rejects_unsupported_domain():
    with pytest.raises(ValueError, match="experiment_type"):
        GenericWorkflowSpec("pipeline", "unsupported", (GenericTaskSpec("train", "Treinar", ("train",)),))