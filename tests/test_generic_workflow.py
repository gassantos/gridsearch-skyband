"""Testes do workflow generico para ML, DL, NLP e LLM."""

import json
from pathlib import Path

import pytest

from experiment.generic_workflow import (
    GenericTaskSpec,
    GenericWorkflowSpec,
    build_generic_task_functions,
    build_generic_workflow,
    load_generic_workflow_spec,
)
from experiment.task_executor import SequentialWorkflowExecutor
from experiment.workflow import ArtifactKind, ExecutionRegime, TaskActivity


_EXAMPLES_DIR = Path(__file__).parents[1] / "examples" / "workflow"


@pytest.mark.parametrize("experiment_type", ["ml_classic", "deep_learning", "nlp", "llm"])
def test_generic_workflow_supports_all_target_domains(experiment_type):
    spec = GenericWorkflowSpec(
        "pipeline", experiment_type,
        (
            GenericTaskSpec("prepare", "Preparar", ("prepare",), task_type="prepare", activity=TaskActivity.INGESTION),
            GenericTaskSpec("train", "Treinar", ("train",), depends_on=("prepare",), activity=TaskActivity.ADAPTATION),
            GenericTaskSpec("evaluate", "Avaliar", ("evaluate",), task_type="evaluate", depends_on=("train",), activity=TaskActivity.EVALUATION_MONITORING),
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
        "tasks": [
            {"task_id": "ingest", "name": "Ingerir", "command": ["load"], "activity": "ingestion"},
            {
                "task_id": "fine_tune", "name": "Fine-tune", "command": ["hf-train"],
                "activity": "adaptation", "depends_on": ["ingest"],
                "config": {"model": "bert-base-uncased"},
                "input_signatures": {"dataset": "glue-mrpc-v1"},
                "metrics_file": str(metrics_file), "artifacts": {"model": "model/"},
            },
            {"task_id": "evaluate", "name": "Avaliar", "command": ["eval"], "activity": "evaluation_monitoring", "depends_on": ["fine_tune"]},
        ],
    }), encoding="utf-8")

    spec = load_generic_workflow_spec(spec_file)
    result = SequentialWorkflowExecutor(
        build_generic_task_functions(spec, command_runner=lambda _command: None)
    ).execute(build_generic_workflow(spec))

    task = result.tasks[1]
    assert task.config == {"model": "bert-base-uncased"}
    assert task.input_signatures == {"dataset": "glue-mrpc-v1"}
    assert task.attempts[0].metrics["evaluation"] == {"f1_score": 0.9}
    assert task.attempts[0].artifacts == {"model": "model/"}


def test_generic_workflow_rejects_missing_canonical_lifecycle():
    spec = GenericWorkflowSpec(
        "invalid", "nlp", (
            GenericTaskSpec("train", "Treinar", ("train",), activity=TaskActivity.ADAPTATION),
        ),
    )

    with pytest.raises(ValueError, match="ingestion"):
        build_generic_workflow(spec)


@pytest.mark.parametrize("filename", ["ml_classic.json", "deep_learning.json", "nlp.json", "llm.json"])
def test_official_generic_workflow_examples_follow_the_canonical_template(filename):
    spec = load_generic_workflow_spec(_EXAMPLES_DIR / filename)
    commands: list[list[str]] = []

    workflow = SequentialWorkflowExecutor(
        build_generic_task_functions(spec, command_runner=commands.append)
    ).execute(build_generic_workflow(spec))

    assert workflow.status == "success"
    assert len(commands) == len(spec.tasks)


def test_generic_workflow_loads_declarative_task_semantics_from_json(tmp_path):
    spec_file = tmp_path / "workflow.json"
    spec_file.write_text(json.dumps({
        "name": "hf-adaptation",
        "experiment_type": "llm",
        "tasks": [
            {"task_id": "ingest", "name": "Ingerir", "command": ["load"], "activity": "ingestion"},
            {
                "task_id": "adapt", "name": "Adaptar", "command": ["hf-train"],
                "activity": "adaptation", "regime": "build", "depends_on": ["ingest"],
                "inputs": [{"artifact_id": "corpus", "kind": "data", "version": "v2"}],
                "outputs": [{"artifact_id": "model", "kind": "model", "version": "v1", "uri": "models/v1"}],
                "resources": {"cpu_cores": 4, "memory_mb": 8192, "gpu_count": 1, "coupling_degree": 0.9},
                "is_composite": True, "stop_predicate": "validation_loss <= 0.1",
            },
            {"task_id": "evaluate", "name": "Avaliar", "command": ["eval"], "activity": "evaluation_monitoring", "depends_on": ["adapt"]},
        ],
    }), encoding="utf-8")

    workflow = build_generic_workflow(load_generic_workflow_spec(spec_file))
    task = workflow.tasks[1]

    assert task.inputs[0].kind is ArtifactKind.DATA
    assert task.outputs[0].uri == "models/v1"
    assert task.activity is TaskActivity.ADAPTATION
    assert task.regime is ExecutionRegime.BUILD
    assert task.resources.gpu_count == 1
    assert task.is_composite is True
    assert task.stop_predicate == "validation_loss <= 0.1"


def test_generic_workflow_rejects_invalid_artifact_kind(tmp_path):
    spec_file = tmp_path / "workflow.json"
    spec_file.write_text(json.dumps({
        "name": "invalid", "experiment_type": "nlp",
        "tasks": [{
            "task_id": "prepare", "name": "Preparar", "command": ["prepare"],
            "outputs": [{"artifact_id": "corpus", "kind": "unknown", "version": "v1"}],
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="workflow_spec possui campos invalidos"):
        load_generic_workflow_spec(spec_file)


def test_generic_workflow_rejects_unsupported_domain():
    with pytest.raises(ValueError, match="experiment_type"):
        GenericWorkflowSpec("pipeline", "unsupported", (GenericTaskSpec("train", "Treinar", ("train",)),))