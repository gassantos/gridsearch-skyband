"""Testes das definicoes basicas de workflow por dominio."""

import pytest

from experiment.workflow import TaskActivity
from experiment.workflow_planner import WorkflowPlanner
from experiment.workflow_templates import (
    DOMAIN_WORKFLOW_PROFILES,
    build_domain_workflow,
)


@pytest.mark.parametrize(
    ("experiment_type", "task_ids"),
    [
        ("ml_classic", ["ingest_data", "prepare_features", "train_model", "evaluate_model"]),
        ("deep_learning", ["ingest_data", "prepare_data", "train_model", "validate_model", "evaluate_model"]),
        ("nlp", ["ingest_data", "preprocess_text", "train_model", "evaluate_model"]),
        ("llm", ["ingest_data", "prepare_corpus", "adapt_model", "evaluate_model", "publish_model"]),
    ],
)
def test_domain_workflows_define_canonical_task_lifecycle(experiment_type, task_ids):
    workflow = build_domain_workflow("experiment", experiment_type)

    assert workflow.experiment_type == experiment_type
    assert [task.task_id for task in WorkflowPlanner().plan(workflow)] == task_ids


@pytest.mark.parametrize(
    ("experiment_type", "framework"),
    [
        ("ml_classic", "scikit-learn"),
        ("deep_learning", "pytorch"),
        ("nlp", "spacy"),
        ("llm", "huggingface"),
    ],
)
def test_domain_profiles_declare_target_framework_and_workflow_semantics(experiment_type, framework):
    profile = DOMAIN_WORKFLOW_PROFILES[experiment_type]
    workflow = build_domain_workflow("experiment", experiment_type)
    tasks = {task.task_id: task for task in workflow.tasks}

    assert profile.framework == framework
    assert all(task.activity is not TaskActivity.CUSTOM for task in workflow.tasks)
    if experiment_type in {"deep_learning", "llm"}:
        train_task = tasks["train_model"] if experiment_type == "deep_learning" else tasks["adapt_model"]
        assert train_task.resources.gpu_count == 1
        assert train_task.resources.coupling_degree == 0.9


def test_domain_workflow_keeps_model_and_dataset_customization_in_task_profiles():
    workflow = build_domain_workflow(
        "hf-nlp", "nlp",
        task_configs={
            "ingest_data": {"dataset": "nyu-mll/glue", "subset": "mrpc"},
            "train_model": {"model": "bert-base-uncased", "epochs": 3},
        },
        task_input_signatures={"ingest_data": {"dataset": "glue-mrpc-v1"}},
    )
    tasks = {task.task_id: task for task in workflow.tasks}

    assert tasks["ingest_data"].config["dataset"] == "nyu-mll/glue"
    assert tasks["ingest_data"].input_signatures == {"dataset": "glue-mrpc-v1"}
    assert tasks["train_model"].config["model"] == "bert-base-uncased"
    assert tasks["evaluate_model"].depends_on == ("train_model",)


def test_domain_workflow_rejects_unsupported_type():
    with pytest.raises(ValueError, match="experiment_type"):
        build_domain_workflow("invalid", "computer_vision")