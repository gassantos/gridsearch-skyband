"""Templates declarativos de etapas basicas por dominio de experimento."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .generic_workflow import SUPPORTED_EXPERIMENT_TYPES
from .workflow import (
    ExecutionRegime,
    ExperimentDefinition,
    ResourceRequirements,
    TaskActivity,
    TaskDefinition,
)


@dataclass(frozen=True)
class WorkflowTaskTemplate:
    """Etapa padrao de um tipo de workflow."""

    task_id: str
    name: str
    task_type: str
    depends_on: tuple[str, ...] = ()
    activity: TaskActivity = TaskActivity.CUSTOM
    regime: ExecutionRegime = ExecutionRegime.BUILD
    resources: ResourceRequirements = ResourceRequirements()


@dataclass(frozen=True)
class DomainWorkflowProfile:
    """Perfil declarativo de um domínio, independente de seu runner futuro."""

    experiment_type: str
    framework: str
    tasks: tuple[WorkflowTaskTemplate, ...]


DOMAIN_WORKFLOW_PROFILES: dict[str, DomainWorkflowProfile] = {
    "ml_classic": DomainWorkflowProfile("ml_classic", "scikit-learn", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_features", "Preparar atributos", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_features",), TaskActivity.ADAPTATION),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "deep_learning": DomainWorkflowProfile("deep_learning", "pytorch", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_data", "Preparar dados", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_data",), TaskActivity.ADAPTATION, resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9)),
        WorkflowTaskTemplate("validate_model", "Validar modelo", "validate", ("train_model",), TaskActivity.EVALUATION_MONITORING, resources=ResourceRequirements(gpu_count=1)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("validate_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "nlp": DomainWorkflowProfile("nlp", "spacy", (
        WorkflowTaskTemplate("ingest_data", "Carregar textos", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("preprocess_text", "Preprocessar textos", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("preprocess_text",), TaskActivity.ADAPTATION),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "llm": DomainWorkflowProfile("llm", "huggingface", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_corpus", "Preparar corpus", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("adapt_model", "Adaptar modelo", "train", ("prepare_corpus",), TaskActivity.ADAPTATION, resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("adapt_model",), TaskActivity.EVALUATION_MONITORING, resources=ResourceRequirements(gpu_count=1)),
        WorkflowTaskTemplate("publish_model", "Publicar modelo", "publish", ("evaluate_model",), TaskActivity.EVALUATION_MONITORING),
    )),
}

DOMAIN_WORKFLOW_TEMPLATES = {
    experiment_type: profile.tasks
    for experiment_type, profile in DOMAIN_WORKFLOW_PROFILES.items()
}


def build_domain_workflow(
    name: str,
    experiment_type: str,
    *,
    task_configs: Mapping[str, Mapping[str, Any]] | None = None,
    task_input_signatures: Mapping[str, Mapping[str, str]] | None = None,
) -> ExperimentDefinition:
    """Cria a DAG base do dominio com perfis opcionais por tarefa.

    Esta funcao define o ciclo de vida comum do experimento, independente de
    biblioteca, modelo ou fonte dos dados. As implementacoes concretas podem
    associar callables, comandos ou pipelines a essas tarefas posteriormente.
    """
    if experiment_type not in SUPPORTED_EXPERIMENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_EXPERIMENT_TYPES))
        raise ValueError(f"experiment_type invalido: {experiment_type}. Use: {supported}.")
    task_configs = task_configs or {}
    task_input_signatures = task_input_signatures or {}
    template = DOMAIN_WORKFLOW_TEMPLATES[experiment_type]
    return ExperimentDefinition(
        name=name,
        experiment_type=experiment_type,
        tasks=tuple(
            TaskDefinition(
                task_id=task.task_id,
                name=task.name,
                task_type=task.task_type,
                depends_on=task.depends_on,
                config=dict(task_configs.get(task.task_id, {})),
                input_signatures=dict(task_input_signatures.get(task.task_id, {})),
                activity=task.activity,
                regime=task.regime,
                resources=task.resources,
            )
            for task in template
        ),
    )