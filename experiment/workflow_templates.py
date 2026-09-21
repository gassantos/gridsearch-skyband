"""Templates declarativos de etapas basicas por dominio de experimento."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .generic_workflow import SUPPORTED_EXPERIMENT_TYPES
from .workflow import (
    ArtifactDefinition,
    ArtifactKind,
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


@dataclass(frozen=True)
class HuggingFaceWorkflowConfig:
    """Configuração do template T0 -> T2 -> T5 para Hugging Face."""

    name: str
    dataset_source: str = "local_json"
    dataset_id: str = "train_task2"
    dataset_config: str | None = None
    dataset_version: str = "input"
    dataset_splits: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "valid": "validation", "test": "test"}
    )
    normalization_schema: tuple[str, ...] = ("guid", "text_a", "text_b", "label")
    ingestion_parameters: dict[str, Any] = field(default_factory=dict)
    adaptation_parameters: dict[str, Any] = field(default_factory=dict)
    model_version: str = "pending"
    metrics_version: str = "pending"
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    def __post_init__(self) -> None:
        if self.dataset_source not in {"hub", "local_json"}:
            raise ValueError("dataset_source deve ser 'hub' ou 'local_json'.")
        if not self.dataset_id:
            raise ValueError("dataset_id deve ser informado.")
        if not self.name:
            raise ValueError("name deve ser informado.")


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


def build_huggingface_workflow(config: HuggingFaceWorkflowConfig) -> ExperimentDefinition:
    """Cria o workflow de construção T0 -> T2 -> T5 para Hugging Face.

    As dependências são inferidas pelos artefatos versionados. Os adaptadores
    executáveis das tarefas pertencem à próxima etapa de integração do runtime.
    """
    dataset_uri = (
        f"hf://datasets/{config.dataset_id}"
        if config.dataset_source == "hub"
        else f"data/{config.dataset_id}.json"
    )
    dataset_metadata: dict[str, Any] = {
        "source": config.dataset_source,
        "dataset_id": config.dataset_id,
        "splits": dict(config.dataset_splits),
        "normalization_schema": list(config.normalization_schema),
    }
    if config.dataset_config is not None:
        dataset_metadata["dataset_config"] = config.dataset_config

    dataset = ArtifactDefinition(
        artifact_id=f"dataset-{config.dataset_id}",
        kind=ArtifactKind.DATA,
        version=config.dataset_version,
        uri=dataset_uri,
        metadata=dataset_metadata,
    )
    model = ArtifactDefinition(
        artifact_id=f"model-{config.name}",
        kind=ArtifactKind.MODEL,
        version=config.model_version,
    )
    metrics = ArtifactDefinition(
        artifact_id=f"metrics-{config.name}",
        kind=ArtifactKind.DATA,
        version=config.metrics_version,
    )

    return ExperimentDefinition(
        name=config.name,
        experiment_type="llm",
        tasks=(
            TaskDefinition(
                task_id="ingest_dataset",
                name="Carregar dataset",
                task_type="ingest",
                config={
                    "hf_dataset_source": config.dataset_source,
                    "hf_dataset_id": config.dataset_id,
                    "hf_dataset_config": config.dataset_config,
                    "normalization_schema": list(config.normalization_schema),
                    **config.ingestion_parameters,
                },
                outputs=(dataset,),
                activity=TaskActivity.INGESTION,
                regime=ExecutionRegime.BUILD,
            ),
            TaskDefinition(
                task_id="adapt_model",
                name="Adaptar modelo",
                config=dict(config.adaptation_parameters),
                inputs=(dataset,),
                outputs=(model,),
                activity=TaskActivity.ADAPTATION,
                regime=ExecutionRegime.BUILD,
                resources=config.resources,
            ),
            TaskDefinition(
                task_id="evaluate_model",
                name="Avaliar modelo",
                task_type="evaluate",
                inputs=(model,),
                outputs=(metrics,),
                activity=TaskActivity.EVALUATION_MONITORING,
                regime=ExecutionRegime.BUILD,
                resources=config.resources,
            ),
        ),
    )