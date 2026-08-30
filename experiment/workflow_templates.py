"""Templates declarativos de etapas basicas por dominio de experimento."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .generic_workflow import SUPPORTED_EXPERIMENT_TYPES
from .workflow import ExperimentDefinition, TaskDefinition


@dataclass(frozen=True)
class WorkflowTaskTemplate:
    """Etapa padrao de um tipo de workflow."""

    task_id: str
    name: str
    task_type: str
    depends_on: tuple[str, ...] = ()


DOMAIN_WORKFLOW_TEMPLATES: dict[str, tuple[WorkflowTaskTemplate, ...]] = {
    "ml_classic": (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest"),
        WorkflowTaskTemplate("prepare_features", "Preparar atributos", "prepare", ("ingest_data",)),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_features",)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",)),
    ),
    "deep_learning": (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest"),
        WorkflowTaskTemplate("prepare_data", "Preparar dados", "prepare", ("ingest_data",)),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_data",)),
        WorkflowTaskTemplate("validate_model", "Validar modelo", "validate", ("train_model",)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("validate_model",)),
    ),
    "nlp": (
        WorkflowTaskTemplate("ingest_data", "Carregar textos", "ingest"),
        WorkflowTaskTemplate("preprocess_text", "Preprocessar textos", "prepare", ("ingest_data",)),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("preprocess_text",)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",)),
    ),
    "llm": (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest"),
        WorkflowTaskTemplate("prepare_corpus", "Preparar corpus", "prepare", ("ingest_data",)),
        WorkflowTaskTemplate("adapt_model", "Adaptar modelo", "train", ("prepare_corpus",)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("adapt_model",)),
        WorkflowTaskTemplate("publish_model", "Publicar modelo", "publish", ("evaluate_model",)),
    ),
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
            )
            for task in template
        ),
    )