"""Workflow multi-dominio orientado por especificacao JSON."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .workflow import ExperimentDefinition, RetryPolicy, TaskDefinition

SUPPORTED_EXPERIMENT_TYPES = frozenset({"ml_classic", "deep_learning", "nlp", "llm"})
CommandRunner = Callable[[list[str]], None]


@dataclass(frozen=True)
class GenericTaskSpec:
    """Descricao executavel de uma etapa de um pipeline multi-dominio."""

    task_id: str
    name: str
    command: tuple[str, ...]
    task_type: str = "train"
    depends_on: tuple[str, ...] = ()
    required: bool = True
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    config: dict[str, Any] = field(default_factory=dict)
    input_signatures: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics_file: str | None = None


@dataclass(frozen=True)
class GenericWorkflowSpec:
    """Especificacao de workflow aplicavel a ML, DL, NLP e LLM."""

    name: str
    experiment_type: str
    tasks: tuple[GenericTaskSpec, ...]
    enable_emissions: bool = False
    environment_cost_per_hour_usd: float | None = None

    def __post_init__(self) -> None:
        if self.experiment_type not in SUPPORTED_EXPERIMENT_TYPES:
            supported = ", ".join(sorted(SUPPORTED_EXPERIMENT_TYPES))
            raise ValueError(f"experiment_type invalido: {self.experiment_type}. Use: {supported}.")
        if not self.tasks:
            raise ValueError("O workflow generico deve conter ao menos uma tarefa.")
        if any(not task.command for task in self.tasks):
            raise ValueError("Cada tarefa do workflow generico deve definir command.")


def load_generic_workflow_spec(path: Path) -> GenericWorkflowSpec:
    """Carrega e valida uma especificacao JSON de workflow generico."""
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Nao foi possivel ler workflow_spec: {path}") from exc
    if not isinstance(source, dict):
        raise ValueError("workflow_spec deve conter um objeto JSON.")  # noqa: TRY004

    try:
        tasks = tuple(_task_spec(item) for item in source["tasks"])
        monitoring = source.get("monitoring", {})
        return GenericWorkflowSpec(
            name=source["name"],
            experiment_type=source["experiment_type"],
            tasks=tasks,
            enable_emissions=bool(monitoring.get("enable_emissions", False)),
            environment_cost_per_hour_usd=monitoring.get("environment_cost_per_hour_usd"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("workflow_spec possui campos invalidos.") from exc


def build_generic_workflow(spec: GenericWorkflowSpec) -> ExperimentDefinition:
    """Converte a especificacao generica em definicao declarativa de DAG."""
    return ExperimentDefinition(
        name=spec.name,
        experiment_type=spec.experiment_type,
        tasks=tuple(
            TaskDefinition(
                task_id=task.task_id,
                name=task.name,
                task_type=task.task_type,
                depends_on=task.depends_on,
                required=task.required,
                retry_policy=task.retry_policy,
                config=task.config,
                input_signatures=task.input_signatures,
            )
            for task in spec.tasks
        ),
    )


def build_generic_task_functions(
    spec: GenericWorkflowSpec,
    *,
    command_runner: CommandRunner | None = None,
) -> Mapping[str, Callable[[], dict[str, Any]]]:
    """Cria adaptadores que executam comandos e leem metricas estruturadas."""
    run = command_runner or _run_command
    return {task.task_id: _task_function(task, run) for task in spec.tasks}


def _task_spec(source: Any) -> GenericTaskSpec:
    if not isinstance(source, dict):
        raise ValueError("Cada tarefa deve ser um objeto JSON.")  # noqa: TRY004
    retry = source.get("retry_policy", {})
    command = source.get("command", [])
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("command deve ser uma lista de strings.")
    return GenericTaskSpec(
        task_id=source["task_id"],
        name=source["name"],
        command=tuple(command),
        task_type=source.get("task_type", "train"),
        depends_on=tuple(source.get("depends_on", [])),
        required=source.get("required", True),
        retry_policy=RetryPolicy(
            max_attempts=retry.get("max_attempts", 1),
            retryable_error_types=tuple(retry.get("retryable_error_types", [])),
        ),
        config=dict(source.get("config", {})),
        input_signatures=dict(source.get("input_signatures", {})),
        artifacts=dict(source.get("artifacts", {})),
        metrics_file=source.get("metrics_file"),
    )


def _task_function(task: GenericTaskSpec, run: CommandRunner) -> Callable[[], dict[str, Any]]:
    def execute() -> dict[str, Any]:
        run(list(task.command))
        return {"metrics": _load_metrics(task.metrics_file), "artifacts": task.artifacts}
    return execute


def _load_metrics(metrics_file: str | None) -> dict[str, Any]:
    if metrics_file is None:
        return {}
    try:
        data = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Nao foi possivel ler metrics_file: {metrics_file}") from exc
    if not isinstance(data, dict):
        raise ValueError("metrics_file deve conter um objeto JSON.")  # noqa: TRY004
    return data


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)