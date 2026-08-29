"""
Executor sequencial de tarefas de um workflow.
============================================

 - Executa tarefas de um workflow de forma sequencial.
 - Esta execução é sequencial, respeitando as dependências entre tarefas.
 - Cada tarefa é executada em uma tentativa única, e os resultados de execução são agregados ao experimento.

"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import psutil

from .helpers import now_iso
from .workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)
from .workflow_planner import WorkflowPlanner

TaskCallable = Callable[[], dict[str, Any] | None]


class SequentialWorkflowExecutor:
    """Executa tarefas independentes ou encadeadas por ``depends_on`` em série."""

    def __init__(self, task_functions: Mapping[str, TaskCallable]) -> None:
        self._task_functions = task_functions

    def execute(self, definition: ExperimentDefinition) -> ExperimentRun:
        """Executa as tarefas no plano topológico e retorna o agregado observado."""
        task_runs: list[TaskRun] = []
        statuses: dict[str, TaskStatus] = {}
        task_plan = WorkflowPlanner().plan(definition)

        for task in task_plan:
            blocked = any(
                statuses.get(dependency) is not TaskStatus.SUCCEEDED
                for dependency in task.depends_on
            )
            if blocked:
                task_runs.append(
                    TaskRun(task.task_id, task.name, task.task_type, TaskStatus.SKIPPED)
                )
                statuses[task.task_id] = TaskStatus.SKIPPED
                continue

            task_fn = self._task_functions.get(task.task_id)
            if task_fn is None:
                raise ValueError(f"Nenhuma função registrada para a tarefa '{task.task_id}'.")

            task_run = self._execute_task(task, task_fn)
            task_runs.append(task_run)
            statuses[task.task_id] = task_run.status

        required_failed = any(
            task.required and statuses[task.task_id] is not TaskStatus.SUCCEEDED
            for task in definition.tasks
        )
        return ExperimentRun(
            experiment_run_id=str(uuid.uuid4()),
            definition_name=definition.name,
            status="failed" if required_failed else "success",
            tasks=task_runs,
        )

    @staticmethod
    def _execute_task(
        task,
        task_fn: TaskCallable,
    ) -> TaskRun:
        attempts: list[TaskExecutionAttempt] = []
        for attempt_number in range(1, task.retry_policy.max_attempts + 1):
            attempt = SequentialWorkflowExecutor._execute_attempt(
                task.task_id,
                attempt_number,
                task_fn,
            )
            attempts.append(attempt)
            if attempt.status is TaskStatus.SUCCEEDED:
                break
            if not task.retry_policy.allows_retry(attempt.error_type or ""):
                break

        return TaskRun(
            task_id=task.task_id,
            name=task.name,
            task_type=task.task_type,
            status=attempts[-1].status,
            attempts=attempts,
        )

    @staticmethod
    def _execute_attempt(
        task_id: str,
        attempt_number: int,
        task_fn: TaskCallable,
    ) -> TaskExecutionAttempt:
        attempt = TaskExecutionAttempt(
            attempt_id=str(uuid.uuid4()),
            attempt_number=attempt_number,
        )
        attempt.transition_to(TaskStatus.READY)
        attempt.transition_to(TaskStatus.RUNNING)
        attempt.started_at = now_iso()
        process = psutil.Process()
        start = time.perf_counter()

        try:
            output = task_fn() or {}
            attempt.metrics = {
                "resources": {
                    "task_time_sec": time.perf_counter() - start,
                    "rss_mb": process.memory_info().rss / (1024 ** 2),
                },
                **output.get("metrics", {}),
            }
            attempt.artifacts = output.get("artifacts", {})
            attempt.transition_to(TaskStatus.SUCCEEDED)
        except Exception as exc:  # noqa: BLE001
            attempt.metrics = {
                "resources": {
                    "task_time_sec": time.perf_counter() - start,
                    "rss_mb": process.memory_info().rss / (1024 ** 2),
                }
            }
            attempt.error = str(exc)
            attempt.error_type = exc.__class__.__name__
            attempt.transition_to(TaskStatus.FAILED)
        finally:
            attempt.completed_at = now_iso()

        return attempt