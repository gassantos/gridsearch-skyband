"""Validação e planejamento determinístico de workflows em DAG."""

from __future__ import annotations

from collections import deque

from .workflow import ExperimentDefinition, TaskDefinition


class WorkflowPlanner:
    """Valida dependências e produz uma ordem topológica estável de tarefas."""

    def plan(self, definition: ExperimentDefinition) -> tuple[TaskDefinition, ...]:
        """Retorna tarefas em ordem válida, preservando a ordem declarada em empates.

        Raises:
            ValueError: Quando uma dependência não existe ou há ciclo no workflow.
        """
        tasks_by_id = {task.task_id: task for task in definition.tasks}
        declaration_order = {
            task.task_id: index for index, task in enumerate(definition.tasks)
        }
        self._validate_dependencies(definition, tasks_by_id)

        dependents: dict[str, list[str]] = {task.task_id: [] for task in definition.tasks}
        in_degree = {task.task_id: len(task.depends_on) for task in definition.tasks}
        for task in definition.tasks:
            for dependency in task.depends_on:
                dependents[dependency].append(task.task_id)

        ready = deque(
            task.task_id for task in definition.tasks if in_degree[task.task_id] == 0
        )
        plan: list[TaskDefinition] = []
        while ready:
            task_id = ready.popleft()
            plan.append(tasks_by_id[task_id])
            for dependent_id in sorted(dependents[task_id], key=declaration_order.__getitem__):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(plan) != len(definition.tasks):
            cyclic_tasks = [
                task.task_id for task in definition.tasks if in_degree[task.task_id] > 0
            ]
            raise ValueError(
                "Workflow contém ciclo envolvendo as tarefas: "
                + ", ".join(cyclic_tasks)
            )
        return tuple(plan)

    @staticmethod
    def _validate_dependencies(
        definition: ExperimentDefinition,
        tasks_by_id: dict[str, TaskDefinition],
    ) -> None:
        for task in definition.tasks:
            missing = [dependency for dependency in task.depends_on if dependency not in tasks_by_id]
            if missing:
                raise ValueError(
                    f"Tarefa '{task.task_id}' depende de tarefa inexistente: "
                    + ", ".join(missing)
                )