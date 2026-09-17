"""Validação e planejamento determinístico de workflows em DAG."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from .workflow import ArtifactDefinition, ExperimentDefinition, TaskDefinition


class WorkflowPlanner:
    """Valida dependências e produz uma ordem topológica estável de tarefas."""

    def plan(self, definition: ExperimentDefinition) -> tuple[TaskDefinition, ...]:
        """Retorna tarefas em ordem válida, preservando a ordem declarada em empates.

        Dependências entre produtor e consumidor são inferidas de artefatos
        versionados. Dependências declaradas em ``depends_on`` permanecem como
        restrições adicionais.

        Raises:
            ValueError: Quando uma dependência não existe ou há ciclo no workflow.
        """
        tasks_by_id = {task.task_id: task for task in definition.tasks}
        declaration_order = {
            task.task_id: index for index, task in enumerate(definition.tasks)
        }
        self._validate_dependencies(definition, tasks_by_id)
        producers = self._artifact_producers(definition)
        effective_tasks = tuple(
            replace(task, depends_on=self._effective_dependencies(task, producers))
            for task in definition.tasks
        )
        tasks_by_id = {task.task_id: task for task in effective_tasks}

        dependents: dict[str, list[str]] = {task.task_id: [] for task in effective_tasks}
        in_degree = {task.task_id: len(task.depends_on) for task in effective_tasks}
        for task in effective_tasks:
            for dependency in task.depends_on:
                dependents[dependency].append(task.task_id)

        ready = deque(
            task.task_id for task in effective_tasks if in_degree[task.task_id] == 0
        )
        plan: list[TaskDefinition] = []
        while ready:
            task_id = ready.popleft()
            plan.append(tasks_by_id[task_id])
            for dependent_id in sorted(dependents[task_id], key=declaration_order.__getitem__):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(plan) != len(effective_tasks):
            cyclic_tasks = [
                task.task_id for task in effective_tasks if in_degree[task.task_id] > 0
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

    @staticmethod
    def _artifact_producers(
        definition: ExperimentDefinition,
    ) -> dict[tuple[str, str], tuple[str, ArtifactDefinition]]:
        producers: dict[tuple[str, str], tuple[str, ArtifactDefinition]] = {}
        for task in definition.tasks:
            for artifact in task.outputs:
                key = (artifact.artifact_id, artifact.version)
                if key in producers:
                    producer_id = producers[key][0]
                    raise ValueError(
                        f"Artefato '{artifact.artifact_id}' versão '{artifact.version}' "
                        f"possui mais de um produtor: {producer_id}, {task.task_id}"
                    )
                producers[key] = (task.task_id, artifact)
        return producers

    @staticmethod
    def _effective_dependencies(
        task: TaskDefinition,
        producers: dict[tuple[str, str], tuple[str, ArtifactDefinition]],
    ) -> tuple[str, ...]:
        dependencies = list(task.depends_on)
        produced_by_id = {
            artifact.artifact_id: (producer_id, artifact)
            for producer_id, artifact in producers.values()
        }
        for artifact in task.inputs:
            producer = producers.get((artifact.artifact_id, artifact.version))
            if producer is None:
                incompatible = produced_by_id.get(artifact.artifact_id)
                if incompatible is not None:
                    producer_id, produced = incompatible
                    raise ValueError(
                        f"Tarefa '{task.task_id}' consome artefato incompatível "
                        f"'{artifact.artifact_id}': esperado {artifact.kind.value} "
                        f"versão '{artifact.version}', produzido por '{producer_id}' como "
                        f"{produced.kind.value} versão '{produced.version}'."
                    )
                continue
            producer_id, produced = producer
            if produced.kind is not artifact.kind:
                raise ValueError(
                    f"Tarefa '{task.task_id}' consome artefato incompatível "
                    f"'{artifact.artifact_id}': esperado {artifact.kind.value}, "
                    f"produzido como {produced.kind.value}."
                )
            if producer_id not in dependencies:
                dependencies.append(producer_id)
        return tuple(dependencies)