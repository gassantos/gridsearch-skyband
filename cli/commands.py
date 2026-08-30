"""
Command Pattern — Despacho de modos de execução
=================================================

Classes de comando para o Gridsearch Experiment Runner (Command Pattern).

Autor: Gustavo Alexandre
"""

import argparse
from abc import ABC, abstractmethod

from experiment.bertpli_workflow import (
    BertPliWorkflowConfig,
    build_bertpli_task_functions,
    build_bertpli_workflow,
)
from experiment.persistence import write_workflow_run
from experiment.task_executor import SequentialWorkflowExecutor

from .runners import (
    _build_dataset_overrides,
    run_grid_search_experiments,
    run_single_experiment,
    run_skyband_analysis,
)


class Command(ABC):
    """Interface para comandos de execução (Command Pattern)."""

    @abstractmethod
    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        """Executa a ação do comando."""


class SkybandOnlyCommand(Command):
    """Executa apenas a análise Skyband sobre estado existente."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        run_skyband_analysis(
            k=args.skyband_k,
            sla_constraints=sla_dict or None,
            sla_profile_name=args.sla_profile,
            metrics=args.skyband_metrics,
            compare=args.skyband_compare,
            state_file=args.skyband_state,
        )


class SingleCommand(Command):
    """Executa um único experimento, opcionalmente seguido de Skyband."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        run_single_experiment(
            args.config,
            train_dataset=args.train_dataset,
            dataset_overrides=_build_dataset_overrides(args),
            gpu_list=args.gpu,
            tpu_cores=args.tpu_cores,
            precision=args.precision,
        )
        if not args.no_skyband:
            # require_state=False: modo single não gera estado de grid search;
            # a ausência do arquivo é aviso, não erro.
            run_skyband_analysis(
                k=args.skyband_k,
                sla_constraints=sla_dict or None,
                sla_profile_name=args.sla_profile,
                metrics=args.skyband_metrics,
                compare=args.skyband_compare,
                state_file=args.skyband_state,
                require_state=False,
            )


class GridCommand(Command):
    """Executa grid search, opcionalmente seguido de Skyband."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        run_grid_search_experiments(
            base_config_path=args.config,
            grid_config_path=args.grid_config,
            parallel=args.parallel,
            resume=args.resume,
            sla_profile_name=args.sla_profile,
            sla_constraints=sla_dict or None,
            train_dataset=args.train_dataset,
            dataset_overrides=_build_dataset_overrides(args),
            gpu_ids=args.gpu,
            tpu_cores=args.tpu_cores,
            precision=args.precision,
        )
        if not args.no_skyband:
            run_skyband_analysis(
                k=args.skyband_k,
                sla_constraints=sla_dict or None,
                sla_profile_name=args.sla_profile,
                metrics=args.skyband_metrics,
                compare=args.skyband_compare,
                state_file=args.skyband_state,
            )


class BertPliWorkflowCommand(Command):
    """Executa o pipeline BERT-PLI composto por tarefas rastreáveis."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        del sla_dict
        gpu = ",".join(map(str, args.gpu)) if args.gpu else None
        config = BertPliWorkflowConfig(gpu=gpu)
        definition = build_bertpli_workflow(config)
        commands: list[list[str]] = []
        task_functions = build_bertpli_task_functions(
            config,
            command_runner=commands.append if args.workflow_dry_run else None,
        )
        if args.workflow_dry_run:
            task_functions = {
                **task_functions,
                "evaluate_retrieval": lambda: {
                    "metrics": {"dry_run": True},
                    "artifacts": {"metrics": config.metrics_result},
                },
            }
        workflow = SequentialWorkflowExecutor(task_functions).execute(definition)
        run_dir = write_workflow_run(workflow)
        if args.workflow_dry_run:
            print(f"Workflow BERT-PLI validado sem treinamento: {run_dir}")
            for command in commands:
                print(" ".join(command))
        elif workflow.status != "success":
            raise RuntimeError(f"Workflow BERT-PLI falhou. Manifesto: {run_dir}")


def _resolve_command(args: argparse.Namespace) -> Command:
    """Mapeia os argumentos do CLI para o Command concreto adequado."""
    if args.workflow == "bertpli":
        return BertPliWorkflowCommand()
    if args.skyband_only:
        return SkybandOnlyCommand()
    if args.mode == "single":
        return SingleCommand()
    if args.mode == "grid":
        return GridCommand()
    raise ValueError(f"Modo inválido: {args.mode}")
