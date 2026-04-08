"""
Command Pattern — Despacho de modos de execução
=================================================

Classes de comando para o Gridsearch Experiment Runner (Command Pattern).

Autor: Gustavo Alexandre
"""

import argparse
from abc import ABC, abstractmethod

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


def _resolve_command(args: argparse.Namespace) -> Command:
    """Mapeia os argumentos do CLI para o Command concreto adequado."""
    if args.skyband_only:
        return SkybandOnlyCommand()
    if args.mode == "single":
        return SingleCommand()
    if args.mode == "grid":
        return GridCommand()
    raise ValueError(f"Modo inválido: {args.mode}")
