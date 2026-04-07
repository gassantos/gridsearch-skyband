"""
Main Entry Point - BERT-PLI Experiment Runner
==============================================

Ponto de entrada fino que delega toda a lógica ao pacote ``cli/``.

Uso simples (com defaults):
    python -m main

Uso avançado:
    python -m main --mode grid --config config/experiments/BertPLI.config
    python -m main --mode single --config config/experiments/BertPLI2.config
    python -m main --mode grid --parallel 2

Autor: Gustavo Alexandre
Data: 2026-02-17
"""

import logging
import multiprocessing
import sys

# Deve-se usar 'spawn' para compatibilidade com CUDA
#  - https://pytorch.org/docs/stable/multiprocessing.html#best-practices
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from gridsearch.core import _LOGFILE
from utils.paths import PathManager
from utils.log_setup import setup_main_logging

from cli.parser import build_argument_parser
from cli.commands import _resolve_command
from cli.runners import _parse_sla_constraints

# Configura logging multiprocessing-safe antes de qualquer log.
# QueueListener é iniciado aqui e parado no finally do main().
_log_listener = setup_main_logging(_LOGFILE)
logger = logging.getLogger(__name__)


def main():
    """Função principal que orquestra a execução."""
    parser = build_argument_parser()
    args = parser.parse_args()

    # Print informações iniciais
    logger.info("=" * 70)
    logger.info("BERT-PLI Experiment Runner")
    logger.info("=" * 70)
    logger.info(f"Diretório base: {PathManager.BASE_DIR}")
    logger.info("")

    # Processa --sla-constraint → dict antes de entrar no try
    try:
        sla_dict = _parse_sla_constraints(args.sla_constraints)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        command = _resolve_command(args)
        command.execute(args, sla_dict)

    except KeyboardInterrupt:
        logger.warning("\nExecução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
        sys.exit(1)
    finally:
        _log_listener.stop()


if __name__ == "__main__":
    main()
