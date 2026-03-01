"""
Main Entry Point - BERT-PLI Experiment Runner
==============================================

Script principal para orquestrar a execução de experimentos.
Centraliza a execução de experimentos únicos ou grid search.

Uso simples (com defaults):
    python -m main

Uso avançado:
    python -m main --mode grid --config config/experiments/BertPLI.config
    python -m main --mode single --config config/experiments/BertPLI2.config
    python -m main --mode grid --parallel 2

Autor: Gustavo Alexandre
Data: 2026-02-17
"""

import argparse
import json
import logging
import sys
import multiprocessing
from pathlib import Path
from typing import Optional

# Deve-se usar 'spawn' para compatibilidade com CUDA
#  - https://pytorch.org/docs/stable/multiprocessing.html#best-practices
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from gridsearch.core import run_grid_search, _LOGFILE, GRID_OUTPUT_DIR
from gridsearch.skyband import (
    compare_skyband_vs_ranking, skyband_report,
    DEFAULT_METRICS,
)
from utils.paths import PathManager
from utils.log_setup import setup_main_logging

# Configura logging multiprocessing-safe antes de qualquer log.
# QueueListener é iniciado aqui e parado no finally do main().
_log_listener = setup_main_logging(_LOGFILE)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURAÇÕES PADRÃO
# =========================
DEFAULT_CONFIG = "config/experiments/BertPLI.config"
DEFAULT_GRID_CONFIG = "gridsearch/config/grid_search_test.json"
DEFAULT_MODE = "grid"
DEFAULT_PARALLEL = 2
DEFAULT_SLA_PROFILES = "gridsearch/config/sla_profiles.json"
DEFAULT_SKYBAND_K = 3


def validate_paths(config_path: str, grid_config_path: Optional[str] = None) -> bool:
    """
    Valida se os caminhos de configuração existem.
    
    Args:
        config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo de grid config (opcional)
        
    Returns:
        True se todos os arquivos existem, False caso contrário
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        return False
    
    if grid_config_path:
        grid_file = Path(grid_config_path)
        if not grid_file.exists():
            logger.error(f"Arquivo de grid config não encontrado: {grid_config_path}")
            return False
    
    return True


def run_single_experiment(config_path: str):
    """
    Executa um único experimento.
    
    Args:
        config_path: Caminho do arquivo de configuração
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from run_experiment import execute_experiment
    
    logger.info("=" * 70)
    logger.info("MODO: Experimento Único")
    logger.info(f"Configuração: {config_path}")
    logger.info("=" * 70)
    
    if not validate_paths(config_path):
        sys.exit(1)
    
    execute_experiment(config_path)
    logger.info("Experimento concluído com sucesso!")


def _parse_sla_constraints(constraint_list: Optional[list]) -> dict:
    """
    Converte a lista de strings "metrica=valor" em dicionário de constraints.

    Cada item deve ter o formato ``nome_metrica=valor_numerico``.
    Chamado internamente ao processar ``--sla-constraint``.

    Exemplo::

        _parse_sla_constraints(["cost_usd=5.0", "train_time_sec=7200"])
        # → {"cost_usd": 5.0, "train_time_sec": 7200.0}
    """
    if not constraint_list:
        return {}
    constraints = {}
    for item in constraint_list:
        if "=" not in item:
            raise ValueError(
                f"Formato inválido para --sla-constraint: '{item}'. "
                "Use 'metrica=valor' (ex: cost_usd=5.0)"
            )
        key, val = item.split("=", 1)
        try:
            constraints[key.strip()] = float(val.strip())
        except ValueError:
            raise ValueError(
                f"Valor não numérico em --sla-constraint: '{item}'. "
                f"Esperado float, recebido: '{val}'"
            )
    return constraints


def run_skyband_analysis(
    k: int = 1,
    sla_constraints: Optional[dict] = None,
    sla_profile_name: Optional[str] = None,
    metrics: Optional[list] = None,
    compare: bool = False,
    state_file: Optional[str] = None,
) -> None:
    """
    Carrega resultados do estado do grid search e executa análise Skyband.

    Carrega automaticamente o arquivo de estado mais recente encontrado em
    ``output/experiments/grid_search/``, normaliza os tipos de recursos e
    executa ``skyband_query()`` com os parâmetros fornecidos.

    Args:
        k:                Ordem do Skyband (padrão 1 = frente de Pareto).
        sla_constraints:  Dicionário {metrica: valor_max} já processado.
        sla_profile_name: Nome de perfil em ``sla_profiles.json```.
                          Se informado, sobrescreve ``sla_constraints``, ``k``
                          e ``metrics`` com os valores do perfil.
        metrics:          Lista de métricas para dominância.
                          None = usa DEFAULT_METRICS (5 critérios).
        compare:          Se True, imprime comparação Skyband vs ranking escalar.
        state_file:       Caminho direto ao JSON de estado. None = detecta o
                          arquivo mais recente em GRID_OUTPUT_DIR.
    """
    # ── Localiza o arquivo de estado ────────────────────────────────────────
    if state_file:
        sf = Path(state_file)
    else:
        candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_state_*.json"), reverse=True)
        if not candidates:
            logger.error(
                "Nenhum arquivo de estado encontrado em: %s", GRID_OUTPUT_DIR
            )
            sys.exit(1)
        sf = candidates[0]

    if not sf.exists():
        logger.error("Arquivo de estado não encontrado: %s", sf)
        sys.exit(1)

    logger.info("Carregando estado de: %s", sf)
    with open(sf, encoding="utf-8") as f:
        state = json.load(f)

    # ── Normaliza campos numéricos (JSON pode armazenar como string) ─────────
    def _norm(r: dict) -> dict:
        for key, val in r.get("resources", {}).items():
            if val is not None:
                try:
                    r["resources"][key] = float(val)
                except (TypeError, ValueError):
                    pass
        return r

    results = [_norm(r) for r in state.get("results", [])]
    success = [r for r in results if r.get("status") == "success"]
    logger.info("%d resultados carregados (%d bem-sucedidos)", len(results), len(success))

    if not success:
        logger.error("Nenhum experimento com status=success encontrado.")
        sys.exit(1)

    # ── Carrega perfil de SLA (sobrescreve constraints/k/metrics se fornecido) ─
    if sla_profile_name:
        profiles_path = Path(DEFAULT_SLA_PROFILES)
        if not profiles_path.exists():
            logger.error("Arquivo de perfis SLA não encontrado: %s", profiles_path)
            sys.exit(1)
        with open(profiles_path, encoding="utf-8") as f:
            sla_cfg = json.load(f)
        profiles = sla_cfg.get("profiles", {})
        if sla_profile_name not in profiles:
            available = list(profiles.keys())
            logger.error(
                "Perfil '%s' não encontrado. Disponíveis: %s",
                sla_profile_name, available,
            )
            sys.exit(1)
        profile = profiles[sla_profile_name]
        sla_constraints = {m: v for m, v in profile["constraints"].items() if v is not None}
        k = profile["skyband_k"]
        metrics = profile["metrics"]
        logger.info(
            "Perfil SLA '%s' carregado: k=%d, metrics=%s, constraints=%s",
            sla_profile_name, k, metrics, sla_constraints,
        )

    effective_metrics = metrics or DEFAULT_METRICS

    # ── Relatório principal ──────────────────────────────────────────────────
    print()
    report = skyband_report(
        success,
        k=k,
        sla_constraints=sla_constraints or None,
        metrics=effective_metrics,
    )
    print(report)

    # ── Comparação opcional Skyband vs Ranking Escalar ───────────────────────
    if compare:
        print()
        cmp = compare_skyband_vs_ranking(
            success,
            sla=sla_constraints or None,
            metrics=effective_metrics,
            k=k,
        )
        print("=" * 72)
        print("SKYBAND vs RANKING ESCALAR")
        print("=" * 72)
        print(f"  k                  : {cmp['k']}")
        print(f"  Jaccard similarity : {cmp['jaccard_similarity']:.3f}")
        print(f"  Somente no Skyband : {cmp['only_in_skyband']}")
        print(f"  Somente no Escalar : {cmp['only_in_scalar']}")
        print(f"  Interseção         : {cmp['intersection']}")
        print()
        print("  Skyband (preserva estrutura de dominância):")
        for r in cmp["skyband"]:
            p  = r["grid_params"]
            rs = r["resources"]
            vals = "  ".join(
                f"{m}={rs.get(m, float('inf')):.4g}" for m in effective_metrics[:3]
            )
            print(
                f"    rank={r['skyband_rank']} dom={r['domination_count']}  "
                f"{p.get('optimizer','?'):<10}  {vals}"
            )
        print()
        print("  Ranking Escalar (score ponderado min-max):")
        for i, r in enumerate(cmp["scalar_top"]):
            p  = r["grid_params"]
            rs = r["resources"]
            vals = "  ".join(
                f"{m}={rs.get(m, float('inf')):.4g}" for m in effective_metrics[:3]
            )
            print(f"    [{i + 1}] {p.get('optimizer','?'):<10}  {vals}")
        print("=" * 72)


def run_grid_search_experiments(
    base_config_path: str,
    grid_config_path: str,
    parallel: int = 1,
    resume: bool = False
):
    """
    Executa grid search de hiperparâmetros.
    
    Args:
        base_config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo JSON com grade de hiperparâmetros
        parallel: Número de processos paralelos
        resume: Se True, retoma execução anterior
    """
    logger.info("=" * 70)
    logger.info("MODO: Grid Search")
    logger.info(f"Configuração base: {base_config_path}")
    logger.info(f"Grid config: {grid_config_path}")
    logger.info(f"Execução: {'Paralela (' + str(parallel) + ' workers)' if parallel > 1 else 'Sequencial'}")
    logger.info("=" * 70)
    
    if not validate_paths(base_config_path, grid_config_path):
        sys.exit(1)
    
    # Carrega configuração da grade
    with open(grid_config_path, 'r', encoding='utf-8') as f:
        grid_config = json.load(f)
    
    # Executa grid search
    results = run_grid_search(
        base_config_path=base_config_path,
        grid_config=grid_config,
        resume=resume,
        parallel=parallel
    )
    
    logger.info("Grid search concluído com sucesso!")
    logger.info(f"Total de experimentos executados: {len(results)}")


def main():
    """Função principal que orquestra a execução."""
    parser = argparse.ArgumentParser(
        description="BERT-PLI Experiment Runner - Execução centralizada de experimentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemplos de uso:

  # Execução padrão — grid search + Skyband k={DEFAULT_SKYBAND_K} automático
  python -m main

  # Experimento único (Skyband também roda ao final por padrão)
  python -m main --mode single

  # Grid search com configuração específica
  python -m main --mode grid --grid-config gridsearch/config/grid_search_test.json

  # Grid search paralelo com 4 workers
  python -m main --mode grid --parallel 4

  # Retomar grid search interrompido (Skyband roda ao final)
  python -m main --mode grid --resume

  # Desativar análise Skyband (somente execução dos experimentos)
  python -m main --mode grid --no-skyband

  # Skyband com perfil de SLA sustentável (k={DEFAULT_SKYBAND_K} default)
  python -m main --mode grid --sla-profile sustentavel

  # Skyband com k personalizado
  python -m main --mode grid --skyband-k 5 --sla-profile balanceado

  # Apenas análise Skyband sobre estado existente (sem novo treino)
  python -m main --skyband-only

  # Skyband-only com k=2 e constraints de SLA customizadas
  python -m main --skyband-only --skyband-k 2 \\
      --sla-constraint cost_usd=5.0 \\
      --sla-constraint train_time_sec=7200

  # Skyband-only com perfil predefinido + comparação vs ranking escalar
  python -m main --skyband-only --sla-profile balanceado --skyband-compare

  # Skyband sobre arquivo de estado específico
  python -m main --skyband-only \\
      --skyband-state output/experiments/grid_search/grid_search_state_GPU_2026-03-01.json \\
      --skyband-k 2 --skyband-metrics train_time_sec cost_usd energy_kwh

  # Skyband com métricas customizadas (2 critérios: tempo e custo)
  python -m main --skyband-only --skyband-metrics train_time_sec cost_usd

Perfis de SLA disponíveis (--sla-profile):
  economico    — custo <= $2.00
  sustentavel  — energia <= 0.05 kWh, CO2 <= 0.01 kg
  tempo        — treino <= 3600 s
  balanceado   — custo <= $5.00, tempo <= 7200 s, energia <= 0.1 kWh
  dev          — tempo <= 1800 s, RAM <= 8192 MB
  producao     — custo <= $20.00, tempo <= 1800 s, RAM <= 16384 MB

Métricas para --sla-constraint (filtro de admissibilidade, checagem de execução):
  train_time_sec   energy_kwh   peak_ram_mb   emissions_kg_co2   cost_usd

Métricas para --skyband-metrics (critérios de dominância Skyband):
  train_time_sec   energy_kwh   total_gflops   emissions_kg_co2   cost_usd

Configurações padrão:
  - Modo: {DEFAULT_MODE}
  - Config: {DEFAULT_CONFIG}
  - Grid config: {DEFAULT_GRID_CONFIG}
  - Parallel: {DEFAULT_PARALLEL}
  - Skyband k: {DEFAULT_SKYBAND_K}
  - SLA profiles: {DEFAULT_SLA_PROFILES}
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "grid"],
        default=DEFAULT_MODE,
        help=f"Modo de execução: 'single' para um único experimento, 'grid' para grid search (padrão: {DEFAULT_MODE})"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Caminho do arquivo de configuração base (padrão: {DEFAULT_CONFIG})"
    )
    
    parser.add_argument(
        "--grid-config",
        type=str,
        default=DEFAULT_GRID_CONFIG,
        help=f"Caminho do arquivo JSON com grade de hiperparâmetros (padrão: {DEFAULT_GRID_CONFIG})"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Número de processos paralelos para grid search (padrão: {DEFAULT_PARALLEL})"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior de grid search usando estado salvo"
    )

    # ── Grupo: análise Skyband ───────────────────────────────────────────────
    skyband_group = parser.add_argument_group(
        "Skyband",
        "Análise multicriterio por dominância de Pareto (Skyband Query Engine)",
    )

    skyband_group.add_argument(
        "--no-skyband",
        action="store_true",
        dest="no_skyband",
        help="Desativa a análise Skyband automática após a execução dos experimentos",
    )

    skyband_group.add_argument(
        "--skyband-only",
        action="store_true",
        help=(
            "Carrega estado existente e executa apenas a análise Skyband, "
            "sem disparar novos experimentos"
        ),
    )

    skyband_group.add_argument(
        "--skyband-k",
        type=int,
        default=DEFAULT_SKYBAND_K,
        metavar="K",
        help=(
            "Ordem do Skyband: retorna experimentos dominados por menos de K outros. "
            "k=1 = frente de Pareto pura. "
            f"k=2 inclui o segundo nível de dominância, etc. (padrão: {DEFAULT_SKYBAND_K})"
        ),
    )

    skyband_group.add_argument(
        "--sla-profile",
        type=str,
        default=None,
        metavar="PERFIL",
        choices=["economico", "sustentavel", "tempo", "balanceado", "dev", "producao"],
        help=(
            "Perfil de SLA predefinido em gridsearch/config/sla_profiles.json. "
            "Sobrescreve --skyband-k, --skyband-metrics e --sla-constraint quando informado. "
            "Opções: economico | sustentavel | tempo | balanceado | dev | producao"
        ),
    )

    skyband_group.add_argument(
        "--sla-constraint",
        action="append",
        metavar="METRICA=VALOR",
        dest="sla_constraints",
        help=(
            "Restrição de SLA no formato metrica=valor_maximo (pode repetir). "
            "Métricas disponíveis (filtro de admissibilidade): train_time_sec, energy_kwh, "
            "peak_ram_mb, emissions_kg_co2, cost_usd. "
            "Ex: --sla-constraint peak_ram_mb=8192 --sla-constraint cost_usd=5.0"
        ),
    )

    skyband_group.add_argument(
        "--skyband-metrics",
        nargs="+",
        metavar="METRICA",
        default=None,
        help=(
            "Lista de métricas a usar na dominância de Pareto (critérios Skyband). "
            "Padrão: train_time_sec energy_kwh total_gflops emissions_kg_co2 cost_usd "
            "(todos os 5 critérios). "
            "Ex: --skyband-metrics train_time_sec cost_usd total_gflops"
        ),
    )

    skyband_group.add_argument(
        "--skyband-compare",
        action="store_true",
        help="Exibe comparação entre Skyband e ranking escalar ponderado (Jaccard + diferenças)",
    )

    skyband_group.add_argument(
        "--skyband-state",
        type=str,
        default=None,
        metavar="ARQUIVO",
        help=(
            "Caminho direto para o arquivo JSON de estado do grid search a ser analisado. "
            "Padrão: detecta automaticamente o arquivo mais recente em "
            "output/experiments/grid_search/"
        ),
    )

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
        # ── Modo: apenas análise Skyband (sem novo grid) ─────────────────────
        if args.skyband_only:
            run_skyband_analysis(
                k=args.skyband_k,
                sla_constraints=sla_dict or None,
                sla_profile_name=args.sla_profile,
                metrics=args.skyband_metrics,
                compare=args.skyband_compare,
                state_file=args.skyband_state,
            )

        elif args.mode == "single":
            run_single_experiment(args.config)
            if not args.no_skyband:
                run_skyband_analysis(
                    k=args.skyband_k,
                    sla_constraints=sla_dict or None,
                    sla_profile_name=args.sla_profile,
                    metrics=args.skyband_metrics,
                    compare=args.skyband_compare,
                    state_file=args.skyband_state,
                )
        elif args.mode == "grid":
            run_grid_search_experiments(
                base_config_path=args.config,
                grid_config_path=args.grid_config,
                parallel=args.parallel,
                resume=args.resume,
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
        else:
            parser.error(f"Modo inválido: {args.mode}")
    
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
