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
from typing import Optional, Dict, Any, List

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
DEFAULT_TRAIN_DATASET = "train_task2"


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


def run_single_experiment(
    config_path: str,
    train_dataset: str = DEFAULT_TRAIN_DATASET,
    dataset_overrides: Optional[Dict[str, str]] = None,
):
    """
    Executa um único experimento.

    Args:
        config_path: Caminho do arquivo de configuração.
        train_dataset: Nome do arquivo de treino sem extensão.
            Padrão: ``DEFAULT_TRAIN_DATASET`` (``"train_task2"``). Outras
            opções: ``"train_task2_v2"``, ``"train_task2_v3"``.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no
            config (ex: ``{"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"}``).
            Quando definido, ativa automaticamente HuggingFaceDataset.
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from run_experiment import execute_experiment

    logger.info("=" * 70)
    logger.info("MODO: Experimento Único")
    logger.info(f"Configuração: {config_path}")
    if dataset_overrides:
        logger.info(f"Dataset HF overrides: {dataset_overrides}")
    else:
        logger.info(f"Dataset de treino: {train_dataset}.json")
    logger.info("=" * 70)

    if not validate_paths(config_path):
        sys.exit(1)

    execute_experiment(
        config_path,
        parallel_workers=1,
        train_file=train_dataset if train_dataset != DEFAULT_TRAIN_DATASET else None,
        dataset_overrides=dataset_overrides,
    )
    logger.info("Experimento concluído com sucesso!")


def _build_dataset_overrides(args) -> Optional[Dict[str, str]]:
    """Constrói o dict de overrides de [data] a partir dos args CLI HF.

    Retorna ``None`` se nenhum argumento HF foi informado.
    Quando ``--dataset-source`` é fornecido, ativa automaticamente
    ``*_dataset_type = HuggingFace`` para train/valid/test.
    """
    if not args.dataset_source:
        return None

    overrides: Dict[str, str] = {
        "hf_dataset_source": args.dataset_source,
        "train_dataset_type": "HuggingFace",
        "valid_dataset_type": "HuggingFace",
        "test_dataset_type": "HuggingFace",
    }
    if args.dataset_id:
        overrides["hf_dataset_id"] = args.dataset_id
    if args.dataset_config:
        overrides["hf_dataset_config"] = args.dataset_config

    return overrides


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


def _load_sla_profile(profile_name: str) -> dict:
    """
    Carrega um perfil de SLA do arquivo de perfis padrão.

    Returns:
        Dicionário do perfil solicitado.

    Raises:
        ValueError: quando o perfil não existe.
        FileNotFoundError: quando o arquivo de perfis não existe.
    """
    profiles_path = Path(DEFAULT_SLA_PROFILES)
    if not profiles_path.exists():
        raise FileNotFoundError(f"Arquivo de perfis SLA não encontrado: {profiles_path}")

    with open(profiles_path, encoding="utf-8") as f:
        sla_cfg = json.load(f)

    profiles = sla_cfg.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Perfil '{profile_name}' não encontrado. Disponíveis: {available}"
        )

    return profiles[profile_name]


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
        try:
            profile = _load_sla_profile(sla_profile_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            sys.exit(1)

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
    resume: bool = False,
    sla_profile_name: Optional[str] = None,
    sla_constraints: Optional[dict] = None,
    train_dataset: str = DEFAULT_TRAIN_DATASET,
    dataset_overrides: Optional[Dict[str, str]] = None,
):
    """
    Executa grid search de hiperparâmetros.

    Args:
        base_config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo JSON com grade de hiperparâmetros
        parallel: Número de processos paralelos
        resume: Se True, retoma execução anterior
        sla_profile_name: Nome do perfil de SLA (se informado, sobrescreve
            ``sla_constraints`` para pré-filtro de execução).
        sla_constraints: Constraints SLA manuais para pré-filtro de execução.
        train_dataset: Nome do arquivo de treino sem extensão.
            Padrão: ``DEFAULT_TRAIN_DATASET`` (``"train_task2"``). Outras
            opções: ``"train_task2_v2"``, ``"train_task2_v3"``.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no
            config (ex: ``{"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"}``).
    """
    logger.info("=" * 70)
    logger.info("MODO: Grid Search")
    logger.info(f"Configuração base: {base_config_path}")
    logger.info(f"Grid config: {grid_config_path}")
    logger.info(f"Execução: {'Paralela (' + str(parallel) + ' workers)' if parallel > 1 else 'Sequencial'}")
    if dataset_overrides:
        logger.info(f"Dataset HF overrides: {dataset_overrides}")
    else:
        logger.info(f"Dataset de treino: {train_dataset}.json")
    logger.info("=" * 70)
    
    if not validate_paths(base_config_path, grid_config_path):
        sys.exit(1)
    
    # Carrega configuração da grade
    with open(grid_config_path, 'r', encoding='utf-8') as f:
        grid_config = json.load(f)
    
    execution_sla_constraints = dict(sla_constraints or {})
    if sla_profile_name:
        try:
            profile = _load_sla_profile(sla_profile_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            sys.exit(1)

        execution_sla_constraints = {
            metric: value
            for metric, value in profile.get("constraints", {}).items()
            if value is not None
        }
        logger.info(
            "Pré-filtro SLA de execução ativo via perfil '%s': constraints=%s",
            sla_profile_name,
            execution_sla_constraints,
        )
    elif execution_sla_constraints:
        logger.info(
            "Pré-filtro SLA de execução ativo via --sla-constraint: constraints=%s",
            execution_sla_constraints,
        )

    # Executa grid search
    results = run_grid_search(
        base_config_path=base_config_path,
        grid_config=grid_config,
        resume=resume,
        parallel=parallel,
        execution_sla_constraints=execution_sla_constraints or None,
        train_dataset=train_dataset,
        dataset_overrides=dataset_overrides,
    )
    
    logger.info("Grid search concluído com sucesso!")
    logger.info(f"Total de experimentos executados: {len(results)}")

    # Exibe resumo SLA pré-execução (impresso + log) para auditoria final.
    state = _load_latest_grid_state()
    sla_prefilter = state.get("sla_prefilter") if state else None
    state_results = state.get("results", []) if state else []
    _emit_sla_execution_summary(
        sla_prefilter=sla_prefilter,
        sla_profile_name=sla_profile_name,
        results=state_results,
    )


def _load_latest_grid_state() -> Optional[Dict[str, Any]]:
    """Carrega o arquivo de estado mais recente do grid search."""
    candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_state_*.json"), reverse=True)
    if not candidates:
        return None

    state_file = candidates[0]
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Não foi possível ler estado para resumo SLA: %s", state_file)
        return None

    return state


def _emit_sla_execution_summary(
    sla_prefilter: Optional[Dict[str, Any]],
    sla_profile_name: Optional[str],
    results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Imprime e registra no logger o resumo final da triagem SLA de execução."""
    if not sla_prefilter or not sla_prefilter.get("enabled"):
        return

    lines = _build_sla_execution_summary_lines(
        sla_prefilter=sla_prefilter,
        sla_profile_name=sla_profile_name,
        results=results,
    )

    print()
    for line in lines:
        print(line)

    for line in lines:
        if line.strip():
            logger.info(line)


def _build_sla_execution_summary_lines(
    sla_prefilter: Dict[str, Any],
    sla_profile_name: Optional[str],
    results: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Monta linhas do bloco de resumo SLA de execução para print/log."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("RESUMO SLA (EXECUÇÃO)")
    lines.append("=" * 72)
    lines.append(f"  Perfil SLA          : {sla_profile_name or 'custom/constraints'}")
    lines.append(f"  Constraints         : {sla_prefilter.get('constraints', {})}")
    lines.append(
        "  Experimentos        : "
        f"originais={sla_prefilter.get('original_total_experiments', 0)} | "
        f"elegíveis={sla_prefilter.get('eligible_total_experiments', 0)} | "
        f"rejeitados={sla_prefilter.get('rejected_total_experiments', 0)}"
    )

    lines.extend(_build_execution_kpi_lines(results or []))

    ranked = sorted(
        (sla_prefilter.get("rejected_by_metric") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [(metric, count) for metric, count in ranked if count > 0]
    if ranked:
        ranking_text = ", ".join(f"{metric}={count}" for metric, count in ranked)
        lines.append(f"  Ranking rejeições   : {ranking_text}")

    non_eval = sla_prefilter.get("non_evaluable_constraints") or []
    if non_eval:
        lines.append(f"  Não avaliáveis      : {non_eval}")

    sample_list = (sla_prefilter.get("rejected_samples") or [])[:3]
    if sample_list:
        lines.append("  Exemplos rejeitados :")
        for sample in sample_list:
            lines.append(
                "    - "
                f"idx={sample.get('grid_experiment_idx')} "
                f"metric={sample.get('metric')} "
                f"estimated={float(sample.get('estimated_value', 0.0)):.4f} "
                f"threshold={float(sample.get('threshold', 0.0)):.4f}"
            )

    truncated = int(sla_prefilter.get("rejected_samples_truncated", 0) or 0)
    if truncated > 0:
        lines.append(
            "  Amostra truncada    : "
            f"{truncated} rejeições omitidas (limite={sla_prefilter.get('rejected_samples_limit')})"
        )

    lines.append("=" * 72)
    return lines


def _build_execution_kpi_lines(results: List[Dict[str, Any]]) -> List[str]:
    """Monta linhas de KPIs agregados da execução real dos experimentos."""
    lines: List[str] = []

    executed = [r for r in results if isinstance(r, dict)]
    successful = [r for r in executed if r.get("status") == "success"]
    failed = [r for r in executed if r.get("status") == "failed"]

    lines.append(
        "  Execução real       : "
        f"rodados={len(executed)} | sucesso={len(successful)} | falha={len(failed)}"
    )

    def _values(path1: str, path2: str) -> List[float]:
        vals: List[float] = []
        for item in successful:
            sub = item.get(path1, {}) if isinstance(item.get(path1), dict) else {}
            val = sub.get(path2)
            if val is None:
                continue
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                continue
        return vals

    time_vals = _values("resources", "train_time_sec")
    if time_vals:
        total_time = sum(time_vals)
        lines.append(
            "  KPI tempo           : "
            f"media={total_time / len(time_vals):.2f}s | total={total_time:.2f}s"
        )

    energy_vals = _values("resources", "energy_kwh")
    if energy_vals:
        lines.append(f"  KPI energia         : total={sum(energy_vals):.6f} kWh")

    co2_vals = _values("resources", "emissions_kg_co2")
    if co2_vals:
        lines.append(f"  KPI CO2             : total={sum(co2_vals):.6f} kg")

    cost_vals = _values("resources", "cost_usd")
    if cost_vals:
        lines.append(f"  KPI custo           : total=${sum(cost_vals):.6f} USD")

    f1_vals: List[float] = []
    for item in successful:
        evaluation = item.get("evaluation", {}) if isinstance(item.get("evaluation"), dict) else {}
        raw_f1 = evaluation.get("f1_score")
        if raw_f1 is None:
            continue
        try:
            f1_vals.append(float(raw_f1))
        except (TypeError, ValueError):
            continue
    if f1_vals:
        lines.append(
            "  KPI F1              : "
            f"melhor={max(f1_vals):.4f} | media={sum(f1_vals)/len(f1_vals):.4f}"
        )

    return lines


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

  # Usar dataset público do HuggingFace Hub (glue/mrpc)
  python -m main --mode single \\
      --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc

  # Usar dataset local JSONL via HuggingFace Datasets (substitui config)
  python -m main --mode single \\
      --dataset-source local_json

  # Grid search com dataset do Hub + SLA
  python -m main --mode grid --sla-profile dev \\
      --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc

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
  - Dataset: {DEFAULT_TRAIN_DATASET}
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

    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=["train_task2", "train_task2_v2", "train_task2_v3"],
        default=DEFAULT_TRAIN_DATASET,
        dest="train_dataset",
        help=(
            f"Arquivo de treino a utilizar (sem extensão). "
            f"Padrão: {DEFAULT_TRAIN_DATASET}. "
            "Opções: train_task2 | train_task2_v2 | train_task2_v3"
        ),
    )

    # ── Grupo: dataset HuggingFace ───────────────────────────────────────────
    hf_group = parser.add_argument_group(
        "Dataset HuggingFace",
        "Sobrescreve as chaves [data] do config para usar HuggingFaceDataset.",
    )

    hf_group.add_argument(
        "--dataset-source",
        type=str,
        choices=["hub", "local_json"],
        default=None,
        dest="dataset_source",
        metavar="FONTE",
        help=(
            "Fonte do dataset: 'hub' (HuggingFace Hub) ou 'local_json' (JSONL local). "
            "Quando informado, ativa automaticamente train/valid/test_dataset_type=HuggingFace."
        ),
    )

    hf_group.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        dest="dataset_id",
        metavar="ID",
        help=(
            "ID do dataset no HuggingFace Hub (ex: 'nyu-mll/glue') ou "
            "caminho local ao usar --dataset-source local_json."
        ),
    )

    hf_group.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        dest="dataset_config",
        metavar="CONFIG",
        help=(
            "Subconfiguração do dataset no Hub (ex: 'mrpc' para glue). "
            "Corresponde ao parâmetro 'name' do load_dataset."
        ),
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
            run_single_experiment(
                args.config,
                train_dataset=args.train_dataset,
                dataset_overrides=_build_dataset_overrides(args),
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
        elif args.mode == "grid":
            run_grid_search_experiments(
                base_config_path=args.config,
                grid_config_path=args.grid_config,
                parallel=args.parallel,
                resume=args.resume,
                sla_profile_name=args.sla_profile,
                sla_constraints=sla_dict or None,
                train_dataset=args.train_dataset,
                dataset_overrides=_build_dataset_overrides(args),
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
