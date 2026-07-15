"""
Funções de execução (runners) do CLI
======================================

Funções que orquestram a execução de experimentos únicos, grid search
e análise Skyband. Também inclui helpers de validação, parsing de SLA
e overrides de dataset.

Autor: Gustavo Alexandre
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from gridsearch.core import run_grid_search, GRID_OUTPUT_DIR
from gridsearch.skyband import (
    compare_skyband_vs_ranking,
    skyband_report,
    DEFAULT_METRICS,
)

from .constants import DEFAULT_SLA_PROFILES, DEFAULT_TRAIN_DATASET

logger = logging.getLogger(__name__)


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
    gpu_list: Optional[List[int]] = None,
):
    """
    Executa um único experimento.

    Args:
        config_path: Caminho do arquivo de configuração.
        train_dataset: Nome do arquivo de treino sem extensão.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no config.
        gpu_list: IDs das GPUs a utilizar (None = detecção automática).
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from experiment import execute_experiment

    logger.info("=" * 70)
    logger.info("MODO: Experimento Único")
    logger.info(f"Configuração: {config_path}")
    if dataset_overrides:
        logger.info(f"Dataset HF overrides: {dataset_overrides}")
    else:
        logger.info(f"Dataset de treino: {train_dataset}.json")
    if gpu_list is not None:
        logger.info(f"GPUs: {gpu_list}")
    logger.info("=" * 70)

    if not validate_paths(config_path):
        sys.exit(1)

    execute_experiment(
        config_path,
        gpu_list=gpu_list,
        parallel_workers=1,
        train_file=train_dataset if train_dataset != DEFAULT_TRAIN_DATASET else None,
        dataset_overrides=dataset_overrides,
    )
    logger.info("Experimento concluído com sucesso!")


def _build_dataset_overrides(args) -> Optional[Dict[str, str]]:
    """Constrói o dict de overrides de [data] a partir dos args CLI HF.

    Retorna ``None`` se nenhum argumento HF foi informado.
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
    require_state: bool = True,
) -> None:
    """
    Carrega resultados do estado do grid search e executa análise Skyband.

    Args:
        k:                Ordem do Skyband (padrão 1 = frente de Pareto).
        sla_constraints:  Dicionário {metrica: valor_max} já processado.
        sla_profile_name: Nome de perfil em ``sla_profiles.json``.
        metrics:          Lista de métricas para dominância.
        compare:          Se True, imprime comparação Skyband vs ranking escalar.
        state_file:       Caminho direto ao JSON de estado.
        require_state:    Quando ``True`` (padrão), encerra o processo com erro
                          se nenhum estado de grid search for encontrado.
                          Quando ``False``, registra aviso e retorna sem análise
                          (comportamento adequado para ``--mode single``, que não
                          gera arquivo de estado).
    """
    # ── Localiza o arquivo de estado ────────────────────────────────────────
    if state_file:
        sf = Path(state_file)
    else:
        candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_state_*.json"), reverse=True)
        if not candidates:
            if require_state:
                logger.error(
                    "Nenhum arquivo de estado encontrado em: %s", GRID_OUTPUT_DIR
                )
                sys.exit(1)
            else:
                logger.info(
                    "Análise Skyband ignorada: nenhum estado de grid search em '%s'. "
                    "Execute '--mode grid' primeiro para gerar o estado, ou use "
                    "'--no-skyband' para suprimir este aviso.",
                    GRID_OUTPUT_DIR,
                )
                return
        sf = candidates[0]

    if not sf.exists():
        if require_state:
            logger.error("Arquivo de estado não encontrado: %s", sf)
            sys.exit(1)
        else:
            logger.warning("Arquivo de estado não encontrado: %s — Skyband ignorado.", sf)
            return

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
            p = r["grid_params"]
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
            p = r["grid_params"]
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
    gpu_ids: Optional[List[int]] = None,
):
    """
    Executa grid search de hiperparâmetros.

    Args:
        base_config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo JSON com grade de hiperparâmetros
        parallel: Número de processos paralelos
        resume: Se True, retoma execução anterior
        sla_profile_name: Nome do perfil de SLA.
        sla_constraints: Constraints SLA manuais para pré-filtro.
        train_dataset: Nome do arquivo de treino sem extensão.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever.
        gpu_ids: IDs das GPUs para distribuição round-robin (None = auto).
    """
    from .sla_summary import _load_latest_grid_state, _emit_sla_execution_summary

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

    # Extrai custo horário por ambiente (fórmula PSLA4ML: cost_usd = t/3600 × rate)
    # Presente em grid_search_multienv.json → environments.details.*.cost_per_hour_usd
    env_cost_registry: Dict[str, float] = {
        name: float(details.get("cost_per_hour_usd", 0.0))
        for name, details in (
            grid_config.get("environments", {}).get("details", {}).items()
        )
        if isinstance(details, dict) and details.get("cost_per_hour_usd") is not None
    }
    if env_cost_registry:
        logger.info(
            "Custo horário por ambiente carregado (fórmula PSLA4ML): %s",
            env_cost_registry,
        )

    # Executa grid search
    results = run_grid_search(
        base_config_path=base_config_path,
        grid_config=grid_config,
        resume=resume,
        parallel=parallel,
        gpu_ids=gpu_ids,
        execution_sla_constraints=execution_sla_constraints or None,
        train_dataset=train_dataset,
        dataset_overrides=dataset_overrides,
        env_cost_registry=env_cost_registry or None,
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
