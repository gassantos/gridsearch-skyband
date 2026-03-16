"""
Grid Search Core - BERT-PLI
============================

Motor de execução para busca em grade de hiperparâmetros.
Versão modularizada com validações de memória integradas.

Uso como módulo:
    from gridsearch import run_grid_search
    
Uso CLI:
    python -m gridsearch.core --config config/experiments/BertPLI.config \
                              --search-config gridsearch/config/grid_search.json \
                              --parallel 2

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import argparse
import json
import logging
import os
import sys
# import multiprocessing
import configparser
import itertools
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.device import get_torch_device
from utils.paths import PathManager
from utils.log_setup import setup_worker_logging, get_log_queue
from .utils import (
    check_memory_availability,
    filter_grid_config,
    ensure_output_directories,
    estimate_memory_requirements,
)

_TDATE = datetime.now().strftime("%Y-%m-%d")
_LOGFILE = PathManager.LOGS_DIR / f"grid_search_{_TDATE}.log"

# Detectado de forma lazy para evitar chamar torch.cuda.is_available() no
# nível do módulo — o que dispara uma mensagem de erro C-level do NVML
# ("gpuGetDeviceCount failed with code 35") em ambientes CPU-only.
_device_type_cache: "str | None" = None


def _get_device_type() -> str:
    """Retorna o tipo de device (CPU/GPU/TPU), com cache para evitar re-detecção."""
    global _device_type_cache
    if _device_type_cache is None:
        _device_type_cache = get_torch_device()['type']
    return _device_type_cache

# Logging configurado via setup_main_logging() em run_grid_search().
# Não chamamos basicConfig aqui para evitar dupla inicialização nos workers.
logger = logging.getLogger(__name__)


# Diretórios
GRID_OUTPUT_DIR = PathManager.EXPERIMENTS_DIR / "grid_search"
GRID_CONFIGS_DIR = GRID_OUTPUT_DIR / "configs"


def _grid_state_file():
    return GRID_OUTPUT_DIR / f"grid_search_state_{_get_device_type()}_{_TDATE}.json"


def _grid_results_file():
    return GRID_OUTPUT_DIR / f"grid_search_results_{_get_device_type()}_{_TDATE}.json"


def _grid_summary_file():
    return GRID_OUTPUT_DIR / f"grid_search_summary_{_get_device_type()}_{_TDATE}.txt"

# Configurações de custo
# Tarifa média de energia em USD/kWh (pode ser configurada via variável de ambiente)
ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))

# Constraints que podem ser avaliadas antes de executar um experimento.
SUPPORTED_EXECUTION_SLA_CONSTRAINTS = {"peak_ram_mb", "train_time_sec"}
MAX_SLA_REJECTED_SAMPLES = 25

# Criar diretórios
ensure_output_directories()


# ============================================================================
# GERAÇÃO DE COMBINAÇÕES
# ============================================================================

def generate_parameter_grid(grid_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Gera todas as combinações possíveis de hiperparâmetros.
    
    Args:
        grid_config: Dicionário com listas de valores para cada hiperparâmetro
        
    Returns:
        Lista de dicionários, cada um representando uma combinação única
        
    Exemplo:
        >>> grid = {
        ...     "learning_rate": [1e-5, 2e-5],
        ...     "batch_size": [8, 16]
        ... }
        >>> generate_parameter_grid(grid)
        [
            {"learning_rate": 1e-5, "batch_size": 8},
            {"learning_rate": 1e-5, "batch_size": 16},
            {"learning_rate": 2e-5, "batch_size": 8},
            {"learning_rate": 2e-5, "batch_size": 16}
        ]
    """
    # Filtra metadados da configuração
    full_config = grid_config if isinstance(grid_config, dict) else {}
    grid_config = filter_grid_config(full_config)

    # Quando a configuração define ambientes ativos, adiciona a dimensão
    # "environment" para expandir hiperparâmetros x ambientes.
    active_envs = (
        full_config.get("environments", {})
        .get("active", [])
        if isinstance(full_config.get("environments", {}), dict)
        else []
    )
    if isinstance(active_envs, list) and active_envs:
        grid_config = {**grid_config, "environment": active_envs}
    
    keys = list(grid_config.keys())
    values = list(grid_config.values())
    
    # Gera produto cartesiano
    combinations = list(itertools.product(*values))
    
    # Converte para lista de dicionários
    param_grid = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_grid.append(param_dict)
    
    logger.info(f"Geradas {len(param_grid)} combinações de hiperparâmetros")
    return param_grid


def create_config_for_combination(
    base_config_path: str,
    params: Dict[str, Any],
    experiment_idx: int,
    train_file: str = "train_task2",
) -> str:
    """
    Cria um arquivo de configuração específico para uma combinação de parâmetros.
    
    Args:
        base_config_path: Caminho do arquivo de configuração base
        params: Dicionário com os parâmetros a serem modificados
        experiment_idx: Índice do experimento na grade
        train_file: Nome do arquivo de treino sem extensão (ex:
            ``"train_task2_v2"``). Substitui ``train_file_list`` na seção
            ``[data]`` do config gerado.
        
    Returns:
        Caminho do novo arquivo de configuração criado
    """
    config = configparser.ConfigParser()
    config.read(base_config_path)
    
    # Atualiza seção [train] com hiperparâmetros
    if "learning_rate" in params:
        config.set("train", "learning_rate", str(params["learning_rate"]))
    
    if "batch_size" in params:
        config.set("train", "batch_size", str(params["batch_size"]))
    
    if "optimizer" in params:
        config.set("train", "optimizer", params["optimizer"])
    
    if "dropout" in params:
        config.set("model", "dropout", str(params["dropout"]))
    
    if "seed" in params:
        config.set("experiment", "seed", str(params["seed"]))

    if "environment" in params:
        if not config.has_section("environment"):
            config.add_section("environment")
        config.set("environment", "runtime_profile", str(params["environment"]))
    
    # Atualiza dataset de treino
    if not config.has_section("data"):
        config.add_section("data")
    config.set("data", "train_file_list", f"{train_file}.json")
    
    # Atualiza nome do experimento
    base_name = config.get("experiment", "name")
    
    # Gera nome descritivo
    param_suffix = "_".join([
        f"{k}{v}".replace(".", "").replace("-", "")
        for k, v in params.items()
    ])
    
    new_name = f"{base_name}_grid{experiment_idx:03d}_{param_suffix}"
    config.set("experiment", "name", new_name)
    
    # Atualiza descrição
    description = f"Grid Search Experiment {experiment_idx}\n"
    description += "Hyperparameters:\n"
    for k, v in params.items():
        description += f"  - {k}: {v}\n"
    config.set("experiment", "description", description)
    
    # Salva nova configuração
    new_config_path = GRID_CONFIGS_DIR / f"grid_exp_{experiment_idx:03d}.config"
    with open(new_config_path, 'w') as f:
        config.write(f)
    
    logger.debug(f"Config criada: {new_config_path}")
    return str(new_config_path)


# ============================================================================
# EXECUÇÃO DE EXPERIMENTOS
# ============================================================================

def run_single_experiment(
    experiment_idx: int,
    config_path: str,
    params: Dict[str, Any],
    gpu_list: List[int] | None = None,
    parallel_workers: int = 1,
    dataset_overrides: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Executa um único experimento e retorna os resultados.

    Args:
        experiment_idx: Índice do experimento
        config_path: Caminho do arquivo de configuração
        params: Parâmetros do experimento
        gpu_list: GPUs a utilizar (ex: [0] ou [1]). None = detecta automaticamente.
        parallel_workers: Número de workers paralelos em uso no grid search
            que chamou este experimento (1 = seqüencial). Salvo nos resultados
            para rastreabilidade.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no config
            (ex: ``{"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"}``).

    Returns:
        Dicionário com resultados do experimento
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from run_experiment import execute_experiment

    logger.info(f"[{experiment_idx}] Iniciando experimento com parâmetros: {params}")

    try:
        # Executa experimento nas GPUs designadas
        execute_experiment(
            config_path,
            gpu_list=gpu_list,
            parallel_workers=parallel_workers,
            dataset_overrides=dataset_overrides,
        )
        
        # Coleta resultados do arquivo JSON mais recente gerado
        metrics_dir = PathManager.BASE_DIR / "output" / "experiments" / "metrics"
        json_files = sorted(metrics_dir.glob("*.json"), key=os.path.getmtime)
        
        if not json_files:
            raise FileNotFoundError("Nenhum arquivo de resultados encontrado")
        
        latest_result = json_files[-1]
        with open(latest_result, 'r') as f:
            result_data = json.load(f)
        
        # Adiciona parâmetros ao resultado
        result_data["grid_params"] = params
        result_data["grid_experiment_idx"] = experiment_idx
        result_data["parallel_workers"] = parallel_workers
        if "environment" in params:
            result_data["selected_environment"] = params["environment"]
        result_data["status"] = "success"
        
        logger.info(f"[{experiment_idx}] Experimento concluído com sucesso")
        return result_data
        
    except Exception as e:
        logger.error(f"[{experiment_idx}] Erro no experimento: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return {
            "grid_experiment_idx": experiment_idx,
            "grid_params": params,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_grid_search(
    base_config_path: str,
    grid_config: Dict[str, List[Any]],
    resume: bool = False,
    parallel: int = 1,
    gpu_ids: List[int] | None = None,
    execution_sla_constraints: Optional[Dict[str, float]] = None,
    train_dataset: str = "train_task2",
    dataset_overrides: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Executa busca em grade completa.

    Args:
        base_config_path: Caminho da configuração base
        grid_config: Configuração da grade de hiperparâmetros
        resume: Se True, retoma execução anterior
        parallel: Número de processos paralelos (1 = sequencial)
        gpu_ids: Lista explícita de GPUs disponíveis para distribuição
                 round-robin entre workers (ex: [0, 1, 2, 3]).
                 None = detecta automaticamente via torch.cuda.
        execution_sla_constraints: Constraints de SLA avaliáveis antes de
             executar o experimento. Atualmente suporta:
             ``peak_ram_mb`` (estimado por batch_size) e
             ``train_time_sec`` (apenas quando ``_meta`` fornece
             ``per_experiment_train_time_sec``).
        train_dataset: Nome do arquivo de treino sem extensão (ex:
                 ``"train_task2_v2"``). Passado a cada config gerado para
                 o experimento. Padrão: ``"train_task2"``.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no config
            (ex: ``{"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"}``).
            Propagado a cada ``run_single_experiment``.

    Returns:
        Lista com resultados de todos os experimentos
    """
    # Carrega estado anterior se existir
    completed_experiments = set()
    all_results = []
    
    if resume and _grid_state_file().exists():
        logger.info("Retomando execução anterior...")
        with open(_grid_state_file(), 'r', encoding='utf-8') as f:
            state = json.load(f)
            completed_experiments = set(state.get("completed_experiments", []))
            all_results = state.get("results", [])
        logger.info(f"Encontrados {len(completed_experiments)} experimentos já concluídos")
    
    # Gera grade de parâmetros (preserva índice original para compatibilidade
    # com retomada e rastreabilidade dos artefatos).
    param_grid = generate_parameter_grid(grid_config)
    indexed_param_grid = list(enumerate(param_grid))
    original_total_experiments = len(indexed_param_grid)

    # Pré-filtro opcional guiado por SLA antes de disparar workers
    sla_filter_info = {
        "enabled": bool(execution_sla_constraints),
        "constraints": execution_sla_constraints or {},
        "original_total_experiments": original_total_experiments,
        "eligible_total_experiments": original_total_experiments,
        "rejected_total_experiments": 0,
        "rejected_by_metric": {},
        "non_evaluable_constraints": [],
    }
    if execution_sla_constraints:
        indexed_param_grid, sla_filter_info = _prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed_param_grid,
            constraints=execution_sla_constraints,
            grid_config=grid_config,
        )

    total_experiments = len(indexed_param_grid)
    eligible_idx = {idx for idx, _ in indexed_param_grid}
    completed_eligible = completed_experiments.intersection(eligible_idx)
    
    logger.info(f"Total de experimentos: {total_experiments}")
    logger.info(f"Já concluídos: {len(completed_eligible)}")
    logger.info(f"Restantes: {total_experiments - len(completed_eligible)}")
    
    # Prepara experimentos pendentes
    pending_experiments = []
    for idx, params in indexed_param_grid:
        if idx in completed_experiments:
            continue
        
        config_path = create_config_for_combination(base_config_path, params, idx, train_file=train_dataset)
        pending_experiments.append((idx, config_path, params))
    
    if not pending_experiments:
        save_state(
            completed_experiments,
            all_results,
            sla_prefilter_info=sla_filter_info,
        )
        logger.info("Todos os experimentos já foram concluídos!")
        return all_results
    
    # Validação de memória antes de executar
    if parallel > 1:
        max_batch_size = max([p.get('batch_size', 16) for _, _, p in pending_experiments])
        is_safe, mem_message = check_memory_availability(parallel, max_batch_size)
        logger.info(f"\n{mem_message}\n")
        
        if not is_safe:
            response = input("Deseja continuar mesmo assim? (s/N): ")
            if response.lower() != 's':
                logger.info("Execução cancelada pelo usuário")
                sys.exit(0)
    
    # Distribui GPUs entre workers em round-robin (um worker → uma GPU)
    import torch as _torch
    _available_gpus: List[int] = (
        gpu_ids
        if gpu_ids is not None
        else list(range(_torch.cuda.device_count()))
    )
    def _gpu_for(idx: int) -> List[int] | None:
        """Retorna [gpu_id] para o worker `idx`, ou None quando não há GPUs."""
        if not _available_gpus:
            return None
        return [_available_gpus[idx % len(_available_gpus)]]

    # Executa experimentos
    if parallel > 1:
        logger.info(
            "Executando em modo paralelo com %d workers | GPUs disponíveis: %s",
            parallel, _available_gpus or "CPU"
        )
        with ProcessPoolExecutor(
            max_workers=parallel,
            initializer=setup_worker_logging,
            initargs=(get_log_queue(),),
        ) as executor:
            futures = {
                executor.submit(
                    run_single_experiment, idx, cfg, params, _gpu_for(idx), parallel,
                    dataset_overrides,
                ): idx
                for idx, cfg, params in pending_experiments
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_experiments.add(idx)
                    
                    # Salva estado incremental
                    save_state(
                        completed_experiments,
                        all_results,
                        sla_prefilter_info=sla_filter_info,
                    )

                    completed_eligible = len(completed_experiments.intersection(eligible_idx))
                    logger.info(f"Progresso: {completed_eligible}/{total_experiments}")
                    
                except Exception as e:
                    logger.error(f"Erro ao executar experimento {idx}: {e}")
    else:
        logger.info("Executando em modo sequencial | GPUs disponíveis: %s", _available_gpus or "CPU")
        for idx, config_path, params in pending_experiments:
            result = run_single_experiment(
                idx, config_path, params, _gpu_for(idx),
                parallel_workers=parallel,
                dataset_overrides=dataset_overrides,
            )
            all_results.append(result)
            completed_experiments.add(idx)

            # Salva estado incremental
            save_state(
                completed_experiments,
                all_results,
                sla_prefilter_info=sla_filter_info,
            )

            completed_eligible = len(completed_experiments.intersection(eligible_idx))
            logger.info(f"Progresso: {completed_eligible}/{total_experiments}")
    
    return all_results


def _prefilter_param_grid_by_execution_sla(
    indexed_param_grid: List[Tuple[int, Dict[str, Any]]],
    constraints: Dict[str, float],
    grid_config: Dict[str, Any],
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """
    Filtra combinações da grade com base em constraints de SLA pré-execução.

    O objetivo é evitar disparo de experimentos sabidamente inviáveis,
    reduzindo custo e tempo antes do ProcessPoolExecutor.
    """
    if not constraints:
        return indexed_param_grid, {
            "enabled": False,
            "constraints": {},
            "original_total_experiments": len(indexed_param_grid),
            "eligible_total_experiments": len(indexed_param_grid),
            "rejected_total_experiments": 0,
            "rejected_by_metric": {},
            "non_evaluable_constraints": [],
            "rejected_samples": [],
            "rejected_samples_limit": MAX_SLA_REJECTED_SAMPLES,
            "rejected_samples_truncated": 0,
        }

    non_evaluable_constraints: List[str] = []
    rejected_by_metric = {metric: 0 for metric in constraints.keys()}
    rejected_samples: List[Dict[str, Any]] = []
    eligible: List[Tuple[int, Dict[str, Any]]] = []

    for idx, params in indexed_param_grid:
        rejection = _first_failing_execution_constraint(
            params=params,
            constraints=constraints,
            grid_config=grid_config,
            non_evaluable_constraints=non_evaluable_constraints,
        )
        if rejection is None:
            eligible.append((idx, params))
        else:
            rejected_metric = rejection["metric"]
            rejected_by_metric[rejected_metric] = (
                rejected_by_metric.get(rejected_metric, 0) + 1
            )
            if len(rejected_samples) < MAX_SLA_REJECTED_SAMPLES:
                rejected_samples.append(
                    {
                        "grid_experiment_idx": idx,
                        "metric": rejected_metric,
                        "estimated_value": rejection["estimated_value"],
                        "threshold": rejection["threshold"],
                        "params": params,
                    }
                )

    info = {
        "enabled": True,
        "constraints": constraints,
        "original_total_experiments": len(indexed_param_grid),
        "eligible_total_experiments": len(eligible),
        "rejected_total_experiments": len(indexed_param_grid) - len(eligible),
        "rejected_by_metric": rejected_by_metric,
        "non_evaluable_constraints": sorted(set(non_evaluable_constraints)),
        "rejected_samples": rejected_samples,
        "rejected_samples_limit": MAX_SLA_REJECTED_SAMPLES,
        "rejected_samples_truncated": max(
            0,
            (len(indexed_param_grid) - len(eligible)) - len(rejected_samples),
        ),
    }

    logger.info(
        "SLA prefilter: elegiveis=%d | rejeitados=%d | constraints=%s",
        info["eligible_total_experiments"],
        info["rejected_total_experiments"],
        constraints,
    )
    _log_sla_prefilter_summary(info)
    if info["non_evaluable_constraints"]:
        logger.warning(
            "SLA prefilter: constraints nao avaliaveis no pre-filtro: %s",
            info["non_evaluable_constraints"],
        )

    return eligible, info


def _first_failing_execution_constraint(
    params: Dict[str, Any],
    constraints: Dict[str, float],
    grid_config: Dict[str, Any],
    non_evaluable_constraints: List[str],
) -> Optional[Dict[str, float | str]]:
    """Retorna detalhes da primeira constraint de execução violada, ou None."""
    for metric, threshold in constraints.items():
        if metric not in SUPPORTED_EXECUTION_SLA_CONSTRAINTS:
            if metric not in non_evaluable_constraints:
                non_evaluable_constraints.append(metric)
            continue

        if metric == "peak_ram_mb":
            batch_size = int(params.get("batch_size", 16))
            estimated_ram_mb = estimate_memory_requirements(
                parallel=1,
                batch_size=batch_size,
            ) * 1024.0
            if estimated_ram_mb > float(threshold):
                return {
                    "metric": metric,
                    "estimated_value": float(estimated_ram_mb),
                    "threshold": float(threshold),
                }

        if metric == "train_time_sec":
            estimated_time_sec = _estimate_train_time_sec(params, grid_config)
            if estimated_time_sec is None:
                if metric not in non_evaluable_constraints:
                    non_evaluable_constraints.append(metric)
                continue
            if estimated_time_sec > float(threshold):
                return {
                    "metric": metric,
                    "estimated_value": float(estimated_time_sec),
                    "threshold": float(threshold),
                }

    return None


def _log_sla_prefilter_summary(info: Dict[str, Any]) -> None:
    """Escreve resumo compacto da triagem SLA para auditoria em logs."""
    if not info.get("enabled"):
        return

    rejected_by_metric = info.get("rejected_by_metric", {})
    ranked = sorted(
        rejected_by_metric.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [(metric, count) for metric, count in ranked if count > 0]
    if ranked:
        rank_text = ", ".join(f"{metric}={count}" for metric, count in ranked)
        logger.info("SLA prefilter: ranking de rejeicoes por metrica -> %s", rank_text)

    samples = info.get("rejected_samples", [])[:3]
    for sample in samples:
        logger.info(
            "SLA prefilter: exemplo rejeitado idx=%s metric=%s estimated=%.4f threshold=%.4f",
            sample.get("grid_experiment_idx"),
            sample.get("metric"),
            float(sample.get("estimated_value", 0.0)),
            float(sample.get("threshold", 0.0)),
        )

    truncated = int(info.get("rejected_samples_truncated", 0) or 0)
    if truncated > 0:
        logger.info(
            "SLA prefilter: %d rejeicoes adicionais omitidas da amostra (limite=%d)",
            truncated,
            int(info.get("rejected_samples_limit", MAX_SLA_REJECTED_SAMPLES)),
        )


def _estimate_train_time_sec(
    params: Dict[str, Any],
    grid_config: Dict[str, Any],
) -> Optional[float]:
    """
    Estima train_time_sec para pré-filtro de SLA.

    Usa, nesta ordem de prioridade:
    1. baseline específico do ambiente em ``environments.details.*.estimated_time_hours``
    2. baseline configurado em ``_meta.time_estimation.baseline_train_time_sec``
    3. fallback legado ``_meta.per_experiment_train_time_sec``

    Em seguida aplica ajustes opcionais por batch size, optimizer e precision.
    Sem baseline válido, retorna None para evitar rejeições indevidas.
    """
    metadata = grid_config.get("_meta", {}) if isinstance(grid_config, dict) else {}
    time_cfg = metadata.get("time_estimation", {}) if isinstance(metadata, dict) else {}

    baseline_sec = _resolve_train_time_baseline_sec(params, grid_config, metadata, time_cfg)
    if baseline_sec is None:
        return None

    reference_batch_size = _safe_float(time_cfg.get("reference_batch_size"), default=16.0)
    batch_size = _safe_float(params.get("batch_size"), default=reference_batch_size)
    if batch_size <= 0:
        batch_size = reference_batch_size
    batch_scale = reference_batch_size / batch_size

    optimizer = str(params.get("optimizer", "")).lower()
    optimizer_factors = time_cfg.get("optimizer_factors", {}) if isinstance(time_cfg, dict) else {}
    optimizer_scale = _safe_float(optimizer_factors.get(optimizer), default=1.0)

    precision = str(params.get("precision", time_cfg.get("default_precision", ""))).lower()
    precision_factors = time_cfg.get("precision_factors", {}) if isinstance(time_cfg, dict) else {}
    precision_scale = _safe_float(precision_factors.get(precision), default=1.0)

    return baseline_sec * batch_scale * optimizer_scale * precision_scale


def _resolve_train_time_baseline_sec(
    params: Dict[str, Any],
    grid_config: Dict[str, Any],
    metadata: Dict[str, Any],
    time_cfg: Dict[str, Any],
) -> Optional[float]:
    """Resolve o baseline de tempo do experimento em segundos."""
    environment = str(params.get("environment", "")).strip().lower()
    if environment:
        env_details = (
            grid_config.get("environments", {})
            .get("details", {})
            .get(environment, {})
        )
        env_hours = (
            env_details.get("estimated_time_hours", {})
            .get("per_experiment")
        )
        env_seconds = _safe_float(env_hours, default=None)
        if env_seconds is not None:
            return env_seconds * 3600.0

    configured_baseline = _safe_float(
        time_cfg.get("baseline_train_time_sec"),
        default=None,
    )
    if configured_baseline is not None:
        return configured_baseline

    legacy_baseline = _safe_float(
        metadata.get("per_experiment_train_time_sec"),
        default=None,
    )
    if legacy_baseline is not None:
        return legacy_baseline

    return None


def _safe_float(value: Any, default: Optional[float]) -> Optional[float]:
    """Converte valor para float, preservando fallback quando inválido."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save_state(
    completed_experiments: set,
    results: List[Dict[str, Any]],
    sla_prefilter_info: Optional[Dict[str, Any]] = None,
):
    """Salva estado da execução para permitir retomada."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "completed_experiments": list(completed_experiments),
        "results": results,
        "sla_prefilter": sla_prefilter_info or {
            "enabled": False,
            "constraints": {},
        },
    }
    
    with open(_grid_state_file(), 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# ============================================================================
# ANÁLISE DE RESULTADOS
# ============================================================================

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analisa resultados e identifica as melhores configurações por múltiplos critérios.
    
    Critérios de análise:
        - Tempo de treinamento (train_time_sec)
        - Eficiência energética (energy_kwh)
        - Uso de memória RAM (peak_ram_mb)
        - Emissão de carbono (emissions_kg_co2)
        - Custo financeiro (cost_usd, calculado a partir de energy_kwh)
    
    Args:
        results: Lista com resultados de todos os experimentos
        
    Returns:
        Dicionário com análise dos resultados incluindo:
        - best_by_time: Melhor configuração por tempo
        - best_by_energy: Melhor configuração por energia
        - best_by_ram: Melhor configuração por memória
        - best_by_carbon: Melhor configuração por emissão de CO2
        - best_by_cost: Melhor configuração por custo financeiro
    """
    logger.info("Analisando resultados...")
    
    # Filtra experimentos bem-sucedidos
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    logger.info(f"Experimentos bem-sucedidos: {len(successful)}")
    logger.info(f"Experimentos falhos: {len(failed)}")
    
    if not successful:
        logger.warning("Nenhum experimento foi concluído com sucesso!")
        return {
            "total_experiments": len(results),
            "successful": 0,
            "failed": len(failed),
            "best_config": None
        }
    
    # Ordena por tempo de treinamento (menor é melhor)
    sorted_by_time = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("train_time_sec", float('inf')))
    )
    
    # Ordena por eficiência energética (menor é melhor)
    sorted_by_energy = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("energy_kwh", float('inf')))
        if x.get("resources", {}).get("energy_kwh") is not None else float('inf')
    )
    
    # Ordena por uso de RAM (menor é melhor)
    sorted_by_ram = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("peak_ram_mb", float('inf')))
        if x.get("resources", {}).get("peak_ram_mb") is not None else float('inf')
    )
    
    # Ordena por emissão de CO2 (menor é melhor)
    sorted_by_carbon = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("emissions_kg_co2", float('inf')))
        if x.get("resources", {}).get("emissions_kg_co2") is not None else float('inf')
    )
    
    # Calcula custo financeiro e ordena (menor é melhor)
    for result in successful:
        energy_kwh = result.get("resources", {}).get("energy_kwh")
        if energy_kwh is not None:
            cost_usd = float(energy_kwh) * ENERGY_COST_USD_PER_KWH
            result["resources"]["cost_usd"] = cost_usd
        else:
            result["resources"]["cost_usd"] = None
    
    sorted_by_cost = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("cost_usd", float('inf')))
        if x.get("resources", {}).get("cost_usd") is not None else float('inf')
    )
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "energy_cost_usd_per_kwh": ENERGY_COST_USD_PER_KWH,
        
        "best_by_time": {
            "experiment_idx": sorted_by_time[0]["grid_experiment_idx"],
            "params": sorted_by_time[0]["grid_params"],
            "train_time_sec": sorted_by_time[0]["resources"]["train_time_sec"]
        } if sorted_by_time else None,
        
        "best_by_energy": {
            "experiment_idx": sorted_by_energy[0]["grid_experiment_idx"],
            "params": sorted_by_energy[0]["grid_params"],
            "energy_kwh": sorted_by_energy[0]["resources"].get("energy_kwh")
        } if sorted_by_energy and sorted_by_energy[0]["resources"].get("energy_kwh") else None,
        
        "best_by_ram": {
            "experiment_idx": sorted_by_ram[0]["grid_experiment_idx"],
            "params": sorted_by_ram[0]["grid_params"],
            "peak_ram_mb": sorted_by_ram[0]["resources"].get("peak_ram_mb")
        } if sorted_by_ram and sorted_by_ram[0]["resources"].get("peak_ram_mb") else None,
        
        "best_by_carbon": {
            "experiment_idx": sorted_by_carbon[0]["grid_experiment_idx"],
            "params": sorted_by_carbon[0]["grid_params"],
            "emissions_kg_co2": sorted_by_carbon[0]["resources"].get("emissions_kg_co2")
        } if sorted_by_carbon and sorted_by_carbon[0]["resources"].get("emissions_kg_co2") else None,
        
        "best_by_cost": {
            "experiment_idx": sorted_by_cost[0]["grid_experiment_idx"],
            "params": sorted_by_cost[0]["grid_params"],
            "cost_usd": sorted_by_cost[0]["resources"].get("cost_usd")
        } if sorted_by_cost and sorted_by_cost[0]["resources"].get("cost_usd") else None,
        
        "all_results": results
    }
    
    return analysis


def generate_summary_report(analysis: Dict[str, Any]) -> str:
    """
    Gera relatório textual resumido dos resultados.
    
    Args:
        analysis: Dicionário com análise dos resultados
        
    Returns:
        String formatada com o relatório
    """
    report = []
    report.append("=" * 80)
    report.append("GRID SEARCH - RELATÓRIO DE RESULTADOS")
    report.append("=" * 80)
    report.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append("")
    
    report.append("RESUMO GERAL:")
    report.append(f"  Total de experimentos: {analysis['total_experiments']}")
    report.append(f"  Bem-sucedidos: {analysis['successful']}")
    report.append(f"  Falhos: {analysis['failed']}")
    report.append("")
    
    if analysis.get("best_by_time"):
        report.append("MELHOR CONFIGURAÇÃO (Tempo de Treinamento):")
        best = analysis["best_by_time"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Tempo: {best['train_time_sec']} segundos")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_energy"):
        report.append("MELHOR CONFIGURAÇÃO (Eficiência Energética):")
        best = analysis["best_by_energy"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Energia: {best['energy_kwh']} kWh")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_ram"):
        report.append("MELHOR CONFIGURAÇÃO (Uso de Memória RAM):")
        best = analysis["best_by_ram"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  RAM Pico: {best['peak_ram_mb']} MB")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_carbon"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Emissão de Carbono):")
        best = analysis["best_by_carbon"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Emissão CO2: {best['emissions_kg_co2']:.6f} kg")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_cost"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Custo Financeiro):")
        best = analysis["best_by_cost"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Custo: ${best['cost_usd']:.4f} USD")
        report.append(f"  (Tarifa: ${analysis['energy_cost_usd_per_kwh']:.4f}/kWh)")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    report.append("=" * 80)
    
    # Adiciona estatísticas gerais
    if analysis['successful'] > 0:
        report.append("")
        report.append("ESTATÍSTICAS GERAIS DOS EXPERIMENTOS BEM-SUCEDIDOS:")
        report.append("")
        
        # Calcula estatísticas agregadas
        all_successful = [r for r in analysis['all_results'] if r.get('status') == 'success']
        
        # Tempo total
        total_time = sum(
            float(r.get('resources', {}).get('train_time_sec', 0))
            for r in all_successful
        )
        report.append(f"  Tempo total de treinamento: {total_time:.2f} segundos ({total_time/3600:.2f} horas)")
        
        # Energia total
        total_energy = sum(
            float(r.get('resources', {}).get('energy_kwh', 0) or 0)
            for r in all_successful
        )
        if total_energy > 0:
            report.append(f"  Energia total consumida: {total_energy:.4f} kWh")
        
        # CO2 total
        total_co2 = sum(
            float(r.get('resources', {}).get('emissions_kg_co2', 0) or 0)
            for r in all_successful
        )
        if total_co2 > 0:
            report.append(f"  Emissão total de CO2: {total_co2:.6f} kg ({total_co2*1000:.2f} g)")
        
        # Custo total
        total_cost = sum(
            float(r.get('resources', {}).get('cost_usd', 0) or 0)
            for r in all_successful
        )
        if total_cost > 0:
            report.append(f"  Custo financeiro total: ${total_cost:.4f} USD")
        
        report.append("")
        report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Ponto de entrada CLI para execução de grid search via ``python -m gridsearch.core``.

    Analisa argumentos de linha de comando e despacha para uma das operações:

    - Busca em grade completa (``--config`` + ``--search-config``)
    - Retomada de execução interrompida (``--resume``)
    - Análise de resultados existentes (``--analyze-only``)
    """
    parser = argparse.ArgumentParser(
        description="Grid Search para hiperparâmetros do BERT-PLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Busca completa com configuração JSON
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json

  # Execução paralela com 4 processos
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json \\
                            --parallel 4

  # Retomar execução interrompida
  python -m gridsearch.core --resume

  # Analisar resultados existentes
  python -m gridsearch.core --analyze-only
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Caminho do arquivo de configuração base"
    )
    
    parser.add_argument(
        "--search-config",
        type=str,
        help="Caminho do arquivo JSON com grade de hiperparâmetros"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior usando estado salvo"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Número de processos paralelos (padrão: 1 = sequencial)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Apenas analisa resultados existentes sem executar novos experimentos"
    )
    
    args = parser.parse_args()
    
    # Modo: apenas análise
    if args.analyze_only:
        # Tenta encontrar o arquivo de resultados: com data de hoje, sem data, ou o mais recente
        results_file = None
        if _grid_results_file().exists():
            results_file = _grid_results_file()
        else:
            # Fallback 1: arquivo sem data
            fallback_no_date = GRID_OUTPUT_DIR / "grid_search_results.json"
            if fallback_no_date.exists():
                results_file = fallback_no_date
                logger.warning(f"Arquivo do dia não encontrado. Usando: {results_file}")
            else:
                # Fallback 2: arquivo com data mais recente disponível
                candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_results_*.json"), reverse=True)
                if candidates:
                    results_file = candidates[0]
                    logger.warning(f"Arquivo do dia não encontrado. Usando o mais recente: {results_file}")
        
        if results_file is None:
            logger.error(f"Nenhum arquivo de resultados encontrado em: {GRID_OUTPUT_DIR}")
            sys.exit(1)
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        analysis = analyze_results(results)
        report = generate_summary_report(analysis)
        
        print("\n" + report)
        
        # Salva com data e também como arquivo canônico sem data
        with open(_grid_summary_file(), 'w', encoding='utf-8') as f:
            f.write(report)
        
        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Relatório salvo em: {_grid_summary_file()}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")
        return
    
    # Modo: retomar execução
    if args.resume:
        if not _grid_state_file().exists():
            logger.error(f"Arquivo de estado não encontrado: {_grid_state_file()}")
            sys.exit(1)
        
        logger.info("Retomando execução...")
        # Continua com os mesmos parâmetros
        
    else:
        # Modo: nova execução
        if not args.config or not args.search_config:
            parser.error("--config e --search-config são obrigatórios para nova execução")
        
        if not os.path.exists(args.config):
            logger.error(f"Arquivo de configuração não encontrado: {args.config}")
            sys.exit(1)
        
        if not os.path.exists(args.search_config):
            logger.error(f"Arquivo de busca não encontrado: {args.search_config}")
            sys.exit(1)
        
        # Carrega configuração da grade
        with open(args.search_config, 'r', encoding='utf-8') as f:
            grid_config = json.load(f)
        
        logger.info(f"Configuração base: {args.config}")
        logger.info(f"Grade de hiperparâmetros: {args.search_config}")
        logger.info(f"Modo de execução: {'Paralelo (' + str(args.parallel) + ' workers)' if args.parallel > 1 else 'Sequencial'}")
        
        # Executa grid search
        results = run_grid_search(
            base_config_path=args.config,
            grid_config=grid_config,
            resume=args.resume,
            parallel=args.parallel
        )
        
        # Salva resultados completos
        with open(_grid_results_file(), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Resultados completos salvos em: {_grid_results_file()}")
        
        # Analisa e gera relatório
        analysis = analyze_results(results)
        report = generate_summary_report(analysis)
        
        print("\n" + report)
        
        # Salva com data e também como arquivo canônico sem data
        with open(_grid_summary_file(), 'w', encoding='utf-8') as f:
            f.write(report)
        
        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Relatório salvo em: {_grid_summary_file()}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")


if __name__ == "__main__":
    main()
