"""
SLA Pre-filter — Pré-filtragem de SLA para execução
====================================================

Filtra combinações de hiperparâmetros sabidamente inviáveis **antes**
de disparar processos, reduzindo custo computacional e tempo.

Constraints suportadas:
    - ``peak_ram_mb``    — estimativa baseada em batch_size
    - ``train_time_sec`` — estimativa baseada em baseline × fatores

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .utils import estimate_memory_requirements

logger = logging.getLogger(__name__)

# Constraints avaliáveis antes da execução real
SUPPORTED_EXECUTION_SLA_CONSTRAINTS = {"peak_ram_mb", "train_time_sec"}
MAX_SLA_REJECTED_SAMPLES = 25


# ============================================================================
# INTERFACE PÚBLICA
# ============================================================================

def prefilter_param_grid_by_execution_sla(
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


# ============================================================================
# FUNÇÕES INTERNAS
# ============================================================================

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
