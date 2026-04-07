"""
Skyband vs Scalar Ranking Comparison
=====================================

Compara a abordagem Skyband (Pareto multicritério) com o ranking
escalar ponderado tradicional.

Autor: Gustavo Alexandre
Data: 2026-03-01
"""

import logging
from typing import Any, Dict, List, Optional

from .dominance import (
    DEFAULT_METRICS,
    _extract_metric_value,
    skyband_query,
    sla_filter,
)

logger = logging.getLogger(__name__)


def compare_skyband_vs_ranking(
    results: List[Dict[str, Any]],
    sla: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    k: int = 1,
) -> Dict[str, Any]:
    """
    Compara o resultado da consulta Skyband com o ranking escalar ponderado.

    O ranking escalar é o método atual em analysis.rank_configurations(),
    que colapsa todos os critérios em um único score via normalização
    min-max ponderada. Esta função evidencia as diferenças qualitativas
    entre as duas abordagens.

    Args:
        results:  Lista de resultados de experimentos.
        sla:      Constraints de SLA aplicadas a ambas as abordagens.
        metrics:  Critérios usados (padrão: DEFAULT_METRICS).
        weights:  Pesos para o ranking escalar (padrão: iguais).
        k:        Ordem do Skyband.

    Returns:
        Dicionário com:
          - "skyband":             lista de experimentos no Skyband_k
          - "scalar_top":          lista dos top-|Skyband| pelo ranking escalar
          - "only_in_skyband":     índices presentes no Skyband mas não no top escalar
          - "only_in_scalar":      índices presentes no top escalar mas não no Skyband
          - "intersection":        índices em ambos
          - "jaccard_similarity":  |interseção| / |união|
          - "skyband_size":        tamanho do Skyband
          - "scalar_top_size":     tamanho do conjunto escalar comparado
          - "k":                   ordem do Skyband utilizada
          - "sla":                 constraints de SLA aplicadas
          - "metrics":             métricas utilizadas
    """
    if metrics is None:
        metrics = DEFAULT_METRICS[:]
    if weights is None:
        weights = [1.0] * len(metrics)

    # --- Skyband ---
    sb_results = skyband_query(results, k=k, sla_constraints=sla, metrics=metrics)
    sb_size = len(sb_results)
    sb_indices = {r.get("grid_experiment_idx") for r in sb_results}

    # --- Ranking escalar (reimplementado localmente para independência) ---
    candidates = sla_filter(results, sla or {})

    # Coleta valores brutos por métrica
    raw_values: List[List[Optional[float]]] = []
    for m in metrics:
        col = [_extract_metric_value(r, m) for r in candidates]
        raw_values.append(col)

    # Normalização min-max por coluna
    normalized: List[List[float]] = []
    for col in raw_values:
        valid = [v for v in col if v != float("inf") and v is not None]
        if not valid:
            normalized.append([0.0] * len(col))
            continue
        mn, mx = min(valid), max(valid)
        rng = mx - mn if mx != mn else 1.0
        normalized.append(
            [(v - mn) / rng if v != float("inf") else 1.0 for v in col]
        )

    # Score escalar ponderado
    scored = []
    for i, r in enumerate(candidates):
        score = sum(weights[j] * normalized[j][i] for j in range(len(metrics)))
        scored.append((score, r.get("grid_experiment_idx"), r))

    scored.sort(key=lambda x: x[0])
    scalar_top_n = scored[:sb_size]
    scalar_indices = {item[1] for item in scalar_top_n}

    # --- Análise de diferenças ---
    only_sb = sorted(sb_indices - scalar_indices)
    only_sc = sorted(scalar_indices - sb_indices)
    intersection = sorted(sb_indices & scalar_indices)
    union = sb_indices | scalar_indices
    jaccard = len(intersection) / len(union) if union else 1.0

    report = {
        "skyband": sb_results,
        "scalar_top": [item[2] for item in scalar_top_n],
        "only_in_skyband": only_sb,
        "only_in_scalar": only_sc,
        "intersection": intersection,
        "jaccard_similarity": jaccard,
        "skyband_size": sb_size,
        "scalar_top_size": sb_size,
        "k": k,
        "sla": sla,
        "metrics": metrics,
    }

    logger.info(
        "Skyband vs Escalar | k=%d | Jaccard=%.3f | "
        "só Skyband=%d | só Escalar=%d | interseção=%d",
        k, jaccard, len(only_sb), len(only_sc), len(intersection),
    )

    return report
