"""
Skyband Dominance Engine
========================

Primitivas de dominância de Pareto, Skyband query e filtro de SLA
pós-execução para seleção multicritério de configurações.

Autor: Gustavo Alexandre
Data: 2026-03-01
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# PORTA DE CONFIGURAÇÃO — Métricas padrão (Hexagonal / OCP)
# ============================================================================
_METRICS_CONFIG_PATH = Path(__file__).parent / "config" / "metrics.json"


def _load_metrics_config() -> tuple[list[str], list[bool]]:
    """Carrega métricas e flags de minimização de arquivo externo.

    O arquivo ``gridsearch/config/metrics.json`` define quais métricas
    são usadas por padrão na dominância Skyband.  Novas métricas podem
    ser adicionadas editando o JSON — sem alterar código (OCP).

    Returns:
        Tupla ``(metric_names, minimize_flags)``.
    """
    with open(_METRICS_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("defaults", [])
    names = [e["name"] for e in entries]
    minimize = [e.get("minimize", True) for e in entries]
    return names, minimize


def _load_quality_metrics_config() -> tuple[list[str], list[bool]]:
    """Carrega métricas de qualidade preditiva (f1_score, accuracy) do JSON.

    Métricas de qualidade devem ser **maximizadas** (``minimize=False``),
    ao contrário das métricas de recurso.  São extraídas de
    ``result["evaluation"]`` em vez de ``result["resources"]``.
    """
    with open(_METRICS_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("quality_metrics", [])
    names = [e["name"] for e in entries]
    minimize = [e.get("minimize", False) for e in entries]
    return names, minimize


DEFAULT_METRICS, DEFAULT_MINIMIZE = _load_metrics_config()
QUALITY_METRICS, QUALITY_MINIMIZE = _load_quality_metrics_config()

# Lookup completo {nome_métrica: minimize_flag} para todas as métricas conhecidas
_METRIC_MINIMIZE_LOOKUP: Dict[str, bool] = {
    **dict(zip(DEFAULT_METRICS, DEFAULT_MINIMIZE)),
    **dict(zip(QUALITY_METRICS, QUALITY_MINIMIZE)),
}


def _get_minimize_flag(metric: str) -> bool:
    """Retorna o flag de minimização para uma métrica.

    Para métricas de recurso (train_time_sec, energy_kwh, etc.) retorna
    ``True`` (minimizar).  Para métricas de qualidade (f1_score, accuracy)
    retorna ``False`` (maximizar).  Métricas desconhecidas assumem ``True``.
    """
    return _METRIC_MINIMIZE_LOOKUP.get(metric, True)


# ============================================================================
# UTILITÁRIOS INTERNOS
# ============================================================================

def _extract_metric_value(result: Dict[str, Any], metric: str) -> float:
    """
    Extrai o valor de uma métrica de um resultado de experimento.

    Hierarquia de busca:
    1. ``result["resources"]`` — métricas de recurso (train_time_sec, etc.)
    2. ``result["evaluation"]`` — métricas de qualidade (f1_score, accuracy)
    3. Raiz do dicionário (compatibilidade retroativa)

    Args:
        result: Dicionário de resultado de um experimento.
        metric: Nome da métrica a extrair.

    Returns:
        Valor numérico da métrica ou float('inf') se ausente/None.
    """
    resources = result.get("resources", {})
    value = resources.get(metric) if isinstance(resources, dict) else None

    if value is None:
        evaluation = result.get("evaluation") or {}
        if isinstance(evaluation, dict):
            value = evaluation.get(metric)

    if value is None:
        value = result.get(metric)

    if value is None:
        return float("inf")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _build_vector(
    result: Dict[str, Any],
    metrics: List[str],
    minimize: List[bool],
) -> List[float]:
    """
    Constrói o vetor de comparação para cálculo de dominância.

    Valores são normalizados pelo sinal de minimização para que
    dominância sempre signifique "menor é melhor".

    Args:
        result:   Dicionário de resultado.
        metrics:  Lista de nomes de métricas.
        minimize: Lista de booleanos (True = minimizar, False = maximizar).

    Returns:
        Lista de floats representando o ponto no espaço de critérios.
    """
    return [
        (1.0 if mini else -1.0) * _extract_metric_value(result, m)
        for m, mini in zip(metrics, minimize)
    ]


# ============================================================================
# FILTRO DE SLA
# ============================================================================

def sla_filter(
    results: List[Dict[str, Any]],
    constraints: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Filtra experimentos que violam as constraints de SLA informadas.

    Constraints são aplicadas como limites superiores sobre as métricas
    de recursos. Apenas experimentos que satisfazem **todas** as constraints
    são mantidos.

    Args:
        results:     Lista de resultados de experimentos.
        constraints: Dicionário {nome_metrica: valor_maximo}.

    Returns:
        Subconjunto de results que satisfaz todas as constraints.
    """
    admissible = []
    rejected = 0

    for r in results:
        if r.get("status") != "success":
            rejected += 1
            continue

        if not constraints:
            admissible.append(r)
            continue

        satisfies_all = all(
            _extract_metric_value(r, metric) <= threshold
            for metric, threshold in constraints.items()
        )

        if satisfies_all:
            admissible.append(r)
        else:
            rejected += 1

    logger.info(
        "SLA filter: %d admissíveis | %d rejeitados | constraints=%s",
        len(admissible), rejected, constraints,
    )
    return admissible


# ============================================================================
# DOMINÂNCIA DE PARETO
# ============================================================================

def dominates(
    vec_i: List[float],
    vec_j: List[float],
) -> bool:
    """
    Verifica se o ponto vec_i domina o ponto vec_j (Pareto).

    e_i domina e_j se:
        - e_i é melhor ou igual em TODOS os critérios, E
        - e_i é estritamente melhor em PELO MENOS UM critério.

    Args:
        vec_i: Vetor de métricas do experimento i (já normalizado por sinal).
        vec_j: Vetor de métricas do experimento j (já normalizado por sinal).

    Returns:
        True se vec_i domina vec_j, False caso contrário.
    """
    at_least_one_better = False

    for a, b in zip(vec_i, vec_j):
        if a > b:
            return False
        if a < b:
            at_least_one_better = True

    return at_least_one_better


# ============================================================================
# FRENTE DE PARETO (k=1)
# ============================================================================

def pareto_front(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    minimize: Optional[List[bool]] = None,
    sla_constraints: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Retorna a frente de Pareto pura (Skyband de ordem k=1).

    Equivalente a skyband_query(results, k=1, ...).

    Args:
        results:         Lista de resultados de experimentos.
        metrics:         Critérios usados na dominância.
        minimize:        Lista de booleanos correspondente a metrics.
        sla_constraints: Constraints de SLA aplicadas antes da dominância.

    Returns:
        Lista de experimentos na frente de Pareto, enriquecidos com
        "domination_count" = 0.
    """
    return skyband_query(
        results,
        k=1,
        sla_constraints=sla_constraints,
        metrics=metrics,
        minimize=minimize,
    )


# ============================================================================
# CONSULTA SKYBAND (k ≥ 1)
# ============================================================================

def skyband_query(
    results: List[Dict[str, Any]],
    k: int = 1,
    sla_constraints: Optional[Dict[str, float]] = None,
    metrics: Optional[List[str]] = None,
    minimize: Optional[List[bool]] = None,
    include_quality_metrics: bool = False,
) -> List[Dict[str, Any]]:
    """
    Consulta Skyband de ordem k com filtro de SLA personalizado.

    Retorna os experimentos dominados por **menos de k** outros pontos,
    aplicando primeiro o filtro de SLA sobre o espaço de candidatos.

    Args:
        results:                 Lista de resultados de experimentos.
        k:                       Ordem do Skyband (>= 1).
        sla_constraints:         Constraints de SLA {metrica: valor_maximo}.
        metrics:                 Lista de nomes de métricas para dominância.
        minimize:                Lista de booleanos por métrica.
        include_quality_metrics: Quando ``True`` e ``metrics is None``,
                                 adiciona ``QUALITY_METRICS`` (f1_score,
                                 accuracy) ao conjunto de critérios com
                                 ``minimize=False`` (maximizar). (BL-07)

    Returns:
        Lista de dicionários de experimento com chaves extras:
          - "domination_count": int
          - "skyband_rank": int

    Raises:
        ValueError: Se k < 1 ou metrics/minimize com tamanhos diferentes.
    """
    if k < 1:
        raise ValueError(f"k deve ser >= 1, recebido: {k}")

    if metrics is None:
        if include_quality_metrics:
            metrics = DEFAULT_METRICS[:] + QUALITY_METRICS[:]
            minimize = DEFAULT_MINIMIZE[:] + QUALITY_MINIMIZE[:]
        else:
            metrics = DEFAULT_METRICS[:]
            minimize = DEFAULT_MINIMIZE[:]
    elif minimize is None:
        # Infere flags de minimização a partir do lookup de todas as métricas
        minimize = [_get_minimize_flag(m) for m in metrics]

    if len(metrics) != len(minimize):
        raise ValueError(
            f"metrics ({len(metrics)}) e minimize ({len(minimize)}) "
            "devem ter o mesmo tamanho."
        )

    # 1. Aplica filtro de SLA
    candidates = sla_filter(results, sla_constraints or {})

    if not candidates:
        logger.warning(
            "Nenhum experimento admissível após filtro SLA (constraints=%s). "
            "Verifique os thresholds ou execute mais experimentos.",
            sla_constraints,
        )
        return []

    # 2. Constrói vetores de métricas normalizados por sinal
    try:
        _use_numpy = True
    except ImportError:
        _use_numpy = False

    if _use_numpy:
        vecs = np.array(
            [_build_vector(r, metrics, minimize) for r in candidates],
            dtype=np.float64,
        )

        n = len(candidates)
        domination_count = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = vecs[j] - vecs[i]
                if np.all(diff <= 0) and np.any(diff < 0):
                    domination_count[i] += 1
                    if domination_count[i] >= k:
                        break
    else:
        vecs = [_build_vector(r, metrics, minimize) for r in candidates]
        n = len(candidates)
        domination_count = [0] * n

        for i in range(n):
            for j in range(n):
                if i != j and dominates(vecs[j], vecs[i]):
                    domination_count[i] += 1
                    if domination_count[i] >= k:
                        break

    # 4. Seleciona o Skyband_k
    skyband: List[Dict[str, Any]] = []

    for i, result in enumerate(candidates):
        if domination_count[i] < k:
            enriched = dict(result)
            enriched["domination_count"] = domination_count[i]
            skyband.append(enriched)

    # 5. Ordena por (domination_count, experiment_idx)
    skyband.sort(key=lambda r: (r["domination_count"], r.get("grid_experiment_idx", 0)))

    # 6. Atribui skyband_rank
    for rank, r in enumerate(skyband):
        r["skyband_rank"] = rank

    logger.info(
        "Skyband_%d: %d/%d candidatos selecionados | metrics=%s | sla=%s",
        k, len(skyband), len(candidates), metrics, sla_constraints,
    )

    return skyband
