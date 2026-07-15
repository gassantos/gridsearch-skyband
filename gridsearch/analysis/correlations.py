"""
Análise de correlações hiperparâmetro × métrica e detecção de multicolinearidade
==================================================================================

Correlação de Pearson entre hiperparâmetros numéricos e métricas de recurso,
além de detecção de multicolinearidade entre métricas do Skyband (BL-06).

Contexto (Seção 4 do artigo PSLA4ML):
    "As métricas train_time_sec, energy_kwh, emissions_kg_co2, cost_usd formam
     um conjunto de alta colinearidade, com coeficientes r > 0,99 em todos os
     pares.  Quando critérios altamente correlacionados são combinados no Skyband,
     a fronteira tende a colapsar para poucos pontos extremos."

Autor: Gustavo Alexandre
"""

import logging
import statistics as _stats
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .metrics_config import _get_resource_metrics

logger = logging.getLogger(__name__)

# Limiar padrão de multicolinearidade (|r| ≥ threshold → alerta)
DEFAULT_COLLINEARITY_THRESHOLD = 0.95


def compute_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Calcula correlação de Pearson entre duas listas de valores.

    Args:
        x: Lista de valores da variável X
        y: Lista de valores da variável Y

    Returns:
        Coeficiente de correlação de Pearson ou None se não for possível calcular
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    try:
        n = len(x)

        mean_x = _stats.mean(x)
        mean_y = _stats.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        denominator = (denominator_x * denominator_y) ** 0.5

        if denominator == 0:
            return None

        return numerator / denominator

    except Exception as e:
        logger.warning(f"Erro ao calcular correlação: {e}")
        return None


def analyze_correlations(results: List[Dict[str, Any]]) -> Dict[str, float | None]:
    """
    Analisa correlações entre hiperparâmetros numéricos e métricas.

    Args:
        results: Lista de resultados

    Returns:
        Dicionário com coeficientes de correlação
    """
    logger.info("Analisando correlações...")

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        return {}

    # Descoberta dinâmica de hiperparâmetros numéricos
    hp_names: set[str] = set()
    for result in successful:
        for key, val in result.get("grid_params", {}).items():
            try:
                float(val)
                hp_names.add(key)
            except (ValueError, TypeError):
                continue

    # Métricas de recurso numéricas a correlacionar (chaves de resources)
    metric_keys = [m["key"] for m in _get_resource_metrics()]

    correlations: dict[str, float | None] = {}

    for hp in sorted(hp_names):
        for metric in metric_keys:
            hp_vals: list[float] = []
            m_vals: list[float] = []

            for result in successful:
                params = result.get("grid_params", {})
                resources = result.get("resources", {})

                hp_raw = params.get(hp)
                m_raw = resources.get(metric)

                if hp_raw is None or m_raw is None:
                    continue
                try:
                    hp_vals.append(float(hp_raw))
                    m_vals.append(float(m_raw))
                except (ValueError, TypeError):
                    continue

            if len(hp_vals) >= 2:
                correlations[f"{hp}_vs_{metric}"] = compute_correlation(hp_vals, m_vals)

    return correlations


# ============================================================================
# BL-06 — DETECÇÃO DE MULTICOLINEARIDADE ENTRE MÉTRICAS DO SKYBAND
# ============================================================================

@dataclass
class CollinearityReport:
    """Resultado da detecção de multicolinearidade entre métricas do Skyband.

    Seção 4 do artigo PSLA4ML:
        "As métricas formam um conjunto de alta colinearidade, com coeficientes
         r > 0,99 em todos os pares.  O conjunto Skyline tende a colapsar para
         poucos pontos extremos quando critérios altamente correlacionados são
         combinados."

    Attributes:
        collinear_pairs: Lista de tuplas ``(metrica_a, metrica_b, r)`` para pares
            com ``|r| ≥ threshold``.  Ordenada por ``|r|`` decrescente.
        correlation_matrix: Matriz completa ``{m_a: {m_b: r}}`` de todos os pares.
        threshold: Limiar ``|r|`` utilizado (padrão 0.95).
        has_collinearity: ``True`` se ao menos um par excede o limiar.
        metrics: Lista de métricas analisadas.
        n_samples: Número de traces com status=success utilizados.
    """

    collinear_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    correlation_matrix: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    threshold: float = DEFAULT_COLLINEARITY_THRESHOLD
    has_collinearity: bool = False
    metrics: List[str] = field(default_factory=list)
    n_samples: int = 0


def _extract_metric_values(
    results: List[Dict[str, Any]],
    metric: str,
) -> List[float]:
    """Extrai valores de uma métrica de uma lista de resultados bem-sucedidos."""
    values = []
    for r in results:
        if r.get("status") != "success":
            continue
        resources = r.get("resources", {})
        raw = resources.get(metric)
        if raw is None:
            raw = r.get(metric)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (ValueError, TypeError):
            continue
    return values


def metric_correlation_matrix(
    results: List[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Calcula a matriz de correlação de Pearson entre métricas.

    Args:
        results: Lista de resultados de experimentos.
        metrics: Lista de nomes de métricas a correlacionar.

    Returns:
        Dicionário aninhado ``{m_a: {m_b: r}}`` com os coeficientes de Pearson.
        ``None`` quando há dados insuficientes para o par.
    """
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    metric_values: Dict[str, List[float]] = {
        m: _extract_metric_values(results, m) for m in metrics
    }

    for m_a in metrics:
        matrix[m_a] = {}
        for m_b in metrics:
            if m_a == m_b:
                matrix[m_a][m_b] = 1.0
                continue
            vals_a = metric_values[m_a]
            vals_b = metric_values[m_b]
            # Alinha pelos índices dos traces com ambos os valores
            paired = [
                (a, b)
                for a, b in zip(vals_a, vals_b)
            ]
            if len(paired) < 2:
                matrix[m_a][m_b] = None
                continue
            xs, ys = zip(*paired)
            matrix[m_a][m_b] = compute_correlation(list(xs), list(ys))

    return matrix


def detect_collinear_metrics(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    threshold: float = DEFAULT_COLLINEARITY_THRESHOLD,
) -> "CollinearityReport":
    """Detecta pares de métricas altamente correlacionadas no conjunto de traces.

    Implementa o diagnóstico de multicolinearidade descrito na Seção 4 do artigo
    PSLA4ML, onde os autores identificam que as métricas ``train_time_sec``,
    ``energy_kwh``, ``emissions_kg_co2`` e ``cost_usd`` possuem coeficientes de
    Pearson ``r > 0,99`` em ambientes de hardware fixo.

    Quando métricas altamente correlacionadas são combinadas na consulta k-Skyband,
    a fronteira de dominância tende a colapsar para poucos pontos extremos, pois
    minimizar qualquer métrica do grupo implica, de forma quase equivalente,
    minimizar as demais.

    Args:
        results:   Lista de resultados de experimentos.
        metrics:   Métricas a analisar.  Padrão: as 4 métricas de eficiência do
                   artigo (``train_time_sec``, ``energy_kwh``,
                   ``emissions_kg_co2``, ``cost_usd``).
        threshold: Limiar de ``|r|`` a partir do qual o par é considerado
                   colinear (padrão: 0.95).

    Returns:
        :class:`CollinearityReport` com pares colineares, matriz de correlação
        completa e flag ``has_collinearity``.

    Example::

        report = detect_collinear_metrics(results, threshold=0.95)
        if report.has_collinearity:
            for m_a, m_b, r in report.collinear_pairs:
                print(f"{m_a} ↔ {m_b}  r={r:.3f}")
    """
    if metrics is None:
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]

    valid = [r for r in results if r.get("status") == "success"]
    n_samples = len(valid)

    matrix = metric_correlation_matrix(valid, metrics)

    collinear_pairs: List[Tuple[str, str, float]] = []
    seen: set = set()

    for m_a in metrics:
        for m_b in metrics:
            if m_a >= m_b:
                continue
            key = (m_a, m_b)
            if key in seen:
                continue
            seen.add(key)
            r = matrix.get(m_a, {}).get(m_b)
            if r is not None and abs(r) >= threshold:
                collinear_pairs.append((m_a, m_b, r))

    collinear_pairs.sort(key=lambda t: -abs(t[2]))

    logger.debug(
        "detect_collinear_metrics: %d pares com |r|≥%.2f em %d amostras — métricas=%s",
        len(collinear_pairs), threshold, n_samples, metrics,
    )

    return CollinearityReport(
        collinear_pairs=collinear_pairs,
        correlation_matrix=matrix,
        threshold=threshold,
        has_collinearity=bool(collinear_pairs),
        metrics=list(metrics),
        n_samples=n_samples,
    )
