"""
Análise de correlações hiperparâmetro × métrica
=================================================

Correlação de Pearson entre hiperparâmetros numéricos e métricas
de recurso dos experimentos.

Autor: Gustavo Alexandre
"""

import logging
import statistics as _stats
from typing import Any, Dict, List, Optional

from .metrics_config import _get_resource_metrics

logger = logging.getLogger(__name__)


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
