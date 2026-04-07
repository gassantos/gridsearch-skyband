"""
Análise por hiperparâmetro
===========================

Avalia o impacto de cada hiperparâmetro individual sobre as métricas
de recurso, e identifica o melhor valor para cada um.

Autor: Gustavo Alexandre
"""

import logging
from typing import Any, Dict, List

from .metrics_config import _get_resource_metrics
from .statistics import calculate_statistics

logger = logging.getLogger(__name__)


def analyze_by_hyperparameter(
    results: List[Dict[str, Any]], param_name: str, metric_name: str = "train_time_sec"
) -> Dict[Any, Dict[str, float]]:
    """
    Analisa o impacto de um hiperparâmetro específico em uma métrica.

    Args:
        results: Lista de resultados
        param_name: Nome do hiperparâmetro a analisar
        metric_name: Nome da métrica a medir (padrão: train_time_sec)

    Returns:
        Dicionário mapeando valores do hiperparâmetro para estatísticas
    """
    logger.info(f"Analisando impacto de '{param_name}' em '{metric_name}'...")

    successful = [r for r in results if r.get("status") == "success"]

    grouped = {}

    resource_keys = {m["key"] for m in _get_resource_metrics()}

    for result in successful:
        params = result.get("grid_params", {})

        if param_name not in params:
            continue

        param_value = params[param_name]

        if metric_name in resource_keys:
            resources = result.get("resources", {})
            metric_value = resources.get(metric_name)
        else:
            metric_value = result.get(metric_name)

        if metric_value is None:
            continue

        if param_value not in grouped:
            grouped[param_value] = []

        grouped[param_value].append(float(metric_value))

    analysis = {}
    for param_value, metric_values in grouped.items():
        analysis[param_value] = calculate_statistics(metric_values)

    return analysis


def find_best_value_per_hyperparameter(
    results: List[Dict[str, Any]],
    metric_name: str = "train_time_sec",
    minimize: bool = True,
) -> Dict[str, Any]:
    """
    Identifica o melhor valor para cada hiperparâmetro individualmente.

    Args:
        results: Lista de resultados
        metric_name: Métrica a otimizar
        minimize: Se True, procura menor valor; se False, maior

    Returns:
        Dicionário com melhor valor de cada hiperparâmetro
    """
    logger.info("Identificando melhores valores para cada hiperparâmetro...")

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        return {}

    all_params = set()
    for result in successful:
        params = result.get("grid_params", {})
        all_params.update(params.keys())

    best_values = {}

    for param_name in all_params:
        analysis = analyze_by_hyperparameter(results, param_name, metric_name)

        if not analysis:
            continue

        best_value = None
        best_mean = None

        for param_value, stats in analysis.items():
            mean = stats.get("mean")

            if mean is None:
                continue

            if best_mean is None:
                best_value = param_value
                best_mean = mean
            elif minimize and mean < best_mean:
                best_value = param_value
                best_mean = mean
            elif not minimize and mean > best_mean:
                best_value = param_value
                best_mean = mean

        best_values[param_name] = {
            "best_value": best_value,
            "mean_metric": best_mean,
            "all_values": analysis,
        }

    return best_values
