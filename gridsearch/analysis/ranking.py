"""
Ranking de configurações por score ponderado
=============================================

Normalização min-max e scoring ponderado para ordenar
configurações de grid search por múltiplas métricas.

Autor: Gustavo Alexandre
"""

import logging
from typing import Any, Dict, List

from .metrics_config import _get_resource_metrics

logger = logging.getLogger(__name__)


def rank_configurations(
    results: List[Dict[str, Any]],
    metrics: List[str] | None = None,
    weights: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Cria ranking de configurações baseado em múltiplas métricas.

    Args:
        results: Lista de resultados
        metrics: Lista de nomes das métricas (padrão: ["train_time_sec", "energy_kwh"])
        weights: Pesos para cada métrica (padrão: igual para todas)

    Returns:
        Lista ordenada de configurações com scores
    """
    if metrics is None:
        metrics = ["train_time_sec", "energy_kwh"]

    if weights is None:
        weights = [1.0] * len(metrics)

    if len(metrics) != len(weights):
        raise ValueError("Número de métricas e pesos deve ser igual")

    logger.info(f"Criando ranking com métricas: {metrics}")

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        return []

    # Normaliza valores de cada métrica
    normalized_data = []

    resource_keys = {m["key"] for m in _get_resource_metrics()}

    for metric in metrics:
        values = []

        for result in successful:
            if metric in resource_keys:
                resources = result.get("resources", {})
                value = resources.get(metric)
            else:
                value = result.get(metric)

            if value is not None:
                values.append(float(value))
            else:
                values.append(None)

        # Normalização min-max (0 = melhor, 1 = pior)
        valid_values = [v for v in values if v is not None]

        if valid_values:
            min_val = min(valid_values)
            max_val = max(valid_values)
            range_val = max_val - min_val if max_val != min_val else 1.0

            normalized = []
            for v in values:
                if v is None:
                    normalized.append(1.0)  # Penalidade máxima
                else:
                    normalized.append((v - min_val) / range_val)
        else:
            normalized = [0.0] * len(values)

        normalized_data.append(normalized)

    # Calcula score ponderado
    ranked = []

    for i, result in enumerate(successful):
        score = 0.0

        for j, metric in enumerate(metrics):
            score += weights[j] * normalized_data[j][i]

        ranked.append(
            {
                "experiment_idx": result.get("grid_experiment_idx"),
                "params": result.get("grid_params"),
                "score": score,
                "metrics": {
                    metric: normalized_data[j][i] for j, metric in enumerate(metrics)
                },
            }
        )

    # Ordena por score (menor é melhor)
    ranked.sort(key=lambda x: x["score"])

    return ranked
