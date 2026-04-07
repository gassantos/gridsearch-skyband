"""
Estatísticas descritivas de resultados
=======================================

Funções puras para cálculo de estatísticas descritivas sobre os
resultados de experimentos do grid search.

Autor: Gustavo Alexandre
"""

import logging
import statistics as _stats
from typing import Any, Dict, List

from .metrics_config import _get_resource_metrics

logger = logging.getLogger(__name__)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calcula estatísticas descritivas de uma lista de valores.

    Args:
        values: Lista de valores numéricos

    Returns:
        Dicionário com estatísticas (média, mediana, desvio padrão, etc.)
    """
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "stdev": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0.0,
        }

    return {
        "mean": _stats.mean(values),
        "median": _stats.median(values),
        "stdev": _stats.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": float(len(values)),
    }


def compute_descriptive_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computa estatísticas descritivas para todas as métricas de recurso.

    As métricas são carregadas dinamicamente de ``metrics.json``
    (seção ``resource_metrics``). Novas métricas podem ser adicionadas
    editando o JSON, sem alterar código (OCP).

    Args:
        results: Lista de resultados dos experimentos

    Returns:
        Dicionário com estatísticas para cada métrica
    """
    logger.info("Computando estatísticas descritivas...")

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        logger.warning("Nenhum resultado válido para análise")
        return {}

    resource_metrics = _get_resource_metrics()

    stats: Dict[str, Any] = {}

    for metric in resource_metrics:
        key = metric["key"]
        label = metric["label"]

        values = []
        for result in successful:
            resources = result.get("resources", {})
            val = resources.get(key)
            if val is not None:
                values.append(float(val))

        stats[label] = calculate_statistics(values)

    stats["total_experiments"] = len(results)
    stats["successful_experiments"] = len(successful)
    stats["failed_experiments"] = len(results) - len(successful)

    return stats
