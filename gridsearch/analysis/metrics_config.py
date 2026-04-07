"""
Configuração dinâmica de métricas de recurso (OCP)
===================================================

Carrega a lista de métricas de recurso a partir de ``metrics.json``.
Novas métricas podem ser adicionadas editando o JSON, sem alterar
código (Open-Closed Principle).

Autor: Gustavo Alexandre
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_METRICS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "metrics.json"


def _load_resource_metrics() -> List[Dict[str, str]]:
    """Carrega a lista de métricas de recurso de ``metrics.json`` (OCP).

    Cada entrada contém ``key`` (chave em ``resources``) e ``label``
    opcional (chave de saída nas estatísticas). Quando ``label`` está
    ausente, ``key`` é usado como label.

    Returns:
        Lista de dicts ``[{"key": ..., "label": ...}, ...]``.
    """
    with open(_METRICS_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("resource_metrics", [])
    for entry in entries:
        entry.setdefault("label", entry["key"])
    return entries


# Cache module-level para evitar I/O repetido
_RESOURCE_METRICS: List[Dict[str, str]] | None = None


def _get_resource_metrics() -> List[Dict[str, str]]:
    """Retorna métricas de recurso com cache em memória."""
    global _RESOURCE_METRICS
    if _RESOURCE_METRICS is None:
        _RESOURCE_METRICS = _load_resource_metrics()
    return _RESOURCE_METRICS
