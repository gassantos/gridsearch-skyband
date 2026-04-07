"""
Skyband Query Engine - BERT-PLI  (Facade)
==========================================

.. deprecated:: 1.2.0
    Este modulo e mantido como facade para compatibilidade retroativa.
    Importe diretamente dos submodulos:
        - ``gridsearch.dominance``     — Pareto / Skyband
        - ``gridsearch.comparison``    — Skyband vs ranking escalar
        - ``gridsearch.visualization`` — graficos e relatorios textuais

Autor: Gustavo Alexandre
"""

# Re-exporta tudo que consumidores historicos importam de gridsearch.skyband

# --- dominance ---
from .dominance import (                                     # noqa: F401
    DEFAULT_METRICS,
    DEFAULT_MINIMIZE,
    _extract_metric_value,
    _build_vector,
    _load_metrics_config,
    sla_filter,
    dominates,
    pareto_front,
    skyband_query,
)

# --- comparison ---
from .comparison import compare_skyband_vs_ranking           # noqa: F401

# --- visualization ---
from .visualization import (                                 # noqa: F401
    plot_pareto_2d,
    skyband_report,
)
