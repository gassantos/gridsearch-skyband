"""
Skyband Query Engine - BERT-PLI  (Facade)
==========================================

.. deprecated:: 1.2.0
    Este modulo e mantido como facade para compatibilidade retroativa.
    Importe diretamente dos submodulos:
        - ``gridsearch.dominance``     — Pareto / Skyband
        - ``gridsearch.comparison``    — Skyband vs ranking escalar
        - ``gridsearch.visualization`` — graficos e relatorios textuais
        - ``gridsearch.tiers``         — discretização PSLA4ML (Algoritmo 1, Passo 9)

Autor: Gustavo Alexandre
"""

# Re-exporta tudo que consumidores historicos importam de gridsearch.skyband

# --- dominance ---
from .dominance import (                                     # noqa: F401
    DEFAULT_METRICS,
    DEFAULT_MINIMIZE,
    QUALITY_METRICS,
    QUALITY_MINIMIZE,
    _get_minimize_flag,
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
from .workflow_skyband import (                              # noqa: F401
    task_skyband_query,
    task_to_skyband_point,
    workflow_skyband_query,
    workflow_to_skyband_point,
)

# --- visualization ---
from .visualization import (                                 # noqa: F401
    plot_pareto_2d,
    skyband_report,
)

# --- tiers (BL-02/03/04: discretização, Tier, generate_psla4ml, TrainingTemplate) ---
from .tiers import (                                         # noqa: F401
    compute_thresholds,
    discretize_metrics,
    DISCRETIZATION_STRATEGIES,
    Tier,
    generate_psla4ml,
    TrainingTemplate,
    filter_by_template,
)
