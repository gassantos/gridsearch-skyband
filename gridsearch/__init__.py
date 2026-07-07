"""
Grid Search Module - BERT-PLI
==============================

Módulo completo para busca em grade de hiperparâmetros com recursos avançados:
- Execução paralela configurável
- Salvamento incremental de estado
- Análise estatística de resultados
- Recuperação automática de falhas
- Métricas de recursos computacionais

Componentes principais:
- core: Execução do grid search
- analysis: Análise e visualização de resultados
- config: Configurações de espaço de busca
- scripts: Scripts auxiliares de execução

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

__version__ = "1.2.0"
__author__ = "BERT-PLI Team"

# --- grid (geração de combinações) ---
from .grid import (
    generate_parameter_grid,
    create_config_for_combination,
)

# --- executor (orquestração) ---
from .executor import (
    run_grid_search,
    run_single_experiment,
)

# --- reporting (análise + relatórios + CLI) ---
from .reporting import (
    analyze_results,
    generate_summary_report,
)

# --- dominance (Pareto / Skyband) ---
from .dominance import (
    sla_filter,
    dominates,
    pareto_front,
    skyband_query,
    DEFAULT_METRICS,
    DEFAULT_MINIMIZE,
)

# --- comparison ---
from .comparison import compare_skyband_vs_ranking

# --- visualization ---
from .visualization import (
    plot_pareto_2d,
    skyband_report,
)

# --- tiers (discretização PSLA4ML — Algoritmo 1, Passos 4–10) ---
from .tiers import (
    compute_thresholds,
    discretize_metrics,
    DISCRETIZATION_STRATEGIES,
    Tier,
    generate_psla4ml,
)

__all__ = [
    # --- grid ---
    'generate_parameter_grid',
    'create_config_for_combination',
    # --- executor ---
    'run_grid_search',
    'run_single_experiment',
    # --- reporting ---
    'analyze_results',
    'generate_summary_report',
    # --- dominance ---
    'sla_filter',
    'dominates',
    'pareto_front',
    'skyband_query',
    'DEFAULT_METRICS',
    'DEFAULT_MINIMIZE',
    # --- comparison ---
    'compare_skyband_vs_ranking',
    # --- visualization ---
    'plot_pareto_2d',
    'skyband_report',
    # --- tiers ---
    'compute_thresholds',
    'discretize_metrics',
    'DISCRETIZATION_STRATEGIES',
    'Tier',
    'generate_psla4ml',
]
