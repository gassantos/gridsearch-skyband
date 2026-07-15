"""
Grid Search Analysis — Pacote de análise (Facade)
===================================================

Re-exporta toda a API pública para manter compatibilidade retroativa
com ``from gridsearch.analysis import ...``.

Submódulos:
    - ``statistics``      — Estatísticas descritivas
    - ``correlations``    — Correlação de Pearson hp × métrica
    - ``ranking``         — Ranking ponderado de configurações
    - ``hyperparameters`` — Análise por hiperparâmetro individual
    - ``report``          — Geração de relatórios e exportação
    - ``metrics_config``  — Carregamento dinâmico de métricas (OCP)

Autor: Gustavo Alexandre
"""

from gridsearch.executor import GRID_OUTPUT_DIR

# Diretórios derivados da fonte canônica (executor.py)
GRID_RESULTS_FILE = GRID_OUTPUT_DIR / "grid_search_results.json"
ANALYSIS_DIR = GRID_OUTPUT_DIR / "analysis"

# --- statistics ---
from .statistics import (                    # noqa: F401, E402
    calculate_statistics,
    compute_descriptive_statistics,
)

# --- correlations ---
from .correlations import (                  # noqa: F401, E402
    compute_correlation,
    analyze_correlations,
    metric_correlation_matrix,
    detect_collinear_metrics,
    CollinearityReport,
    DEFAULT_COLLINEARITY_THRESHOLD,
)

# --- ranking ---
from .ranking import rank_configurations    # noqa: F401, E402

# --- hyperparameters ---
from .hyperparameters import (               # noqa: F401, E402
    analyze_by_hyperparameter,
    find_best_value_per_hyperparameter,
)

# --- report ---
from .report import (                        # noqa: F401, E402
    export_analysis_to_json,
    generate_analysis_report,
    main,
)

# --- metrics_config (API interna, exportada para reuso) ---
from .metrics_config import (                # noqa: F401, E402
    _get_resource_metrics,
    _load_resource_metrics,
)
