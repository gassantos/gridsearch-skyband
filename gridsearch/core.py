"""
Grid Search Core - BERT-PLI  (Facade)
=======================================

.. deprecated:: 1.2.0
    Este modulo e mantido como facade para compatibilidade retroativa.
    Importe diretamente dos submodulos:
        - ``gridsearch.grid``          — geracao de grade
        - ``gridsearch.executor``      — orquestracao
        - ``gridsearch.sla_prefilter`` — pre-filtro SLA
        - ``gridsearch.reporting``     — analise e relatorios

Autor: Gustavo Alexandre
"""

# Re-exporta tudo que consumidores historicos importam de gridsearch.core

# --- grid ---
from .grid import (                                          # noqa: F401
    generate_parameter_grid,
    create_config_for_combination,
    _load_param_mapping,
)

# --- executor ---
from .executor import (                                      # noqa: F401
    run_grid_search,
    run_single_experiment,
    save_state,
    GRID_OUTPUT_DIR,
    GRID_CONFIGS_DIR,
    ENERGY_COST_USD_PER_KWH,
    _LOGFILE,
    _TDATE,
    _get_device_type,
    _resolve_output_dir,
    _grid_state_file,
    _grid_results_file,
    _grid_summary_file,
)

# --- sla_prefilter ---
from .sla_prefilter import (                                 # noqa: F401
    prefilter_param_grid_by_execution_sla as _prefilter_param_grid_by_execution_sla,
    SUPPORTED_EXECUTION_SLA_CONSTRAINTS,
    MAX_SLA_REJECTED_SAMPLES,
    _estimate_train_time_sec,
    _resolve_train_time_baseline_sec,
    _safe_float,
    _first_failing_execution_constraint,
    _log_sla_prefilter_summary,
)

# --- reporting ---
from .reporting import (                                     # noqa: F401
    analyze_results,
    generate_summary_report,
    main,
)
