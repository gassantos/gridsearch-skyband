"""
Experiment — Pacote de execução de experimentos (Facade)
=========================================================

Pacote canônico para execução de experimentos BERT-PLI.
Uso: ``from experiment import execute_experiment``

CLI standalone: ``python -m experiment <config_path> [gpu_id ...]``

Submódulos:
    - ``runner``      — Motor de execução (orquestrador)
    - ``helpers``     — Utilitários (TeeStream, load_config, etc.)
    - ``evaluation``  — Extração de métricas de avaliação
    - ``persistence`` — Persistência JSON e CSV

Autor: Gustavo Alexandre
"""

from .helpers import (  # noqa: F401
    ENERGY_COST_USD_PER_KWH,
    METRICS_DIR,
    TeeStream,
    estimate_bert_flops,
    load_config,
    now_iso,
)
from .runner import execute_experiment  # noqa: F401
from .workflow import (  # noqa: F401
    ExperimentDefinition,
    ExperimentRun,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)
