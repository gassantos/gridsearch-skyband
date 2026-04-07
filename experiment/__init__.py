"""
Experiment — Pacote de execução de experimentos (Facade)
=========================================================

Re-exporta ``execute_experiment`` para manter compatibilidade
com ``from run_experiment import execute_experiment`` via a
facade ``run_experiment.py`` na raiz.

Submódulos:
    - ``runner``      — Motor de execução (orquestrador)
    - ``helpers``     — Utilitários (TeeStream, load_config, etc.)
    - ``evaluation``  — Extração de métricas de avaliação
    - ``persistence`` — Persistência JSON e CSV

Autor: Gustavo Alexandre
"""

from .runner import execute_experiment      # noqa: F401
from .helpers import (                      # noqa: F401
    ENERGY_COST_USD_PER_KWH,
    METRICS_DIR,
    TeeStream,
    estimate_bert_flops,
    load_config,
    now_iso,
)
