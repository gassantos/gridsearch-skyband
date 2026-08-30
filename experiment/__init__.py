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
    - ``task_cache``   — Cache de tarefas para evitar recomputação desnecessária
    - ``task_executor`` — Execução de tarefas dentro de workflows experimentais
    - ``workflow_planner`` — Planejador de execução de workflows experimentais
    - ``workflow``      — Definições e execução de workflows experimentais
    - ``bertpli_workflow`` — Workflow de referência BERT-PLI
    - ``workflow_reporting`` — Relatorios textuais e timeline estruturada de workflows

Autor: Gustavo Alexandre
"""

from .aggregation import (  # noqa: F401
    MetricAggregation,
    MetricAggregationPolicy,
    aggregate_evaluation_metrics,
    aggregate_workflow_run,
)
from .bertpli_workflow import (  # noqa: F401
    BertPliWorkflowConfig,
    build_bertpli_task_functions,
    build_bertpli_workflow,
)
from .estimation import estimate_workflow_resources  # noqa: F401
from .helpers import (  # noqa: F401
    ENERGY_COST_USD_PER_KWH,
    METRICS_DIR,
    TeeStream,
    estimate_bert_flops,
    load_config,
    now_iso,
)
from .persistence import load_workflow_run  # noqa: F401
from .runner import execute_experiment  # noqa: F401
from .task_cache import TaskCache  # noqa: F401
from .task_executor import (  # noqa: F401
    ParallelWorkflowExecutor,
    SequentialWorkflowExecutor,
)
from .workflow import (  # noqa: F401
    ExperimentDefinition,
    ExperimentRun,
    RetryPolicy,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)
from .workflow_planner import WorkflowPlanner  # noqa: F401
from .workflow_reporting import workflow_report, workflow_timeline  # noqa: F401
