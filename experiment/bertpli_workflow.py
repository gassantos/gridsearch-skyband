"""Workflow de referência BERT-PLI decomposto em tarefas rastreáveis.

Notação descrita no documento de especificação do workflow:
T0 Ingestão e preparação 
T1 Pré-treinamento 
T2 Adaptação 
T3 Indexação 
T4 Recuperação 
T5 Avaliação e monitoração

https://drive.google.com/file/d/1m8co8_Ozwn_drbEp9Ki0Snm9YnzuEgun/view
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.eval_tool import compute_metrics, parse_gru_results

from .helpers import load_config
from .workflow import (
    ArtifactDefinition,
    ArtifactKind,
    ExecutionRegime,
    ExperimentDefinition,
    ResourceRequirements,
    TaskActivity,
    TaskDefinition,
)

CommandRunner = Callable[[list[str]], None]


def _gpu_count(gpu: str | None) -> int:
    """Deriva o numero de GPUs a partir da string ``--gpu`` (ex.: ``"0,1"``)."""
    return len(gpu.split(",")) if gpu else 1


@dataclass(frozen=True)
class BertPliWorkflowConfig:
    """Caminhos e parâmetros dos artefatos usados pelo workflow BERT-PLI."""

    bert_config: str = "config/nlp/BertPoint.config"
    poolout_config: str = "config/nlp/BertPoolOutMax.config"
    rnn_config: str = "config/nlp/AttenLSTM.config"
    bert_checkpoint: str = "output/checkpoints/bert_finetuned/2.pkl"
    rnn_checkpoint: str = "output/checkpoints/attenlstm/59.pkl"
    poolout_result: str = "output/results/poolout.json"
    train_input: str = "data/test_paragraphs_processed_data.json"
    valid_input: str = "data/test_paragraphs_processed_data.json"
    train_poolout: str = "output/results/train_poolout.json"
    valid_poolout: str = "output/results/valid_poolout.json"
    test_result: str = "output/results/lstm_results.json"
    parsed_result: str = "output/results/lstm_parsed_result.json"
    metrics_result: str = "output/results/metrics.json"
    labels_file: str = "data/task1_test_labels_2024.json"
    gpu: str | None = None


def build_bertpli_workflow(config: BertPliWorkflowConfig) -> ExperimentDefinition:
    """Cria o DAG BERT-PLI com sete tarefas e seus artefatos declarados.

    As tarefas sao classificadas conforme as atividades T0-T5 do template de
    workflows de modelos de linguagem: ``fine_tune_bert`` e
    ``train_attention_rnn`` atualizam pesos (ADAPTATION, alto coup_t);
    ``poolout`` e as conversoes preparam dados/features sem atualizar theta
    (INGESTION); ``test_attention_rnn`` e ``evaluate_retrieval`` fecham o
    ciclo de avaliacao (EVALUATION_MONITORING). Todas operam em regime BUILD
    (lote), pois o workflow nao expoe um subgrafo de servico.
    """
    gpu_count = _gpu_count(config.gpu)
    train_data = ArtifactDefinition(
        artifact_id="bertpli-train-data",
        kind=ArtifactKind.DATA,
        version="input",
        uri=config.train_input,
    )
    valid_data = ArtifactDefinition(
        artifact_id="bertpli-valid-data",
        kind=ArtifactKind.DATA,
        version="input",
        uri=config.valid_input,
    )
    labels = ArtifactDefinition(
        artifact_id="bertpli-test-labels",
        kind=ArtifactKind.DATA,
        version="input",
        uri=config.labels_file,
    )
    bert_model = ArtifactDefinition(
        artifact_id="bertpli-bert-model",
        kind=ArtifactKind.MODEL,
        version="fine-tuned",
        uri=config.bert_checkpoint,
    )
    poolout_features = ArtifactDefinition(
        artifact_id="bertpli-poolout-features",
        kind=ArtifactKind.DATA,
        version="generated",
        uri=config.poolout_result,
    )
    train_features = ArtifactDefinition(
        artifact_id="bertpli-train-features",
        kind=ArtifactKind.DATA,
        version="generated",
        uri=config.train_poolout,
    )
    valid_features = ArtifactDefinition(
        artifact_id="bertpli-valid-features",
        kind=ArtifactKind.DATA,
        version="generated",
        uri=config.valid_poolout,
    )
    rnn_model = ArtifactDefinition(
        artifact_id="bertpli-rnn-model",
        kind=ArtifactKind.MODEL,
        version="trained",
        uri=config.rnn_checkpoint,
    )
    predictions = ArtifactDefinition(
        artifact_id="bertpli-predictions",
        kind=ArtifactKind.INTERACTION,
        version="generated",
        uri=config.test_result,
    )
    metrics = ArtifactDefinition(
        artifact_id="bertpli-evaluation-metrics",
        kind=ArtifactKind.DATA,
        version="generated",
        uri=config.metrics_result,
    )
    return ExperimentDefinition(
        name="bertpli-reference-workflow",
        experiment_type="nlp",
        tasks=(
            TaskDefinition(
                "fine_tune_bert", "Fine-tuning BERT", config={"config": config.bert_config},
                inputs=(train_data,), outputs=(bert_model,),
                activity=TaskActivity.ADAPTATION, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(gpu_count=gpu_count, coupling_degree=0.9),
            ),
            TaskDefinition(
                "poolout", "Extração de interações", depends_on=("fine_tune_bert",),
                config={"config": config.poolout_config, "checkpoint": config.bert_checkpoint},
                input_signatures={"bert_checkpoint": config.bert_checkpoint},
                inputs=(bert_model,), outputs=(poolout_features,),
                activity=TaskActivity.INGESTION, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(gpu_count=gpu_count, coupling_degree=0.2),
            ),
            TaskDefinition(
                "convert_poolout_train", "Conversão pool-out de treino", depends_on=("poolout",),
                config={"input": config.train_input, "result": config.train_poolout},
                input_signatures={"poolout": config.poolout_result},
                inputs=(train_data, poolout_features), outputs=(train_features,),
                activity=TaskActivity.INGESTION, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(coupling_degree=0.0),
            ),
            TaskDefinition(
                "convert_poolout_valid", "Conversão pool-out de validação", depends_on=("poolout",),
                config={"input": config.valid_input, "result": config.valid_poolout},
                input_signatures={"poolout": config.poolout_result},
                inputs=(valid_data, poolout_features), outputs=(valid_features,),
                activity=TaskActivity.INGESTION, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(coupling_degree=0.0),
            ),
            TaskDefinition(
                "train_attention_rnn", "Treino Attention-RNN",
                depends_on=("convert_poolout_train", "convert_poolout_valid"),
                config={"config": config.rnn_config},
                input_signatures={"train": config.train_poolout, "valid": config.valid_poolout},
                inputs=(train_features, valid_features), outputs=(rnn_model,),
                activity=TaskActivity.ADAPTATION, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(gpu_count=gpu_count, coupling_degree=0.9),
            ),
            TaskDefinition(
                "test_attention_rnn", "Inferência Attention-RNN", depends_on=("train_attention_rnn",),
                config={"config": config.rnn_config, "checkpoint": config.rnn_checkpoint},
                input_signatures={"rnn_checkpoint": config.rnn_checkpoint},
                inputs=(rnn_model,), outputs=(predictions,),
                activity=TaskActivity.EVALUATION_MONITORING, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(gpu_count=gpu_count, coupling_degree=0.1),
            ),
            TaskDefinition(
                "evaluate_retrieval", "Avaliação de recuperação", depends_on=("test_attention_rnn",),
                config={"labels": config.labels_file, "result": config.metrics_result},
                input_signatures={"predictions": config.test_result},
                inputs=(labels, predictions), outputs=(metrics,),
                activity=TaskActivity.EVALUATION_MONITORING, regime=ExecutionRegime.BUILD,
                resources=ResourceRequirements(coupling_degree=0.0),
            ),
        ),
    )


def build_bertpli_task_functions(
    config: BertPliWorkflowConfig,
    *,
    command_runner: CommandRunner | None = None,
) -> Mapping[str, Callable[[], dict[str, Any]]]:
    """Retorna adaptadores de tarefas que invocam os CLIs BERT-PLI existentes."""
    run = command_runner or _run_command
    gpu_args = ["--gpu", config.gpu] if config.gpu else []

    def fine_tune() -> dict[str, Any]:
        run([*_python_module("scripts.train"), "--config", config.bert_config, *gpu_args])
        return {"metrics": {"resources": _profiling_metrics(config.bert_config)},
            "artifacts": {"bert_checkpoint": config.bert_checkpoint}}

    def poolout() -> dict[str, Any]:
        run([
            *_python_module("scripts.poolout"), "--config", config.poolout_config,
            "--checkpoint", config.bert_checkpoint, "--result", config.poolout_result, *gpu_args,
        ])
        return {"artifacts": {"poolout": config.poolout_result}}

    def convert(source: str, target: str) -> Callable[[], dict[str, Any]]:
        def task() -> dict[str, Any]:
            run([
                *_python_module("scripts.poolout_to_train"), "--paras-file", source,
                "--poolout-file", config.poolout_result, "--result", target,
            ])
            return {"artifacts": {"dataset": target}}
        return task

    def train_rnn() -> dict[str, Any]:
        run([*_python_module("scripts.train"), "--config", config.rnn_config, *gpu_args])
        return {"metrics": {"resources": _profiling_metrics(config.rnn_config)},
            "artifacts": {"rnn_checkpoint": config.rnn_checkpoint}}

    def test_rnn() -> dict[str, Any]:
        run([
            *_python_module("scripts.test"), "--config", config.rnn_config,
            "--checkpoint", config.rnn_checkpoint, "--result", config.test_result, *gpu_args,
        ])
        return {"artifacts": {"predictions": config.test_result}}

    def evaluate() -> dict[str, Any]:
        parse_gru_results(config.test_result, config.parsed_result)
        metrics = compute_metrics(config.labels_file, config.parsed_result)
        Path(config.metrics_result).parent.mkdir(parents=True, exist_ok=True)
        Path(config.metrics_result).write_text(_json(metrics), encoding="utf-8")
        return {"metrics": {"evaluation": metrics}, "artifacts": {"metrics": config.metrics_result}}

    return {
        "fine_tune_bert": fine_tune,
        "poolout": poolout,
        "convert_poolout_train": convert(config.train_input, config.train_poolout),
        "convert_poolout_valid": convert(config.valid_input, config.valid_poolout),
        "train_attention_rnn": train_rnn,
        "test_attention_rnn": test_rnn,
        "evaluate_retrieval": evaluate,
    }


def _python_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2)


def _profiling_metrics(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    profile_path = Path(config.get("output", "model_path")) / config.get("output", "model_name") / "profiling_metrics.json"
    if not profile_path.exists():
        return {}
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        return {
            name: float(data[name])
            for name in ("total_gflops", "avg_gflops_per_batch")
            if name in data
        }
    except (OSError, ValueError, TypeError):
        return {}