"""Workflow de referência BERT-PLI decomposto em tarefas rastreáveis."""

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
from .workflow import ExperimentDefinition, TaskDefinition

CommandRunner = Callable[[list[str]], None]


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
    """Cria o DAG BERT-PLI com sete tarefas e seus artefatos declarados."""
    return ExperimentDefinition(
        name="bertpli-reference-workflow",
        experiment_type="nlp",
        tasks=(
            TaskDefinition("fine_tune_bert", "Fine-tuning BERT", config={"config": config.bert_config}),
            TaskDefinition(
                "poolout", "Extração de interações", depends_on=("fine_tune_bert",),
                config={"config": config.poolout_config, "checkpoint": config.bert_checkpoint},
                input_signatures={"bert_checkpoint": config.bert_checkpoint},
            ),
            TaskDefinition(
                "convert_poolout_train", "Conversão pool-out de treino", depends_on=("poolout",),
                config={"input": config.train_input, "result": config.train_poolout},
                input_signatures={"poolout": config.poolout_result},
            ),
            TaskDefinition(
                "convert_poolout_valid", "Conversão pool-out de validação", depends_on=("poolout",),
                config={"input": config.valid_input, "result": config.valid_poolout},
                input_signatures={"poolout": config.poolout_result},
            ),
            TaskDefinition(
                "train_attention_rnn", "Treino Attention-RNN",
                depends_on=("convert_poolout_train", "convert_poolout_valid"),
                config={"config": config.rnn_config},
                input_signatures={"train": config.train_poolout, "valid": config.valid_poolout},
            ),
            TaskDefinition(
                "test_attention_rnn", "Inferência Attention-RNN", depends_on=("train_attention_rnn",),
                config={"config": config.rnn_config, "checkpoint": config.rnn_checkpoint},
                input_signatures={"rnn_checkpoint": config.rnn_checkpoint},
            ),
            TaskDefinition(
                "evaluate_retrieval", "Avaliação de recuperação", depends_on=("test_attention_rnn",),
                config={"labels": config.labels_file, "result": config.metrics_result},
                input_signatures={"predictions": config.test_result},
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