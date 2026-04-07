"""
Extração de métricas de avaliação
==================================

Funções para extrair métricas de avaliação (precision, recall, F1, accuracy)
a partir do stdout capturado (pool_out) ou de subprocesso de teste.

Autor: Gustavo Alexandre
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Padrão de regex para extrair métricas do log de validação do train_tool
_VALID_RE = re.compile(
    r"valid set: micro_prec_query=([\d.]+),\s*micro_recall_query=([\d.]+),\s*micro_f1_query=([\d.]+),\s*accuracy=([\d.]+)"
)


def extract_eval_metrics(
    status: str,
    output_lines: List[str],
    cfg: Any,
    config_path: str,
    convert_test_results_fn: Callable,
    compute_metrics_fn: Callable,
) -> Dict[str, Any]:
    """Extrai métricas de avaliação do experimento.

    Suporta dois modos:
    - ``pool_out=True``: extrai métricas da última linha de validação no stdout
    - ``run_test_at_end=True``: executa subprocesso de teste e computa métricas

    Args:
        status: Status do treinamento ("success" ou "failed").
        output_lines: Linhas de stdout capturadas durante o treino.
        cfg: ConfigParser com a configuração do experimento.
        config_path: Caminho do arquivo de configuração.
        convert_test_results_fn: Callable para converter resultados de teste.
        compute_metrics_fn: Callable para computar métricas.

    Returns:
        Dicionário com métricas de avaliação ou dict vazio.
    """
    eval_metrics: Dict[str, Any] = {}

    if status != "success":
        return eval_metrics

    try:
        run_test_at_end = cfg.getboolean("eval", "run_test_at_end", fallback=False)
        pool_out_mode = cfg.getboolean("output", "pool_out", fallback=False)

        if pool_out_mode:
            eval_metrics = _extract_from_pool_out(output_lines)
        elif run_test_at_end:
            eval_metrics = _extract_from_test_subprocess(
                cfg, config_path, convert_test_results_fn, compute_metrics_fn
            )
    except Exception as exc:
        logger.warning("Erro ao calcular métricas de avaliação: %s", exc)

    return eval_metrics


def _extract_from_pool_out(output_lines: List[str]) -> Dict[str, Any]:
    """Extrai métricas da última ocorrência do padrão de validação no stdout."""
    last_match = None
    for line in output_lines:
        m = _VALID_RE.search(line)
        if m:
            last_match = m

    if last_match:
        metrics = {
            "precision": float(last_match.group(1)),
            "recall": float(last_match.group(2)),
            "f1_score": float(last_match.group(3)),
            "accuracy": float(last_match.group(4)),
            "source": "validation_log",
        }
        logger.info(
            "Métricas (validação final, pool_out): P=%.4f  R=%.4f  F1=%.4f  Acc=%.4f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["accuracy"],
        )
        return metrics

    logger.warning(
        "pool_out=True mas nenhuma linha 'valid set:' encontrada no stdout. "
        "Métricas de avaliação indisponíveis."
    )
    return {}


def _extract_from_test_subprocess(
    cfg: Any,
    config_path: str,
    convert_test_results_fn: Callable,
    compute_metrics_fn: Callable,
) -> Dict[str, Any]:
    """Executa subprocesso de teste e computa métricas."""
    model_out_path = (
        Path(cfg.get("output", "model_path")) / cfg.get("output", "model_name")
    )
    labels_path = cfg.get(
        "data", "test_labels_file", fallback="data/task1_test_labels_2024.json"
    )
    test_result_path = model_out_path / "test_results.json"

    # Localiza o último checkpoint salvo (por número de época)
    last_epoch = cfg.getint("train", "epoch") - 1
    checkpoint_path = model_out_path / f"{last_epoch}.pkl"
    if not checkpoint_path.exists():
        pkl_files = sorted(
            model_out_path.glob("*.pkl"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
        )
        checkpoint_path = pkl_files[-1] if pkl_files else None

    if not (checkpoint_path and checkpoint_path.exists() and Path(labels_path).exists()):
        logger.warning(
            "Checkpoint (%s) ou labels (%s) não encontrado. "
            "Métricas de avaliação indisponíveis.",
            checkpoint_path,
            labels_path,
        )
        return {}

    logger.info("Executando avaliação com checkpoint: %s", checkpoint_path)
    test_proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/test.py",
            "-c", config_path,
            "-g", "0",
            "--checkpoint", str(checkpoint_path),
            "--result", str(test_result_path),
        ],
        capture_output=True,
        text=True,
    )

    if test_proc.returncode != 0 or not test_result_path.exists():
        logger.warning(
            "Subprocess de teste falhou (código %d). "
            "Métricas de avaliação indisponíveis.\n%s",
            test_proc.returncode,
            test_proc.stderr[-500:],
        )
        return {}

    task1_predicted = convert_test_results_fn(str(test_result_path))
    task1_path = model_out_path / "test_results_task1.json"
    with open(task1_path, "w") as f:
        json.dump(task1_predicted, f)

    eval_metrics = compute_metrics_fn(labels_path, str(task1_path))
    logger.info(
        "Métricas de avaliação: P=%.4f  R=%.4f  F1=%.4f  Acc=%.4f",
        eval_metrics["precision"],
        eval_metrics["recall"],
        eval_metrics["f1_score"],
        eval_metrics.get("accuracy", 0.0),
    )
    return eval_metrics
