"""
Persistência de resultados de experimentos
============================================

Construção do dicionário JSON e escrita em CSV acumulado.

Autor: Gustavo Alexandre
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .helpers import METRICS_DIR

logger = logging.getLogger(__name__)


def build_result_dict(
    *,
    experiment_id: str,
    json_filename: str,
    seed: int,
    status: str,
    date_exec: str,
    start_iso: str,
    end_iso: str,
    device_type: str,
    device_name: str,
    precision: str,
    parallel_workers: int,
    train_dataset_name: str,
    optimizer: str,
    learning_rate: float,
    avg_gflops_per_batch: float,
    batch_size: int,
    epoch: int,
    exec_time: float,
    energy_kwh: Optional[float],
    emissions_kg: Optional[float],
    cost_usd: Optional[float],
    avg_ram: Optional[float],
    peak_ram: Optional[float],
    total_gflops: float,
    eval_metrics: Dict[str, Any],
    stdout: str,
    stderr: str,
    tpu_check: Any = None,
) -> Dict[str, Any]:
    """Constrói o dicionário padronizado de resultado de um experimento."""
    result: Dict[str, Any] = {
        "experiment": {
            "id": experiment_id,
            "config_name": json_filename,
            "seed": seed,
            "status": status,
            "date": date_exec,
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
        },
        "environment": {
            "device_type": device_type,
            "device_name": device_name,
            "precision": precision,
        },
        "execution": {
            "parallel_workers": parallel_workers,
            "train_dataset": train_dataset_name,
        },
        "hyperparameters": {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "avg_gflops_per_batch": avg_gflops_per_batch,
            "batch_size": batch_size,
            "epoch": epoch,
        },
        "resources": {
            "train_time_sec": f"{exec_time:.2f}",
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg,
            "cost_usd": cost_usd,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "total_gflops": total_gflops,
        },
        "evaluation": eval_metrics if eval_metrics else None,
        "logs": {
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
        },
    }

    # BL-08: inclui status de ativação do TPU quando disponível
    if tpu_check is not None:
        tpu_dict = (
            tpu_check.to_dict()
            if hasattr(tpu_check, "to_dict")
            else dict(tpu_check)
        )
        result["tpu_acceleration_check"] = tpu_dict
        if tpu_dict.get("warning"):
            result["warnings"] = result.get("warnings", []) + [tpu_dict["warning"]]

    return result


def write_json_result(result: Dict[str, Any], json_filename: str) -> Path:
    """Escreve o dicionário de resultado em arquivo JSON."""
    json_path = METRICS_DIR / json_filename
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return json_path


def append_csv_row(
    *,
    experiment_id: str,
    json_filename: str,
    seed: int,
    device_type: str,
    parallel_workers: int,
    train_dataset_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    epoch: int,
    exec_time: float,
    energy_kwh: Optional[float],
    emissions_kg: Optional[float],
    cost_usd: Optional[float],
    avg_ram: Optional[float],
    peak_ram: Optional[float],
    avg_gflops_per_batch: float,
    total_gflops: float,
    status: str,
    end_iso: str,
    eval_metrics: Dict[str, Any],
) -> Path:
    """Acrescenta uma linha no CSV acumulado de sumário."""
    csv_filename = (
        f"experiment_summary_{device_type}"
        f"{datetime.now().strftime('%Y%m%d')}.csv"
    )
    csv_path = METRICS_DIR / csv_filename
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "experiment_id",
                "config_name",
                "seed",
                "device_type",
                "parallel_workers",
                "train_dataset",
                "optimizer",
                "learning_rate",
                "batch_size",
                "epoch",
                "train_time_sec",
                "energy_kwh",
                "emissions_kg",
                "cost_usd",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_gflops_per_batch",
                "total_gflops",
                "status",
                "timestamp",
                "eval_precision",
                "eval_recall",
                "eval_f1",
                "eval_accuracy",
                "eval_source",
            ])

        writer.writerow([
            experiment_id,
            json_filename,
            seed,
            device_type,
            parallel_workers,
            train_dataset_name,
            optimizer,
            learning_rate,
            batch_size,
            epoch,
            f"{exec_time:.2f}",
            energy_kwh,
            emissions_kg,
            f"{cost_usd:.6f}" if cost_usd is not None else None,
            avg_ram,
            peak_ram,
            avg_gflops_per_batch,
            total_gflops,
            status,
            end_iso,
            f"{eval_metrics['precision']:.4f}" if eval_metrics else None,
            f"{eval_metrics['recall']:.4f}" if eval_metrics else None,
            f"{eval_metrics['f1_score']:.4f}" if eval_metrics else None,
            f"{eval_metrics['accuracy']:.4f}" if eval_metrics and 'accuracy' in eval_metrics else None,
            eval_metrics.get("source") if eval_metrics else None,
        ])

    return csv_path
