"""Valida evidências de uma execução real do BL-08 em TPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment.helpers import METRICS_DIR


def validate_tpu_result(result: dict[str, Any], expected_cores: int = 8) -> list[str]:
    errors: list[str] = []
    experiment = result.get("experiment", {})
    environment = result.get("environment", {})
    execution = result.get("execution", {})
    tpu_check = result.get("tpu_acceleration_check", {})
    runtime_metrics = tpu_check.get("xla_runtime_metrics", {})

    if experiment.get("status") != "success":
        errors.append("o experimento não terminou com status=success")
    if environment.get("device_type") != "TPU":
        errors.append("environment.device_type não é TPU")
    if execution.get("xla_world_size") != expected_cores:
        errors.append(
            f"execution.xla_world_size deve ser {expected_cores}, "
            f"mas foi {execution.get('xla_world_size')}"
        )
    if not tpu_check.get("accelerator_active"):
        errors.append("tpu_acceleration_check.accelerator_active não é true")
    if runtime_metrics.get("compile_count", 0) <= 0:
        errors.append("nenhuma compilação XLA foi registrada")
    if runtime_metrics.get("execute_count", 0) <= 0:
        errors.append("nenhuma execução XLA foi registrada")

    return errors


def latest_result(metrics_dir: Path = METRICS_DIR) -> Path:
    candidates = sorted(metrics_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Nenhum resultado JSON encontrado em {metrics_dir}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Homologa uma execução multicore em TPU/XLA.")
    parser.add_argument("result", nargs="?", type=Path, help="JSON do experimento; usa o mais recente por padrão")
    parser.add_argument("--expected-cores", type=int, default=8)
    args = parser.parse_args()

    result_path = args.result or latest_result()
    with result_path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)

    errors = validate_tpu_result(result, expected_cores=args.expected_cores)
    if errors:
        print(f"BL-08 NÃO homologado: {result_path}")
        for error in errors:
            print(f"- {error}")
        return 1

    metrics = result["tpu_acceleration_check"]["xla_runtime_metrics"]
    print(
        f"BL-08 homologado: cores={args.expected_cores}, "
        f"compilações={metrics['compile_count']}, execuções={metrics['execute_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())