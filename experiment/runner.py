"""
Motor de execução de experimentos BERT-PLI
============================================

Executa um experimento rastreável de ponta a ponta:
- Carrega e aplica a configuração em cascata (default → experimento)
- Mede tempo de execução, uso de RAM e emissões de CO₂ (via codecarbon)
- Captura stdout do loop de treino sem bloquear a saída em tempo real
- Persiste artefatos em JSON (por execução) e CSV (histórico acumulado)

Autor: Gustavo Alexandre
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import psutil
import torch

from utils.device import get_torch_device
from utils.util import print_system_info

# Imports concretos de tools/ — usados como defaults quando nenhum
# callable alternativo é injetado via parâmetros (DIP).
from tools.eval_tool import (
    convert_test_results_to_task1 as _default_convert_test_results,
    compute_metrics as _default_compute_metrics,
)

from gridsearch.protocols import (
    InitFn,
    TrainFn,
    ConvertTestResultsFn,
    ComputeMetricsFn,
)

from .helpers import (
    ENERGY_COST_USD_PER_KWH,
    TeeStream,
    estimate_bert_flops,
    load_config,
    now_iso,
)
from .evaluation import extract_eval_metrics
from .persistence import append_csv_row, build_result_dict, write_json_result

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def execute_experiment(
    config_path: str,
    gpu_list: list[int] | None = None,
    parallel_workers: int = 1,
    train_file: str | None = None,
    dataset_overrides: dict | None = None,
    *,
    init_fn: InitFn | None = None,
    train_fn: TrainFn | None = None,
    convert_results_fn: ConvertTestResultsFn | None = None,
    compute_metrics_fn: ComputeMetricsFn | None = None,
) -> None:
    """Executa um experimento completo de forma rastreável.

    Realiza treino in-process com captura de stdout, amostragem contínua
    de RAM, rastreamento de emissões de CO₂ (opcional) e persistência
    de artefatos em JSON e CSV.

    Args:
        config_path: Caminho para o arquivo ``.config`` do experimento.
        gpu_list: Lista de IDs de GPU a utilizar.
        parallel_workers: Número de workers paralelos.
        train_file: Nome do arquivo de treino sem extensão.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever.
        init_fn: Callable compatível com ``InitFn`` protocol.
        train_fn: Callable compatível com ``TrainFn`` protocol.
        convert_results_fn: Callable compatível com ``ConvertTestResultsFn``.
        compute_metrics_fn: Callable compatível com ``ComputeMetricsFn``.
    """
    import tempfile as _tempfile

    # Coleta informações do sistema
    print_system_info()
    _torch_device_info = get_torch_device()
    device_name = _torch_device_info['name']

    # Aplica train_file e/ou dataset_overrides criando um config temporário
    _temp_config_path: str | None = None
    if train_file is not None or dataset_overrides:
        _base_cfg = load_config(config_path)
        if not _base_cfg.has_section("data"):
            _base_cfg.add_section("data")
        if train_file is not None:
            _base_cfg.set("data", "train_file_list", f"{train_file}.json")
        if dataset_overrides:
            for key, value in dataset_overrides.items():
                section = "data"
                _base_cfg.set(section, key, str(value))
        _fd, _temp_config_path = _tempfile.mkstemp(suffix=".config")
        os.close(_fd)
        with open(_temp_config_path, "w") as _f:
            _base_cfg.write(_f)
        config_path = _temp_config_path

    cfg = load_config(config_path)

    exp = cfg["experiment"]
    train = cfg["train"]
    env = cfg["environment"]
    mon = cfg["monitoring"]

    # Nome do dataset de treino utilizado (para rastreabilidade)
    _train_dataset_name = (
        cfg.get("data", "train_file_list", fallback="").replace(".json", "")
        or "train_task2"
    )

    experiment_id = str(uuid.uuid4())
    device_type = _torch_device_info['type']

    # Sincroniza o CUDA para garantir que as medições de tempo sejam precisas
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    start_iso = now_iso()
    DATE_EXEC = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------- ENERGY TRACKER --------
    tracker = None
    if EmissionsTracker and mon.getboolean("enable_monitoring"):
        from .helpers import METRICS_DIR
        tracker = EmissionsTracker(
            project_name=exp["name"],
            output_dir=METRICS_DIR.as_posix(),
            log_level="error",
            output_file=f"EmissionsCO2_{device_type}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        tracker.start()

    # -------- EXEC TRAIN (IN-PROCESS) --------
    from utils.config import create_config
    if init_fn is None:
        from tools.init_tool import init_all
        init_fn = init_all
    if train_fn is None:
        from tools.train_tool import train as run_train
        train_fn = run_train

    # Resolve defaults de avaliação
    convert_test_results_to_task1 = convert_results_fn or _default_convert_test_results
    compute_metrics = compute_metrics_fn or _default_compute_metrics

    if gpu_list is None:
        gpu_list = [0] if torch.cuda.is_available() else []

    tee = TeeStream(sys.stdout)
    sys.stdout = tee

    # Amostragem de RAM em thread daemon (1 amostra/segundo).
    ram_samples: list = []
    _stop_ram = threading.Event()

    def _sample_ram() -> None:
        proc = psutil.Process(os.getpid())
        while not _stop_ram.is_set():
            try:
                ram_samples.append(proc.memory_info().rss / (1024 ** 2))
            except psutil.NoSuchProcess:
                break
            _stop_ram.wait(timeout=1.0)

    ram_thread = threading.Thread(target=_sample_ram, daemon=True)
    ram_thread.start()

    status = "failed"
    stdout = ""
    stderr = ""
    output_lines: list = []

    try:
        config = create_config(config_path)
        parameters = init_fn(config, gpu_list, None, "train")
        train_fn(parameters, config, gpu_list)
        status = "success"
    except Exception as exc:
        logger.error("Treinamento falhou: %s", exc, exc_info=True)
        stderr = str(exc)
    finally:
        _stop_ram.set()
        ram_thread.join(timeout=5)
        sys.stdout = tee.original
        output_lines = tee.lines
        stdout = "".join(output_lines)
        # Remove config temporário se foi criado
        if _temp_config_path and os.path.exists(_temp_config_path):
            try:
                os.unlink(_temp_config_path)
            except OSError:
                pass

    # -------- STOP ENERGY TRACKER --------
    emissions_kg = None
    energy_kwh = None

    if tracker:
        emissions_kg = tracker.stop()
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data is not None:
                energy_kwh = tracker.final_emissions_data.energy_consumed
            elif hasattr(tracker, '_total_energy') and tracker._total_energy is not None:
                energy_kwh = tracker._total_energy.kWh
        except Exception as e:
            logger.warning("Não foi possível obter energy_kwh do tracker: %s", e)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exec_time = time.perf_counter() - start_time
    end_iso = now_iso()

    # -------- METRICS (proxy / external) --------
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else None
    peak_ram = max(ram_samples) if ram_samples else None

    profiling_path = Path(cfg.get("output", "model_path")) / cfg.get("output", "model_name") / "profiling_metrics.json"
    total_gflops = 0
    avg_gflops_per_batch = 0

    if profiling_path.exists():
        try:
            with open(profiling_path, "r") as f:
                profiling_data = json.load(f)
                total_gflops = profiling_data.get("total_gflops", 0)
                avg_gflops_per_batch = profiling_data.get("avg_gflops_per_batch", 0)
                logger.info(f"Loaded profiling metrics: {avg_gflops_per_batch:.2f} GFLOPs/batch")
        except Exception as e:
            logger.warning(f"Could not load profiling metrics: {e}")
            total_gflops = estimate_bert_flops(seq_len=256)
    else:
        logger.warning("Profiling metrics not found, using estimation")
        total_gflops = estimate_bert_flops(seq_len=256)

    # -------- EVAL METRICS --------
    eval_metrics = extract_eval_metrics(
        status=status,
        output_lines=output_lines,
        cfg=cfg,
        config_path=config_path,
        convert_test_results_fn=convert_test_results_to_task1,
        compute_metrics_fn=compute_metrics,
    )

    # -------- JSON OUTPUT --------
    exp_id = exp["name"]
    optmzr = train["optimizer"]
    lr = f"lr{train['learning_rate']}".replace('-', '')
    bs = f"bs{train['batch_size']}"
    ep = f"ep{train['epoch']}"
    json_filename = f"{exp_id}_{optmzr}_{lr}_{bs}_{ep}_{DATE_EXEC}.json"

    cost_usd = float(energy_kwh) * ENERGY_COST_USD_PER_KWH if energy_kwh is not None else None

    result = build_result_dict(
        experiment_id=experiment_id,
        json_filename=json_filename,
        seed=int(exp["seed"]),
        status=status,
        date_exec=DATE_EXEC,
        start_iso=start_iso,
        end_iso=end_iso,
        device_type=device_type,
        device_name=device_name,
        precision=env["precision"],
        parallel_workers=parallel_workers,
        train_dataset_name=_train_dataset_name,
        optimizer=train["optimizer"],
        learning_rate=float(train["learning_rate"]),
        avg_gflops_per_batch=avg_gflops_per_batch,
        batch_size=int(train["batch_size"]),
        epoch=int(train["epoch"]),
        exec_time=exec_time,
        energy_kwh=energy_kwh,
        emissions_kg=emissions_kg,
        cost_usd=cost_usd,
        avg_ram=avg_ram,
        peak_ram=peak_ram,
        total_gflops=total_gflops,
        eval_metrics=eval_metrics,
        stdout=stdout,
        stderr=stderr,
    )

    write_json_result(result, json_filename)

    # -------- CSV AGGREGATION --------
    append_csv_row(
        experiment_id=experiment_id,
        json_filename=json_filename,
        seed=int(exp["seed"]),
        device_type=device_type,
        parallel_workers=parallel_workers,
        train_dataset_name=_train_dataset_name,
        optimizer=train["optimizer"],
        learning_rate=float(train["learning_rate"]),
        batch_size=int(train["batch_size"]),
        epoch=int(train["epoch"]),
        exec_time=exec_time,
        energy_kwh=energy_kwh,
        emissions_kg=emissions_kg,
        cost_usd=cost_usd,
        avg_ram=avg_ram,
        peak_ram=peak_ram,
        avg_gflops_per_batch=avg_gflops_per_batch,
        total_gflops=total_gflops,
        status=status,
        end_iso=end_iso,
        eval_metrics=eval_metrics,
    )

    print(f"[OK] Wrapper finalizou em {exec_time:.2f} segundos - {exp['name']} ({status})")
