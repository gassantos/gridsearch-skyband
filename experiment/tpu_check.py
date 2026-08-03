"""
TPU Acceleration Check — Diagnóstico de execução XLA em ambiente TPU (BL-08)
=============================================================================

Implementa a detecção do problema identificado na Seção 4 do artigo PSLA4ML:

    "Uma avaliação adequada requer ativação do XLA e workflow compatível
     (e.g., padding estático), com evidência de execução no runtime XLA."

O módulo verifica, pós-execução, se um workflow declarado em ambiente TPU
foi efetivamente processado pelo runtime XLA. Os dados do CodeCarbon são
preservados como telemetria auxiliar, sem determinar o diagnóstico.

Autor: Gustavo Alexandre
"""

import importlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TpuAccelerationStatus:
    """Status da verificação de ativação do acelerador TPU.

    Attributes:
        device_type: Tipo de dispositivo detectado por ``get_torch_device()``
            (``"TPU"``, ``"GPU"``, ``"CPU"``).
        is_tpu_environment: ``True`` se ``device_type == "TPU"``.
        xla_available: ``True`` se ``torch_xla`` está instalado e importável.
        gpu_energy_kwh: Energia GPU/TPU registrada pelo CodeCarbon (kWh).
            ``None`` quando o CodeCarbon não estava ativo ou não reportou.
        xla_runtime_metrics: Contagens nativas de compilação, execução e
            sincronização reportadas pelo Torch XLA.
        accelerator_active: ``True`` se o acelerador MXU foi efetivamente
            utilizado. ``False`` indica ausência de evidência no runtime XLA.
        warning: Mensagem de aviso legível, ou ``None`` se não há problema.
        recommendations: Lista de passos para corrigir o problema.
    """

    device_type: str = "CPU"
    is_tpu_environment: bool = False
    xla_available: bool = False
    gpu_energy_kwh: float | None = None
    xla_runtime_metrics: dict[str, Any] = field(default_factory=dict)
    accelerator_active: bool = True
    warning: str | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializa para dicionário JSON-compatível."""
        return {
            "device_type": self.device_type,
            "is_tpu_environment": self.is_tpu_environment,
            "xla_available": self.xla_available,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "xla_runtime_metrics": self.xla_runtime_metrics,
            "accelerator_active": self.accelerator_active,
            "warning": self.warning,
            "recommendations": self.recommendations,
        }


def _xla_available() -> bool:
    """Verifica se ``torch_xla`` está instalado e importável."""
    try:
        import torch_xla.core.xla_model  # noqa: F401
        return True
    except ImportError:
        return False


def _get_gpu_energy_from_tracker(tracker: Any) -> float | None:
    """Extrai a energia GPU/TPU registrada pelo CodeCarbon (kWh).

    Tenta acessar ``tracker.final_emissions_data.gpu_energy``.
    Retorna ``None`` se os dados não estiverem disponíveis.
    """
    if tracker is None:
        return None
    try:
        final_data = getattr(tracker, "final_emissions_data", None)
        if final_data is None:
            return None
        gpu_energy = getattr(final_data, "gpu_energy", None)
        if gpu_energy is None:
            return None
        return float(gpu_energy)
    except Exception:
        return None


def _extract_total_samples(report: str, metric_name: str) -> int:
    match = re.search(
        rf"Metric:\s*{re.escape(metric_name)}\s+.*?TotalSamples:\s*(\d+)",
        report,
        flags=re.DOTALL,
    )
    return int(match.group(1)) if match else 0


def _extract_counter(report: str, counter_name: str) -> int:
    match = re.search(
        rf"Counter:\s*{re.escape(counter_name)}\s+Value:\s*(\d+)",
        report,
        flags=re.DOTALL,
    )
    return int(match.group(1)) if match else 0


def collect_xla_runtime_metrics() -> dict[str, Any]:
    """Coleta evidências nativas de compilação e execução do Torch XLA."""
    try:
        metrics_module = importlib.import_module("torch_xla.debug.metrics")
        report = metrics_module.metrics_report()
        return {
            "available": True,
            "compile_count": _extract_total_samples(report, "CompileTime"),
            "execute_count": _extract_total_samples(report, "ExecuteTime"),
            "mark_step_count": _extract_counter(report, "MarkStep"),
        }
    except (ImportError, RuntimeError) as exc:
        return {
            "available": False,
            "compile_count": 0,
            "execute_count": 0,
            "mark_step_count": 0,
            "error": str(exc),
        }


def check_tpu_acceleration(
    device_type: str,
    tracker: Any = None,
    exec_time_sec: float | None = None,
    xla_runtime_metrics: dict[str, Any] | None = None,
) -> TpuAccelerationStatus:
    """Verifica se o acelerador TPU foi efetivamente utilizado.

    Detecta o problema da Seção 4 do artigo PSLA4ML: o workflow declarado
    em ambiente TPU pode ser executado silenciosamente na host CPU quando
    ``torch_xla`` não está corretamente configurado (sem XLA compilado e
    sem padding estático no tokenizer).

     A detecção é baseada em:
     1. ``device_type == "TPU"`` (declarado via ``get_torch_device()``)
     2. ``ExecuteTime.TotalSamples > 0`` no relatório nativo do Torch XLA.
         A energia do CodeCarbon é mantida apenas como telemetria auxiliar.

    Args:
        device_type: Tipo de dispositivo de ``get_torch_device()["type"]``.
        tracker: Instância do ``EmissionsTracker`` do CodeCarbon após
            ``tracker.stop()``.  Usado para ler ``gpu_energy``.
        exec_time_sec: Tempo de execução em segundos (para contexto no log).
        xla_runtime_metrics: Métricas XLA já coletadas. Quando ``None``, são
            coletadas no processo atual.

    Returns:
        :class:`TpuAccelerationStatus` com diagnóstico completo.
    """
    is_tpu = device_type.upper() == "TPU"

    if not is_tpu:
        # Não é ambiente TPU — nenhuma verificação necessária
        return TpuAccelerationStatus(
            device_type=device_type,
            is_tpu_environment=False,
            xla_available=_xla_available(),
            accelerator_active=True,
        )

    xla_ok = _xla_available()
    gpu_energy = _get_gpu_energy_from_tracker(tracker)
    runtime_metrics = xla_runtime_metrics if xla_runtime_metrics is not None else (
        collect_xla_runtime_metrics()
        if xla_ok
        else {
            "available": False,
            "compile_count": 0,
            "execute_count": 0,
            "mark_step_count": 0,
            "error": "torch_xla não está disponível",
        }
    )

    # ExecuteTime comprova que grafos foram executados pelo runtime XLA.
    accelerator_active = xla_ok and runtime_metrics.get("execute_count", 0) > 0

    if accelerator_active:
        logger.info(
            "[TPU-CHECK] Acelerador ativo — gpu_energy=%.6f kWh, xla=%s",
            gpu_energy or 0.0, xla_ok,
        )
        return TpuAccelerationStatus(
            device_type=device_type,
            is_tpu_environment=True,
            xla_available=xla_ok,
            gpu_energy_kwh=gpu_energy,
            xla_runtime_metrics=runtime_metrics,
            accelerator_active=True,
        )

    # Acelerador NÃO ativo — gera aviso e recomendações
    time_str = f" (exec_time={exec_time_sec:.1f}s)" if exec_time_sec else ""
    gpu_energy_str = f"{gpu_energy:.2e}" if gpu_energy is not None else "N/A"
    warning = (
        f"[BL-08] TPU acelerador NÃO ativo{time_str}: "
        f"nenhuma execução foi registrada pelo runtime XLA; gpu_energy={gpu_energy_str} kWh. "
        "Não há evidência suficiente de processamento na MXU. "
        "Veja 'recommendations' para corrigir."
    )

    recommendations = [
        ("1. Instale torch_xla compatível com seu ambiente: "
        "'uv sync --extra tpu' (Linux) ou use Google Colab com runtime TPU."),
        ("2. Ative o XLA antes do treinamento: "
        "'import torch_xla.core.xla_model as xm; device = xm.xla_device()'."),
        ("3. Use padding estático no tokenizer para evitar recompilações XLA: "
        "'tokenizer(..., padding=\"max_length\", max_length=256, truncation=True)'."),
        ("4. Confirme que o modelo e os tensores estão no device XLA: "
        "'model.to(device)' e 'batch.to(device)' antes do forward pass."),
        ("5. Confirme xla_runtime_metrics.execute_count > 0 no resultado do experimento."),
    ]

    logger.warning(warning)

    return TpuAccelerationStatus(
        device_type=device_type,
        is_tpu_environment=True,
        xla_available=xla_ok,
        gpu_energy_kwh=gpu_energy,
        xla_runtime_metrics=runtime_metrics,
        accelerator_active=False,
        warning=warning,
        recommendations=recommendations,
    )
