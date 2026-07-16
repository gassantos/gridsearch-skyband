"""
TPU Acceleration Check — Detecção de execução na host CPU em ambiente TPU (BL-08)
==================================================================================

Implementa a detecção do problema identificado na Seção 4 do artigo PSLA4ML:

    "O ambiente TPU não utilizou efetivamente o acelerador: os logs do
     CodeCarbon indicam gpu_power = 0,0 W e device_type = CPU, sugerindo
     ausência de execução na MXU.  Uma avaliação adequada requer ativação
     do XLA e workflow compatível (e.g., padding estático)."

O módulo verifica, pós-execução, se um workflow declarado em ambiente TPU
foi efetivamente processado pelo acelerador (MXU) ou caiu silenciosamente
na host CPU — o que invalida os traces de energia e tempo para fins de
comparação entre ambientes.

Autor: Gustavo Alexandre
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Limiar mínimo de energia GPU/TPU para considerar o acelerador ativo (kWh)
# Valor abaixo → execução na host CPU detectada
_MIN_ACCELERATOR_ENERGY_KWH = 1e-6

# Fator de plausibilidade: se tempo_exec / energia > este limiar e device=TPU,
# indica execução em CPU (que consome ~0.1 kW/h vs. TPU ~0.35 kW/h)
_CPU_ENERGY_RATIO_THRESHOLD = 0.05  # kWh por hora de CPU


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
        accelerator_active: ``True`` se o acelerador MXU foi efetivamente
            utilizado.  ``False`` indica execução silenciosa na host CPU.
        warning: Mensagem de aviso legível, ou ``None`` se não há problema.
        recommendations: Lista de passos para corrigir o problema.
    """

    device_type: str = "CPU"
    is_tpu_environment: bool = False
    xla_available: bool = False
    gpu_energy_kwh: Optional[float] = None
    accelerator_active: bool = True
    warning: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dicionário JSON-compatível."""
        return {
            "device_type": self.device_type,
            "is_tpu_environment": self.is_tpu_environment,
            "xla_available": self.xla_available,
            "gpu_energy_kwh": self.gpu_energy_kwh,
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


def _get_gpu_energy_from_tracker(tracker: Any) -> Optional[float]:
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


def check_tpu_acceleration(
    device_type: str,
    tracker: Any = None,
    exec_time_sec: Optional[float] = None,
) -> TpuAccelerationStatus:
    """Verifica se o acelerador TPU foi efetivamente utilizado.

    Detecta o problema da Seção 4 do artigo PSLA4ML: o workflow declarado
    em ambiente TPU pode ser executado silenciosamente na host CPU quando
    ``torch_xla`` não está corretamente configurado (sem XLA compilado e
    sem padding estático no tokenizer).

    A detecção é baseada em:
    1. ``device_type == "TPU"`` (declarado via ``get_torch_device()``)
    2. ``gpu_energy_kwh ≤ _MIN_ACCELERATOR_ENERGY_KWH`` nos dados do
       CodeCarbon — indica que o MXU não consumiu energia, logo não computou

    Args:
        device_type: Tipo de dispositivo de ``get_torch_device()["type"]``.
        tracker: Instância do ``EmissionsTracker`` do CodeCarbon após
            ``tracker.stop()``.  Usado para ler ``gpu_energy``.
        exec_time_sec: Tempo de execução em segundos (para contexto no log).

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

    # Determina se o acelerador está ativo
    if gpu_energy is not None:
        accelerator_active = gpu_energy > _MIN_ACCELERATOR_ENERGY_KWH
    else:
        # Sem dados de energia: assume ativo apenas se XLA está disponível
        # (diagnóstico menos preciso, mas conservador)
        accelerator_active = xla_ok

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
            accelerator_active=True,
        )

    # Acelerador NÃO ativo — gera aviso e recomendações
    time_str = f" (exec_time={exec_time_sec:.1f}s)" if exec_time_sec else ""
    gpu_energy_str = f"{gpu_energy:.2e}" if gpu_energy is not None else "N/A"
    warning = (
        f"[BL-08] TPU acelerador NÃO ativo{time_str}: "
        f"gpu_energy={gpu_energy_str} kWh ≈ 0 — workflow executado na host CPU. "
        "O trace de energia e tempo deste experimento reflete CPU, não TPU. "
        "Veja 'recommendations' para corrigir."
    )

    recommendations = [
        "1. Instale torch_xla compatível com seu ambiente: "
        "'uv sync --extra tpu' (Linux) ou use Google Colab com runtime TPU.",
        "2. Ative o XLA antes do treinamento: "
        "'import torch_xla.core.xla_model as xm; device = xm.xla_device()'.",
        "3. Use padding estático no tokenizer para evitar recompilações XLA: "
        "'tokenizer(..., padding=\"max_length\", max_length=256, truncation=True)'.",
        "4. Confirme que o modelo e os tensores estão no device XLA: "
        "'model.to(device)' e 'batch.to(device)' antes do forward pass.",
        "5. Verifique o CodeCarbon após a execução: gpu_energy deve ser > 0 kWh.",
    ]

    logger.warning(warning)

    return TpuAccelerationStatus(
        device_type=device_type,
        is_tpu_environment=True,
        xla_available=xla_ok,
        gpu_energy_kwh=gpu_energy,
        accelerator_active=False,
        warning=warning,
        recommendations=recommendations,
    )
