"""
Módulo para detecção automática de dispositivo de computação.
Suporta CUDA (NVIDIA), TPU (TorchXLA), MPS (Apple Silicon) e CPU.
"""
from __future__ import annotations

import sys
import os
import platform
import logging
import torch


logger = logging.getLogger(__name__)


try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    xm = None
    _XLA_AVAILABLE = False


def _get_xla_device_or_none():
    """Retorna um device XLA quando disponível; caso contrário, ``None``."""
    if not _XLA_AVAILABLE or xm is None:
        return None

    try:
        supported = xm.get_xla_supported_devices()
        if supported:
            return xm.xla_device()
    except Exception as exc:
        logger.warning("Falha ao detectar dispositivo XLA: %s", exc)

    return None


def get_tpu_info() -> dict:
    """Retorna informações de TPU/XLA quando disponíveis.

    Campos ausentes são retornados como ``None`` para facilitar serialização
    e consumo uniforme por relatórios.
    """
    info = {
        "xla_available": bool(_XLA_AVAILABLE and xm is not None),
        "xla_supported_devices": [],
        "xla_device_count": 0,
        "tpu_kind": None,
        "xla_device": None,
        "pjrt_device": os.getenv("PJRT_DEVICE"),
    }

    if not info["xla_available"]:
        return info

    try:
        supported = xm.get_xla_supported_devices()
        info["xla_supported_devices"] = supported
        info["xla_device_count"] = len(supported)
        if supported:
            info["tpu_kind"] = xm.xla_device_kind()
            info["xla_device"] = str(xm.xla_device())
    except Exception as exc:
        logger.warning("Falha ao coletar informações de TPU/XLA: %s", exc)

    return info


def get_device(prefer_cpu: bool = False):
    """
    Detecta o melhor dispositivo disponível de forma multiplataforma.
    
    Args:
        prefer_cpu: Se True, força uso de CPU mesmo com GPU disponível
    
    Returns:
        torch.device: Device otimizado para a plataforma atual, ou None se torch indisponível
    """
    if torch is None:
        logger.warning("PyTorch não disponível. Retornando device=None.")
        return None

    if prefer_cpu:
        logger.info("CPU mode forced by user")
        return torch.device("cpu")
    
    system = platform.system()
    
    # Windows/Linux com CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        return device
    
    elif system == "Linux":
        xla_device = _get_xla_device_or_none()
        if xla_device is not None:
            logger.info("Using XLA device (TPU)")
            return xla_device

    # macOS com Apple Silicon (MPS)
    elif system == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
        return device
    
    # CPU fallback
    device = torch.device("cpu")
    logger.info(f"Using CPU on {system} system")
    if system == "Darwin":
        logger.warning("MPS not available. Ensure PyTorch version supports Apple Silicon.")
    elif system in ["Windows", "Linux"]:
        logger.warning("CUDA not available. Install CUDA toolkit for GPU acceleration.")
    return device


def get_device_info():
    """
    Retorna informações detalhadas sobre o dispositivo.
    
    Returns:
        dict: Informações sobre dispositivo, memória e capacidade
    """
    if torch is None:
        return {
            "device_type": "unavailable",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "pytorch_version": None,
            "error": "PyTorch não disponível neste ambiente"
        }
    device = get_device()
    assert device is not None  # torch is not None, portanto get_device() sempre retorna um device
    info = {
        "device_type": device.type,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })
    elif device.type == "xla":
        info.update(get_tpu_info())
    
    return info


def set_device_optimization(device):
    """
    Configura otimizações específicas do dispositivo.
    
    Args:
        device: torch.device para otimizar
    """
    if torch is None or device is None:
        logger.warning("PyTorch não disponível. Otimizações de device ignoradas.")
        return
    if device.type == "cuda":
        # Habilita TF32 para Ampere GPUs (melhor performance)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Habilita benchmark para encontrar melhor algoritmo
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimizations enabled (TF32, cuDNN benchmark)")
    elif device.type == "mps":
        # MPS ainda é experimental, sem otimizações específicas por enquanto
        logger.info("MPS device set (experimental support)")
    else:
        logger.info("CPU mode - no specific optimizations applied")


def get_torch_device() -> dict:
    """Retorna o dispositivo PyTorch disponível (CPU, GPU ou TPU)."""
    if torch is None:
        return {'type': 'unavailable', 'name': None, 'device': None}

    if platform.system() == "Linux" and _XLA_AVAILABLE and xm is not None:
        try:
            if len(xm.get_xla_supported_devices()) > 0:
                return {
                    'type': 'TPU',
                    'name': xm.xla_device_kind(),
                    'device': xm.xla_device()
                }
        except Exception as exc:
            logger.warning("Torch_XLA disponível, mas sem dispositivo utilizável: %s", exc)
    
    if torch.cuda.is_available():
        return {
            'type': 'GPU',
            'name': torch.cuda.get_device_name(0),
            'device': torch.device('cuda')
        }
    return {
        'type': 'CPU',
        'name': platform.processor(),
        'device': torch.device('cpu')
    }