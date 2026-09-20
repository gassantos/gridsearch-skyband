"""
Resource Discovery — Preenchimento de parâmetros de recurso para o MILP PSLA4ML
=================================================================================

Preenche os parâmetros de recurso do modelo de otimização CC-IP
(Yuri Frota, 2026) — ``d_i`` (disco), ``m_i`` (memória), ``n_i^p``
(núcleos por tipo de processador) e ``g_i^p`` (poder de processamento
por núcleo) — a partir de duas fontes:

1. **Detecção local**: para o recurso ``local``, hardware real da máquina
   de execução via ``psutil``/``torch.cuda`` (disco, RAM, GPU).
2. **Catálogo estático curado**: para recursos de nuvem (``colab``, ``gcp``,
   ``aws``, ``azure``), especificações públicas de datasheet por modelo de
   GPU, casadas pelo campo ``gpu`` já declarado em
   ``environments.details.<env>.gpu`` (ex.: "NVIDIA T4").

Por que não consultar as APIs de billing/catálogo ao vivo (GCP/AWS/Azure)?
    GCP (Cloud Billing Catalog) e AWS (Price List API) exigem credenciais
    de conta que não estão disponíveis neste ambiente. Uma integração ao
    vivo tornaria o pipeline de pesquisa dependente de rede e não
    determinístico entre execuções — inadequado para reprodutibilidade
    científica. O catálogo estático é versionado e auditável; pode ser
    substituído por uma consulta real quando credenciais estiverem
    disponíveis (ver ``fetch_live`` como ponto de extensão futuro).

``c_ij`` (custo de comunicação entre recursos) não é modelado aqui: o
pipeline atual executa cada experimento em um único recurso isolado (sem
workload distribuído entre múltiplos ``R`` simultaneamente), então a
matriz é inicializada com zeros — mesma lacuna identificada no backlog
BL-W6 (banda entre nós do template de workflow).

Autor: Gustavo Alexandre
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


# ============================================================================
# DETECÇÃO LOCAL DE HARDWARE
# ============================================================================

@dataclass(frozen=True)
class LocalHardwareSpec:
    """Especificação de hardware detectada na máquina de execução local."""

    cpu_logical_cores: int
    ram_total_gb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    gpu_multiprocessor_count: int | None = None


def detect_local_hardware(disk_path: str = ".") -> LocalHardwareSpec:
    """Detecta as especificações reais de hardware da máquina local.

    Usa ``psutil`` para CPU/RAM/disco (sem dependências novas) e
    ``torch.cuda`` para GPU, quando disponível. Nunca acessa a rede.

    Args:
        disk_path: Caminho usado para medir o disco (padrão: diretório atual).

    Returns:
        ``LocalHardwareSpec`` preenchido com os valores detectados. Campos
        de GPU ficam ``None`` quando CUDA não está disponível.
    """
    disk = shutil.disk_usage(disk_path)
    spec = {
        "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_free_gb": disk.free / (1024 ** 3),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_multiprocessor_count": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            spec["gpu_name"] = props.name
            spec["gpu_vram_gb"] = props.total_memory / (1024 ** 3)
            spec["gpu_multiprocessor_count"] = props.multi_processor_count
    except Exception as exc:  # pragma: no cover - defensivo, ambiente sem CUDA
        logger.debug("Detecção de GPU local indisponível: %s", exc)

    return LocalHardwareSpec(**spec)


# ============================================================================
# CATÁLOGO ESTÁTICO DE GPUs (especificações públicas de datasheet)
# ============================================================================

# Valores de referência pública (datasheets NVIDIA, FP16/Tensor peak TFLOPS
# e contagem de Streaming Multiprocessors). Usados como estimativa quando
# não há benchmark local disponível — não substituem medição real.
GPU_CATALOG: dict[str, dict[str, float]] = {
    "NVIDIA RTX 3090":  {"fp16_tflops": 71.0,  "sm_count": 82, "vram_gb": 24.0},
    "NVIDIA T4":        {"fp16_tflops": 65.0,  "sm_count": 40, "vram_gb": 16.0},
    "NVIDIA V100":      {"fp16_tflops": 125.0, "sm_count": 80, "vram_gb": 16.0},
    "NVIDIA A100":      {"fp16_tflops": 312.0, "sm_count": 108, "vram_gb": 40.0},
    "NVIDIA RTX 2050":  {"fp16_tflops": 22.0,  "sm_count": 16, "vram_gb": 4.0},
}

# Disco padrão assumido para VMs de nuvem quando não declarado explicitamente
# em ``environments.details`` (não obtido de API ao vivo — valor configurável).
DEFAULT_CLOUD_DISK_GB = 100.0


def _match_gpu_catalog(gpu_name: Optional[str]) -> Optional[dict[str, float]]:
    """Casa o nome de GPU declarado/detectado com uma entrada do catálogo estático.

    Tolera variantes de nome (ex.: ``torch`` reporta "NVIDIA GeForce RTX 2050",
    enquanto o catálogo usa a chave "NVIDIA RTX 2050") via correspondência por
    substring do modelo, após remover o prefixo "NVIDIA"/"GeForce".
    """
    if not gpu_name:
        return None
    if gpu_name in GPU_CATALOG:
        return GPU_CATALOG[gpu_name]
    normalized = gpu_name.strip().lower()
    for name, spec in GPU_CATALOG.items():
        if name.lower() == normalized:
            return spec
    for name, spec in GPU_CATALOG.items():
        model = name.lower().replace("nvidia", "").strip()
        if model and model in normalized.replace("geforce", "").strip():
            return spec
    return None


# ============================================================================
# MONTAGEM DO CATÁLOGO DE RECURSOS (parâmetros do MILP por recurso i ∈ R)
# ============================================================================

def build_resource_spec(
    env_name: str,
    env_details: dict[str, Any],
    *,
    local_hardware: Optional[LocalHardwareSpec] = None,
) -> dict[str, Any]:
    """Monta a especificação de recurso ``i`` para o modelo MILP PSLA4ML.

    Args:
        env_name: Nome do ambiente (ex.: "local", "colab", "gcp").
        env_details: Entrada correspondente em ``environments.details``.
        local_hardware: Hardware detectado via :func:`detect_local_hardware`.
            Obrigatório apenas quando ``env_name == "local"``.

    Returns:
        Dicionário com os parâmetros do modelo: ``c_i`` (custo/hora),
        ``m_i`` (memória em GB), ``d_i`` (disco em GB), ``n_i`` (núcleos
        por tipo de processador, dict ``{"GPU": int}``), ``g_i`` (GFLOPS
        por núcleo, dict ``{"GPU": float}``), e ``source`` (rastreabilidade
        de onde cada campo veio: "detected_local" | "static_catalog" |
        "declared" | "default").
    """
    c_i = float(env_details.get("cost_per_hour_usd", 0.0))
    gpu_name = env_details.get("gpu")

    if env_name == "local" and local_hardware is not None:
        m_i = local_hardware.gpu_vram_gb or local_hardware.ram_total_gb
        d_i = local_hardware.disk_free_gb
        sm_count = local_hardware.gpu_multiprocessor_count
        catalog = _match_gpu_catalog(local_hardware.gpu_name) or _match_gpu_catalog(gpu_name)
        g_i_gpu = (catalog["fp16_tflops"] * 1000 / catalog["sm_count"]) if catalog else None
        n_i_gpu = sm_count or (catalog["sm_count"] if catalog else None)
        source = "detected_local"
    else:
        catalog = _match_gpu_catalog(gpu_name)
        m_i = float(env_details.get("vram_gb", catalog["vram_gb"] if catalog else 0.0))
        d_i = float(env_details.get("disk_gb", DEFAULT_CLOUD_DISK_GB))
        n_i_gpu = catalog["sm_count"] if catalog else None
        g_i_gpu = (catalog["fp16_tflops"] * 1000 / catalog["sm_count"]) if catalog else None
        source = "static_catalog" if catalog else "default"

    return {
        "resource_id": env_name,
        "c_i": c_i,
        "m_i": m_i,
        "d_i": d_i,
        "n_i": {"GPU": n_i_gpu} if n_i_gpu is not None else {},
        "g_i": {"GPU": g_i_gpu} if g_i_gpu is not None else {},
        "gpu_name": gpu_name,
        "source": source,
    }


def estimate_energy_rate_kwh_per_hour(
    results: list[dict[str, Any]],
    environment: str,
) -> Optional[float]:
    """Estima ``e_i`` (consumo energético por hora) empiricamente a partir do histórico.

    Usa a média de ``energy_kwh / (train_time_sec / 3600)`` entre os
    experimentos concluídos com sucesso no ambiente informado. Mais preciso
    que uma estimativa de catálogo, pois reflete o consumo real medido pelo
    CodeCarbon durante execuções passadas.

    Args:
        results: Lista de resultados de grid search (``status``,
            ``selected_environment`` ou ``grid_params.environment``,
            ``resources``).
        environment: Nome do ambiente a filtrar.

    Returns:
        Taxa média em kWh/hora, ou ``None`` se não houver amostras válidas.
    """
    rates = []
    for r in results:
        if r.get("status") != "success":
            continue
        env = r.get("selected_environment") or r.get("grid_params", {}).get("environment")
        if env != environment:
            continue
        resources = r.get("resources", {})
        try:
            energy_kwh = float(resources.get("energy_kwh"))
            train_time_hours = float(resources.get("train_time_sec")) / 3600.0
        except (TypeError, ValueError):
            continue
        if train_time_hours <= 0:
            continue
        rates.append(energy_kwh / train_time_hours)

    if not rates:
        return None
    return sum(rates) / len(rates)


def build_resource_catalog(
    environments_details: dict[str, dict[str, Any]],
    *,
    results: Optional[list[dict[str, Any]]] = None,
    local_hardware: Optional[LocalHardwareSpec] = None,
) -> dict[str, dict[str, Any]]:
    """Monta o catálogo completo de recursos ``R`` para o modelo MILP PSLA4ML.

    Args:
        environments_details: ``environments.details`` do grid config
            multiambiente (ex.: ``grid_search_multienv.json``).
        results: Histórico de resultados de grid search, usado para estimar
            ``e_i`` empiricamente via :func:`estimate_energy_rate_kwh_per_hour`.
            ``None`` deixa ``e_i`` como ``None`` (não estimável sem histórico).
        local_hardware: Hardware local pré-detectado. ``None`` chama
            :func:`detect_local_hardware` automaticamente quando o ambiente
            ``"local"`` estiver presente.

    Returns:
        Dicionário ``{nome_ambiente: especificação_de_recurso}``.
    """
    catalog: dict[str, dict[str, Any]] = {}
    for env_name, env_details in environments_details.items():
        if not isinstance(env_details, dict):
            continue
        hw = local_hardware
        if env_name == "local" and hw is None:
            hw = detect_local_hardware()
        spec = build_resource_spec(env_name, env_details, local_hardware=hw)
        if results is not None:
            spec["e_i"] = estimate_energy_rate_kwh_per_hour(results, env_name)
        else:
            spec["e_i"] = None
        catalog[env_name] = spec
    return catalog


def communication_cost_matrix(
    resource_ids: list[str],
    *,
    default: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Monta a matriz ``c_ij`` de custo de comunicação entre recursos.

    Placeholder documentado (BL-W6): o pipeline atual executa cada
    experimento em um único recurso isolado, sem workload distribuído entre
    múltiplos elementos de ``R`` simultaneamente — por isso todos os pares
    recebem ``default`` (0.0 = sem custo de comunicação modelado). Extensão
    futura: estimar via preço de egress público por provedor × volume de
    dados transferido (checkpoints), quando essa telemetria existir.

    Args:
        resource_ids: Lista de identificadores de recursos (``R``).
        default: Custo atribuído a todos os pares ``i < j``.

    Returns:
        Dicionário ``{(i, j): custo}`` para todos os pares ``i < j``.
    """
    pairs = {}
    for idx, i in enumerate(resource_ids):
        for j in resource_ids[idx + 1:]:
            pairs[(i, j)] = default
    return pairs
