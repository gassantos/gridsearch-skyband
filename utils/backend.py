"""Abstração de backend de execução (fase inicial).

Nesta fase, o pipeline de treino permanece em PyTorch.
Este módulo apenas normaliza a preferência do usuário e registra
``backend_requested``/``backend_resolved`` para migração gradual.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


_BACKEND_ALIASES = {
    "torch": "torch",
    "pytorch": "torch",
    "jax": "jax",
    "flax": "jax",
}


def _normalize_backend_name(value: str | None) -> str:
    """Normaliza aliases de backend para um nome canônico."""
    if not value:
        return "torch"
    return _BACKEND_ALIASES.get(value.strip().lower(), "torch")


def _safe_config_get(config, section: str, option: str) -> str | None:
    """Lê uma chave de configuração retornando ``None`` em caso de ausência."""
    try:
        return config.get(section, option)
    except Exception:
        return None


def get_requested_backend(config) -> str:
    """Extrai o backend solicitado da configuração.

    Prioridade:
    1. ``[environment] backend``
    2. ``[environment] framework``
    3. fallback ``torch``
    """
    explicit_backend = _safe_config_get(config, "environment", "backend")
    if explicit_backend:
        return _normalize_backend_name(explicit_backend)

    framework = _safe_config_get(config, "environment", "framework")
    return _normalize_backend_name(framework)


def is_jax_available() -> bool:
    """Indica se JAX está instalado no ambiente."""
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def resolve_execution_backend(config) -> dict[str, str]:
    """Resolve backend solicitado para backend executável no pipeline atual.

    Como o treino ainda é PyTorch-only nesta fase, ``backend_resolved``
    é sempre ``torch``. Quando ``jax`` é solicitado, registramos motivo
    explícito para facilitar diagnóstico e migração futura.
    """
    requested = get_requested_backend(config)

    if requested == "jax":
        if is_jax_available():
            reason = "jax_requested_but_not_integrated_yet"
        else:
            reason = "jax_not_installed"
        logger.warning(
            "Backend '%s' solicitado; pipeline atual executa com PyTorch. reason=%s",
            requested,
            reason,
        )
        return {
            "backend_requested": requested,
            "backend_resolved": "torch",
            "backend_reason": reason,
        }

    return {
        "backend_requested": requested,
        "backend_resolved": "torch",
        "backend_reason": "default_or_torch_requested",
    }
