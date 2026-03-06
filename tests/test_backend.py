"""Testes unitários para utils/backend.py."""

from unittest.mock import patch

import configparser

from utils.backend import (
    get_requested_backend,
    is_jax_available,
    resolve_execution_backend,
)


def _make_cfg(extra: dict[str, str] | None = None) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("environment")
    if extra:
        for key, value in extra.items():
            cfg.set("environment", key, value)
    return cfg


class TestBackendSelection:
    def test_default_backend_is_torch(self):
        cfg = _make_cfg()
        assert get_requested_backend(cfg) == "torch"

    def test_framework_alias_pytorch_maps_to_torch(self):
        cfg = _make_cfg({"framework": "pytorch"})
        assert get_requested_backend(cfg) == "torch"

    def test_backend_jax_maps_to_jax(self):
        cfg = _make_cfg({"backend": "jax"})
        assert get_requested_backend(cfg) == "jax"

    def test_backend_flax_alias_maps_to_jax(self):
        cfg = _make_cfg({"backend": "flax"})
        assert get_requested_backend(cfg) == "jax"

    def test_unknown_backend_falls_back_to_torch(self):
        cfg = _make_cfg({"backend": "foo"})
        assert get_requested_backend(cfg) == "torch"


class TestExecutionResolver:
    def test_jax_requested_and_installed_is_torch_for_now(self):
        cfg = _make_cfg({"backend": "jax"})
        with patch("utils.backend.is_jax_available", return_value=True):
            out = resolve_execution_backend(cfg)
        assert out["backend_requested"] == "jax"
        assert out["backend_resolved"] == "torch"
        assert out["backend_reason"] == "jax_requested_but_not_integrated_yet"

    def test_jax_requested_not_installed_falls_back_with_reason(self):
        cfg = _make_cfg({"backend": "jax"})
        with patch("utils.backend.is_jax_available", return_value=False):
            out = resolve_execution_backend(cfg)
        assert out["backend_requested"] == "jax"
        assert out["backend_resolved"] == "torch"
        assert out["backend_reason"] == "jax_not_installed"

    def test_torch_requested_remains_torch(self):
        cfg = _make_cfg({"backend": "torch"})
        out = resolve_execution_backend(cfg)
        assert out["backend_requested"] == "torch"
        assert out["backend_resolved"] == "torch"
        assert out["backend_reason"] == "default_or_torch_requested"


def test_is_jax_available_returns_bool():
    assert isinstance(is_jax_available(), bool)
