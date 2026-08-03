from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.train_tool import _optimizer_step


def test_optimizer_step_uses_xla_runtime():
    optimizer = MagicMock()
    scaler = MagicMock()
    xla_model = MagicMock()

    with patch("tools.train_tool.xm", xla_model):
        _optimizer_step(optimizer, scaler, SimpleNamespace(type="xla"))

    xla_model.optimizer_step.assert_called_once_with(optimizer, barrier=True)
    scaler.step.assert_not_called()
    scaler.update.assert_not_called()


def test_optimizer_step_preserves_grad_scaler_path():
    optimizer = MagicMock()
    scaler = MagicMock()

    _optimizer_step(optimizer, scaler, SimpleNamespace(type="cpu"))

    scaler.step.assert_called_once_with(optimizer)
    scaler.update.assert_called_once_with()


def test_optimizer_step_rejects_xla_without_torch_xla():
    with patch("tools.train_tool.xm", None):  # noqa: SIM117
        with pytest.raises(RuntimeError, match="torch_xla"):
            _optimizer_step(MagicMock(), MagicMock(), SimpleNamespace(type="xla"))