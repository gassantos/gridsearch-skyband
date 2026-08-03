from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.train_tool import _save_checkpoint


def test_xla_checkpoint_synchronizes_and_saves_on_master_only():
    save_params = {"trained_epoch": 1}
    xla_model = MagicMock()

    with patch("tools.train_tool.xm", xla_model), \
         patch("tools.train_tool.torch.save") as torch_save:
        _save_checkpoint(save_params, "checkpoint.pkl", SimpleNamespace(type="xla"))

    xla_model.mark_step.assert_called_once_with()
    xla_model.wait_device_ops.assert_called_once_with()
    xla_model.save.assert_called_once_with(save_params, "checkpoint.pkl", master_only=True)
    torch_save.assert_not_called()


def test_cpu_checkpoint_preserves_torch_save():
    save_params = {"trained_epoch": 1}

    with patch("tools.train_tool.torch.save") as torch_save:
        _save_checkpoint(save_params, "checkpoint.pkl", SimpleNamespace(type="cpu"))

    torch_save.assert_called_once_with(save_params, "checkpoint.pkl")


def test_xla_checkpoint_rejects_missing_torch_xla():
    with patch("tools.train_tool.xm", None):
        with pytest.raises(RuntimeError, match="torch_xla"):
            _save_checkpoint({}, "checkpoint.pkl", SimpleNamespace(type="xla"))