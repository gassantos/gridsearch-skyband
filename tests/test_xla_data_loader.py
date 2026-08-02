from unittest.mock import MagicMock, patch

import pytest
import torch

from utils.device import prepare_data_loader


def test_prepare_data_loader_preserves_loader_outside_xla():
    loader = MagicMock()

    result = prepare_data_loader(loader, torch.device("cpu"))

    assert result is loader


def test_prepare_data_loader_wraps_loader_for_xla():
    loader = MagicMock()
    device = MagicMock()
    device.type = "xla"
    parallel_loader = MagicMock()
    wrapped_loader = MagicMock()
    parallel_loader.MpDeviceLoader.return_value = wrapped_loader

    with patch("utils.device.xla_parallel_loader", parallel_loader):
        result = prepare_data_loader(loader, device)

    assert result is wrapped_loader
    parallel_loader.MpDeviceLoader.assert_called_once_with(loader, device)


def test_prepare_data_loader_rejects_missing_xla_loader():
    device = MagicMock()
    device.type = "xla"

    with patch("utils.device.xla_parallel_loader", None):
        with pytest.raises(RuntimeError, match="MpDeviceLoader"):
            prepare_data_loader(MagicMock(), device)