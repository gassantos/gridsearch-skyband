from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.device import prepare_data_loader, set_data_loader_epoch


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


def test_prepare_data_loader_shards_batches_across_xla_workers():
    loader = DataLoader(TensorDataset(torch.arange(16)), batch_size=2, shuffle=True)
    device = MagicMock(type="xla")
    parallel_loader = MagicMock()
    wrapped_loader = MagicMock()
    parallel_loader.MpDeviceLoader.return_value = wrapped_loader
    runtime = MagicMock()
    runtime.world_size.return_value = 8
    runtime.global_ordinal.return_value = 3

    with (
        patch("utils.device.xla_parallel_loader", parallel_loader),
        patch("utils.device.xr", runtime),
    ):
        result = prepare_data_loader(loader, device)

    sharded_loader = parallel_loader.MpDeviceLoader.call_args.args[0]
    assert result is wrapped_loader
    assert isinstance(sharded_loader.sampler, torch.utils.data.DistributedSampler)
    assert sharded_loader.sampler.num_replicas == 8
    assert sharded_loader.sampler.rank == 3
    assert wrapped_loader._xla_distributed_sampler is sharded_loader.sampler


def test_set_data_loader_epoch_updates_distributed_sampler():
    loader = MagicMock()
    sampler = MagicMock()
    loader._xla_distributed_sampler = sampler

    set_data_loader_epoch(loader, 4)

    sampler.set_epoch.assert_called_once_with(4)