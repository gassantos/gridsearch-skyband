from unittest.mock import MagicMock, patch

import torch

from tools.poolout_tool import pool_out
from tools.test_tool import test as run_test
from utils.device import move_batch_to_device


def _config():
    config = MagicMock()
    config.getint.side_effect = lambda section, key: {
        ("output", "output_time"): 1,
        ("output", "save_step"): 0,
    }[(section, key)]
    return config


def _parameters():
    model = MagicMock()
    model.return_value = {"output": [["case_doc", [0.1, 0.9]]]}
    return {
        "model": model,
        "test_dataset": [{"input_ids": torch.tensor([[1, 2]])}],
    }


def test_move_batch_to_device_preserves_metadata():
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "guid": ["case_doc"],
    }

    result = move_batch_to_device(batch, torch.device("cpu"))

    assert result is batch
    assert result["input_ids"].device.type == "cpu"
    assert result["guid"] == ["case_doc"]


def test_test_flow_prepares_loader_and_moves_batch():
    parameters = _parameters()

    with patch("tools.test_tool.get_device", return_value=torch.device("cpu")), \
         patch("tools.test_tool.prepare_data_loader", side_effect=lambda loader, _: loader) as prepare, \
         patch("tools.test_tool.move_batch_to_device", wraps=move_batch_to_device) as move, \
         patch("tools.test_tool.output_value"):
        result = run_test(parameters, _config(), [])

    prepare.assert_called_once()
    move.assert_called_once()
    assert result == [["case_doc", [0.1, 0.9]]]


def test_poolout_flow_prepares_loader_and_moves_batch(tmp_path):
    parameters = _parameters()

    with patch("tools.poolout_tool.get_device", return_value=torch.device("cpu")), \
         patch("tools.poolout_tool.prepare_data_loader", side_effect=lambda loader, _: loader) as prepare, \
         patch("tools.poolout_tool.move_batch_to_device", wraps=move_batch_to_device) as move, \
         patch("tools.poolout_tool.output_value"):
        result = pool_out(parameters, _config(), [], str(tmp_path / "poolout.jsonl"))

    prepare.assert_called_once()
    move.assert_called_once()
    assert result == [["case_doc", [0.1, 0.9]]]