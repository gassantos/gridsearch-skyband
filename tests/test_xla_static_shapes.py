import pytest
import torch

from utils.device import validate_xla_batch_shape


def test_static_shape_accepts_repeated_batch_signature():
    batch = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
        "label": torch.ones((2,), dtype=torch.long),
    }

    signature = validate_xla_batch_shape(batch)

    assert validate_xla_batch_shape(batch, signature) == signature


def test_static_shape_rejects_changed_batch_size():
    initial = {"input_ids": torch.ones((2, 8), dtype=torch.long)}
    changed = {"input_ids": torch.ones((1, 8), dtype=torch.long)}
    signature = validate_xla_batch_shape(initial)

    with pytest.raises(RuntimeError, match="Forma dinâmica"):
        validate_xla_batch_shape(changed, signature)


def test_static_shape_rejects_changed_sequence_length():
    initial = {"input_ids": torch.ones((2, 8), dtype=torch.long)}
    changed = {"input_ids": torch.ones((2, 16), dtype=torch.long)}
    signature = validate_xla_batch_shape(initial)

    with pytest.raises(RuntimeError, match="max_seq_length"):
        validate_xla_batch_shape(changed, signature)


def test_static_shape_ignores_non_tensor_metadata():
    initial = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "guid": ["case_1", "case_2"],
    }
    changed_metadata = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "guid": ["case_3", "case_4"],
    }

    signature = validate_xla_batch_shape(initial)

    assert validate_xla_batch_shape(changed_metadata, signature) == signature