from types import SimpleNamespace

import pytest
import torch

from tools.train_tool import _configure_mixed_precision


def test_xla_bf16_enables_autocast_without_grad_scaler():
    use_amp, amp_dtype, scaler = _configure_mixed_precision(
        SimpleNamespace(type="xla"),
        "bf16",
    )

    assert use_amp is True
    assert amp_dtype is torch.bfloat16
    assert scaler.is_enabled() is False


def test_xla_fp32_disables_autocast_and_grad_scaler():
    use_amp, _, scaler = _configure_mixed_precision(
        SimpleNamespace(type="xla"),
        "fp32",
    )

    assert use_amp is False
    assert scaler.is_enabled() is False


def test_xla_rejects_fp16():
    with pytest.raises(ValueError, match="bf16 ou fp32"):
        _configure_mixed_precision(SimpleNamespace(type="xla"), "fp16")


def test_cpu_fp16_preserves_grad_scaler_path():
    use_amp, amp_dtype, scaler = _configure_mixed_precision(
        SimpleNamespace(type="cpu"),
        "fp16",
    )

    assert use_amp is True
    assert amp_dtype is torch.float16
    assert scaler.is_enabled() is True