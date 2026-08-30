"""Testes do parser de configuração hierárquico."""

from utils.config import ConfigParser


def test_config_parser_uses_lower_priority_layer_before_requested_fallback():
    config = ConfigParser()
    config.default_config.read_string("[train]\nwarmup_ratio = 0.2")
    config.config.read_string("[train]\nepoch = 3")

    assert config.getfloat("train", "warmup_ratio", fallback=0.1) == 0.2 # type: ignore


def test_config_parser_returns_requested_fallback_when_option_is_absent():
    config = ConfigParser()

    assert config.getfloat("train", "warmup_ratio", fallback=0.1) == 0.1 # type: ignore