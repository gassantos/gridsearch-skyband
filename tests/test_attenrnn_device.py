"""Testes de posicionamento do estado oculto da Attention-RNN."""

from unittest.mock import MagicMock

import torch

from model.nlp.AttenRNN import AttentionRNN


def test_attention_rnn_hidden_state_uses_input_device():
    model = object.__new__(AttentionRNN)
    model.direction = 1
    model.num_layers = 1
    model.hidden_dim = 4
    config = MagicMock()
    config.get.return_value = "lstm"

    model.init_hidden(config, batch_size=2, gpu_list=[], device=torch.device("meta"))

    hidden_state, cell_state = model.hidden
    assert hidden_state.device.type == "meta"
    assert cell_state.device.type == "meta"