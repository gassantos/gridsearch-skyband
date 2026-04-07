"""
Helpers de execução de experimentos
====================================

Funções utilitárias, constantes e classes auxiliares usadas pelo
motor de execução de experimentos.

Autor: Gustavo Alexandre
"""

import configparser
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import PathManager

logger = logging.getLogger(__name__)

# Constantes de diretórios e arquivos
METRICS_DIR = PathManager.EXPERIMENTS_DIR / "metrics"
PathManager.ensure_dir(METRICS_DIR)

# Tarifa de energia (USD/kWh) — configurável via variável de ambiente
ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))


def now_iso() -> str:
    """Retorna o instante atual em formato ISO 8601 com timezone UTC."""
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str) -> configparser.ConfigParser:
    """Lê e retorna um ConfigParser padrão da stdlib a partir do arquivo.

    Args:
        path: Caminho para o arquivo ``.config``.

    Returns:
        ``configparser.ConfigParser`` com as seções e chaves carregadas.
    """
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def estimate_bert_flops(
    seq_len: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> float:
    """Estima os GFLOPs de uma passagem forward pelo BERT.

    Args:
        seq_len: Comprimento da sequência de tokens.
        hidden_size: Dimensão oculta do modelo. Padrão: 768 (BERT-base).
        num_layers: Número de camadas Transformer. Padrão: 12.
        num_heads: Número de cabeças de atenção. Padrão: 12.

    Returns:
        GFLOPs estimados por forward pass.
    """
    attention = (
        4 * seq_len * hidden_size * hidden_size +
        2 * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    )
    ffn = 8 * seq_len * hidden_size * hidden_size
    return num_layers * (attention + ffn) / 1e9  # GFLOPs


class TeeStream:
    """Stream que escreve simultaneamente no terminal e em um buffer em memória.

    Permite capturar o stdout do loop de treino sem perder a saída em
    tempo real. Implementa a interface mínima exigida pelo Python e pelo
    Transformers >= 5.x (``write``, ``flush``, ``isatty``, ``fileno``).
    """

    def __init__(self, original) -> None:
        self.original = original
        self.lines: list = []

    def write(self, text: str) -> None:
        """Escreve ``text`` no stream original e acrescenta à lista ``lines``."""
        self.original.write(text)
        self.lines.append(text)

    def flush(self) -> None:
        """Propaga flush para o stream original."""
        self.original.flush()

    def isatty(self) -> bool:
        """Delega ``isatty()`` ao stream original; retorna ``False`` se não implementado."""
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self) -> int:
        """Retorna o descritor de arquivo do stream original."""
        return self.original.fileno()
