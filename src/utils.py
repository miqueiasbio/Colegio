"""Utilitários do projeto de preços de supermercado."""

from __future__ import annotations

import json
import logging
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logging básico para toda a aplicação."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return config


def ensure_dirs(*paths: str) -> None:
    """Garante que diretórios existem."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    """Define seed para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(value: Any) -> str:
    """Normaliza texto removendo acentos, espaços extras e padronizando caixa."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return text
