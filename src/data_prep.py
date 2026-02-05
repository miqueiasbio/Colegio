"""Carregamento, validação e limpeza de dados."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from .utils import normalize_text

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "data",
    "municipio",
    "supermercado",
    "categoria",
    "produto",
    "unidade",
    "quantidade",
    "preco",
]

OPTIONAL_COLUMNS = ["bairro", "marca", "promocao"]


def validate_schema(df: pd.DataFrame, required_columns: List[str] | None = None) -> None:
    """Valida colunas obrigatórias."""
    required = required_columns or REQUIRED_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")


def load_and_clean_data(csv_path: str, municipio: str) -> pd.DataFrame:
    """Carrega CSV, valida schema e limpa inconsistências de dados."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}") from exc

    validate_schema(df)

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    text_columns = ["municipio", "supermercado", "bairro", "categoria", "produto", "marca", "unidade"]
    for col in text_columns:
        df[col] = df[col].apply(normalize_text)

    municipio_norm = normalize_text(municipio)
    df = df[df["municipio"] == municipio_norm].copy()
    if df.empty:
        raise ValueError(f"Nenhum registro encontrado para município: {municipio}")

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    invalid_dates = df["data"].isna().sum()
    if invalid_dates > 0:
        LOGGER.warning("Removendo %s linhas com data inválida.", invalid_dates)
        df = df.dropna(subset=["data"])

    numeric_cols = ["quantidade", "preco", "promocao"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["promocao"] = df["promocao"].fillna(0).clip(0, 1)

    before = len(df)
    df = df.dropna(subset=["quantidade", "preco"])
    df = df[(df["quantidade"] > 0) & (df["preco"] > 0)]
    LOGGER.info("Removidas %s linhas inválidas de preço/quantidade.", before - len(df))

    df = df.sort_values("data").reset_index(drop=True)
    return df
