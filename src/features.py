"""Engenharia de atributos para previsão e anomalia."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas de preço para análise temporal."""
    df = df.copy()
    df["preco_normalizado"] = df["preco"] / df["quantidade"]

    group_cols = ["produto", "supermercado"]
    df = df.sort_values(["produto", "supermercado", "data"])

    df["ma_7"] = (
        df.groupby(group_cols)["preco_normalizado"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["ma_30"] = (
        df.groupby(group_cols)["preco_normalizado"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    )

    df["lag_7"] = df.groupby(group_cols)["preco_normalizado"].shift(7)
    df["var_pct_semana"] = ((df["preco_normalizado"] - df["lag_7"]) / df["lag_7"]).replace(
        [float("inf"), float("-inf")], 0.0
    )
    df["var_pct_semana"] = df["var_pct_semana"].fillna(0.0)

    df["dia_semana"] = df["data"].dt.dayofweek
    df["mes"] = df["data"].dt.month

    return df


def fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Treina encoders para colunas categóricas."""
    encoders: dict[str, LabelEncoder] = {}
    for col in ["supermercado", "categoria", "produto"]:
        encoder = LabelEncoder()
        df[f"{col}_id"] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Aplica encoders treinados e trata desconhecidos."""
    df = df.copy()
    for col, encoder in encoders.items():
        classes = set(encoder.classes_)
        mapped = df[col].astype(str).apply(lambda value: value if value in classes else encoder.classes_[0])
        df[f"{col}_id"] = encoder.transform(mapped)
    return df
