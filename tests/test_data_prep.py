import pandas as pd
import pytest

from src.data_prep import validate_schema
from src.features import add_price_features


def test_validate_schema_missing_column() -> None:
    df = pd.DataFrame({"data": ["2025-01-01"], "municipio": ["irece"]})
    with pytest.raises(ValueError):
        validate_schema(df)


def test_preco_normalizado_conversion() -> None:
    df = pd.DataFrame(
        {
            "data": pd.to_datetime(["2025-01-01"]),
            "municipio": ["irece"],
            "supermercado": ["x"],
            "categoria": ["arroz"],
            "produto": ["arroz tipo 1 5kg"],
            "unidade": ["kg"],
            "quantidade": [5.0],
            "preco": [30.0],
            "promocao": [0],
        }
    )
    out = add_price_features(df)
    assert out.loc[0, "preco_normalizado"] == pytest.approx(6.0)
