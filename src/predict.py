"""Predição de curto prazo para um produto específico."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_prep import load_and_clean_data
from .features import add_price_features, apply_label_encoders
from .model import ForecastLSTM
from .train import FEATURE_COLUMNS
from .utils import load_config, normalize_text, setup_logging

LOGGER = logging.getLogger(__name__)


def inverse_target_scale(values: np.ndarray, scaler, feature_columns: list[str]) -> np.ndarray:
    """Inverte escalonamento da coluna alvo (preco_normalizado)."""
    idx = feature_columns.index("preco_normalizado")
    return values * scaler.scale_[idx] + scaler.mean_[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predição para produto")
    parser.add_argument("--config", required=True)
    parser.add_argument("--produto", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    with (Path(config["model_dir"]) / "artifacts.pkl").open("rb") as file:
        artifacts = pickle.load(file)

    df = load_and_clean_data(config["data_path"], config["municipio"])
    df = add_price_features(df)
    df = apply_label_encoders(df, artifacts["encoders"])

    produto_norm = normalize_text(args.produto)
    product_df = df[df["produto"] == produto_norm].copy()
    if product_df.empty:
        raise ValueError(f"Produto não encontrado no município: {args.produto}")

    product_daily = (
        product_df.sort_values("data")
        .groupby("data", as_index=False)
        .agg({col: "mean" for col in FEATURE_COLUMNS})
    )
    product_daily.loc[:, FEATURE_COLUMNS] = artifacts["scaler"].transform(product_daily[FEATURE_COLUMNS])

    if len(product_daily) < config["window_size"]:
        raise ValueError("Dados insuficientes para janela de previsão.")

    last_window = product_daily[FEATURE_COLUMNS].tail(config["window_size"]).to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["forecast_horizon"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(Path(config["model_dir"]) / "forecast_best.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(torch.tensor(last_window[None, :, :], dtype=torch.float32).to(device)).cpu().numpy()[0]

    pred_values = inverse_target_scale(pred_scaled, artifacts["scaler"], FEATURE_COLUMNS)

    start_date = product_daily["data"].max() + pd.Timedelta(days=1)
    pred_dates = [start_date + pd.Timedelta(days=i) for i in range(config["forecast_horizon"])]

    out_df = pd.DataFrame(
        {
            "data": pred_dates,
            "produto": produto_norm,
            "preco_normalizado_previsto": pred_values,
            "municipio": normalize_text(config["municipio"]),
        }
    )
    output_path = Path(config["report_dir"]) / f"previsao_{produto_norm.replace(' ', '_')}.csv"
    out_df.to_csv(output_path, index=False)
    LOGGER.info("Previsão salva em %s", output_path)


if __name__ == "__main__":
    main()
