"""Geração de relatórios CSV e gráficos para análise municipal."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data_prep import load_and_clean_data
from .features import add_price_features, apply_label_encoders
from .model import ForecastLSTM, PriceAutoencoder
from .train import FEATURE_COLUMNS
from .utils import ensure_dirs, load_config, normalize_text, setup_logging

LOGGER = logging.getLogger(__name__)


def inverse_target_scale(values: np.ndarray, scaler, feature_columns: list[str]) -> np.ndarray:
    idx = feature_columns.index("preco_normalizado")
    return values * scaler.scale_[idx] + scaler.mean_[idx]


def predict_product_series(df: pd.DataFrame, produto: str, model: ForecastLSTM, scaler, config: dict) -> pd.DataFrame:
    """Gera previsão de 7 dias para um produto usando última janela."""
    prod_df = df[df["produto"] == produto].copy()
    prod_daily = prod_df.sort_values("data").groupby("data", as_index=False).agg({col: "mean" for col in FEATURE_COLUMNS + ["preco_normalizado"]})
    prod_daily.loc[:, FEATURE_COLUMNS] = scaler.transform(prod_daily[FEATURE_COLUMNS])

    if len(prod_daily) < config["window_size"]:
        return pd.DataFrame()

    window = prod_daily[FEATURE_COLUMNS].tail(config["window_size"]).to_numpy(dtype=np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        pred_scaled = model(torch.tensor(window[None, :, :], dtype=torch.float32).to(device)).cpu().numpy()[0]

    pred = inverse_target_scale(pred_scaled, scaler, FEATURE_COLUMNS)
    start = prod_daily["data"].max() + pd.Timedelta(days=1)
    dates = [start + pd.Timedelta(days=i) for i in range(config["forecast_horizon"])]
    return pd.DataFrame({"data": dates, "produto": produto, "preco_previsto": pred})


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatórios e gráficos")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    ensure_dirs(config["report_dir"])

    with (Path(config["model_dir"]) / "artifacts.pkl").open("rb") as file:
        artifacts = pickle.load(file)

    df = load_and_clean_data(config["data_path"], config["municipio"])
    df = add_price_features(df)
    df = apply_label_encoders(df, artifacts["encoders"])

    scaler = artifacts["scaler"]
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

    # Previsões por produto
    forecast_frames = []
    for product in [normalize_text(p) for p in config.get("report_products", [])]:
        pred_df = predict_product_series(df, product, model, scaler, config)
        if not pred_df.empty:
            pred_df["municipio"] = normalize_text(config["municipio"])
            forecast_frames.append(pred_df)

            # gráfico real vs previsto
            hist = (
                df[df["produto"] == product]
                .sort_values("data")
                .groupby("data", as_index=False)["preco_normalizado"]
                .mean()
                .tail(30)
            )
            plt.figure(figsize=(8, 4))
            plt.plot(hist["data"], hist["preco_normalizado"], label="Real", marker="o")
            plt.plot(pred_df["data"], pred_df["preco_previsto"], label="Previsto", marker="x")
            plt.title(f"Série real vs prevista - {product}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.legend()
            plt.savefig(Path(config["report_dir"]) / f"serie_real_vs_prevista_{product.replace(' ', '_')}.png")
            plt.close()

    previsoes_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    previsoes_df.to_csv(Path(config["report_dir"]) / f"previsoes_{normalize_text(config['municipio'])}.csv", index=False)

    # anomalias
    ae = PriceAutoencoder(input_size=len(FEATURE_COLUMNS), hidden_size=config["autoencoder_hidden"]).to(device)
    ae.load_state_dict(torch.load(Path(config["model_dir"]) / "autoencoder.pt", map_location=device))
    ae.eval()
    arr = scaler.transform(df[FEATURE_COLUMNS]).astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(arr, dtype=torch.float32).to(device)
        rec = ae(x)
        scores = ((rec - x) ** 2).mean(dim=1).cpu().numpy()

    with (Path(config["model_dir"]) / "anomaly_threshold.json").open("r", encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]

    anomalies = df.copy()
    anomalies["anomaly_score"] = scores
    anomalies["is_anomaly"] = anomalies["anomaly_score"] > threshold
    anomalies.to_csv(Path(config["report_dir"]) / f"anomalias_{normalize_text(config['municipio'])}.csv", index=False)

    # variação por item e supermercado
    variacao = (
        df.sort_values("data")
        .groupby(["produto", "supermercado"])["preco_normalizado"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    variacao.to_csv(Path(config["report_dir"]) / f"variacao_item_supermercado_{normalize_text(config['municipio'])}.csv", index=False)

    # ranking mais caros/baratos por cesta
    basket = [normalize_text(x) for x in config.get("selected_basket", [])]
    basket_df = df[df["produto"].isin(basket)]
    ranking = basket_df.groupby("supermercado")["preco_normalizado"].mean().sort_values().reset_index(name="media_cesta")
    ranking.to_csv(Path(config["report_dir"]) / f"ranking_supermercados_{normalize_text(config['municipio'])}.csv", index=False)

    plt.figure(figsize=(8, 4))
    df.boxplot(column="preco_normalizado", by="supermercado", rot=45)
    plt.title("Boxplot de preços por supermercado")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(Path(config["report_dir"]) / f"boxplot_supermercado_{normalize_text(config['municipio'])}.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(ranking["supermercado"], ranking["media_cesta"])
    plt.xticks(rotation=45)
    plt.title("Ranking de supermercados por cesta")
    plt.tight_layout()
    plt.savefig(Path(config["report_dir"]) / f"ranking_cesta_{normalize_text(config['municipio'])}.png")
    plt.close()

    LOGGER.info("Relatórios gerados em %s", config["report_dir"])


if __name__ == "__main__":
    main()
