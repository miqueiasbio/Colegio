"""Avaliação dos modelos treinados."""

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
from sklearn.metrics import mean_absolute_error

from .data_prep import load_and_clean_data
from .features import add_price_features, apply_label_encoders
from .model import ForecastLSTM, PriceAutoencoder
from .train import FEATURE_COLUMNS, build_sequences, split_sequences_by_date, temporal_split
from .utils import ensure_dirs, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação dos modelos")
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
    df.loc[:, FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    _, val_df, test_df = temporal_split(df, config["val_days"], config["test_days"])
    X_all, y_all, meta = build_sequences(df, FEATURE_COLUMNS, config["window_size"], config["forecast_horizon"])
    _, _, test_dataset = split_sequences_by_date(X_all, y_all, meta, val_df["data"].min(), test_df["data"].min())
    if len(test_dataset) == 0:
        raise ValueError("Sem sequências para avaliação de teste.")

    X_test = test_dataset.X.numpy()
    y_test = test_dataset.y.numpy()
    test_meta = [m for m in meta if m["target_start"] >= test_df["data"].min()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecast_model = ForecastLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["forecast_horizon"],
        dropout=config["dropout"],
    ).to(device)
    forecast_model.load_state_dict(torch.load(Path(config["model_dir"]) / "forecast_best.pt", map_location=device))
    forecast_model.eval()

    with torch.no_grad():
        pred = forecast_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_true = y_test[:, 0]
    y_pred = pred[:, 0]

    eval_rows = []
    for idx, info in enumerate(test_meta):
        eval_rows.append({"produto": info["produto"], "y_true": y_true[idx], "y_pred": y_pred[idx]})
    eval_df = pd.DataFrame(eval_rows)

    product_metrics = (
        eval_df.groupby("produto")
        .apply(
            lambda g: pd.Series(
                {
                    "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "mape": mean_absolute_percentage_error(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                }
            )
        )
        .reset_index()
    )

    ae = PriceAutoencoder(input_size=len(FEATURE_COLUMNS), hidden_size=config["autoencoder_hidden"]).to(device)
    ae.load_state_dict(torch.load(Path(config["model_dir"]) / "autoencoder.pt", map_location=device))
    ae.eval()

    test_rows = df[df["data"] >= test_df["data"].min()].copy()
    test_array = test_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    tensor = torch.tensor(test_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        rec = ae(tensor)
        scores = ((rec - tensor) ** 2).mean(dim=1).cpu().numpy()

    with (Path(config["model_dir"]) / "anomaly_threshold.json").open("r", encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]

    anomaly_df = test_rows.copy()
    anomaly_df["anomaly_score"] = scores
    anomaly_df["is_anomaly"] = anomaly_df["anomaly_score"] > threshold

    product_metrics.to_csv(Path(config["report_dir"]) / f"metricas_previsao_{config['municipio']}.csv", index=False)
    anomaly_df.sort_values("anomaly_score", ascending=False).head(20).to_csv(
        Path(config["report_dir"]) / f"top_anomalias_{config['municipio']}.csv", index=False
    )

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=15, alpha=0.7)
    plt.axvline(threshold, color="red", linestyle="--", label="Limiar")
    plt.title("Distribuição de score de anomalia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config["report_dir"]) / f"distribuicao_anomalia_{config['municipio']}.png")
    plt.close()

    LOGGER.info("Avaliação concluída. Arquivos em %s", config["report_dir"])


if __name__ == "__main__":
    main()
