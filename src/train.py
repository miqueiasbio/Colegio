"""Treina modelos de previsão e detecção de anomalias."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from .data_prep import load_and_clean_data
from .features import add_price_features, fit_label_encoders
from .model import ForecastLSTM, PriceAutoencoder, SequenceDataset
from .utils import ensure_dirs, load_config, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "preco_normalizado",
    "ma_7",
    "ma_30",
    "var_pct_semana",
    "dia_semana",
    "mes",
    "supermercado_id",
    "categoria_id",
    "produto_id",
    "promocao",
]


def temporal_split(df: pd.DataFrame, val_days: int, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realiza split temporal em treino/validação/teste."""
    max_date = df["data"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["data"] < val_start]
    val_df = df[(df["data"] >= val_start) & (df["data"] < test_start)]
    test_df = df[df["data"] >= test_start]

    return train_df, val_df, test_df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cria janelas de entrada e alvos multi-step para cada produto."""
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for product, group in df.groupby("produto"):
        product_daily = (
            group.sort_values("data")
            .groupby("data", as_index=False)
            .agg({col: "mean" for col in feature_cols})
        )

        values = product_daily[feature_cols].values
        target = product_daily["preco_normalizado"].values
        dates = product_daily["data"].values

        total = len(product_daily)
        for idx in range(window_size, total - horizon + 1):
            X_list.append(values[idx - window_size : idx])
            y_list.append(target[idx : idx + horizon])
            meta.append({"produto": product, "target_start": pd.to_datetime(dates[idx])})

    if not X_list:
        return np.empty((0, window_size, len(feature_cols))), np.empty((0, horizon)), []
    return np.stack(X_list), np.stack(y_list), meta


def split_sequences_by_date(
    X: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Segmenta sequências conforme data de início do alvo."""
    train_idx, val_idx, test_idx = [], [], []
    for i, item in enumerate(meta):
        date = item["target_start"]
        if date < val_start:
            train_idx.append(i)
        elif date < test_start:
            val_idx.append(i)
        else:
            test_idx.append(i)

    def make_dataset(indices: list[int]) -> SequenceDataset:
        return SequenceDataset(X[indices], y[indices]) if indices else SequenceDataset(np.empty((0, X.shape[1], X.shape[2])), np.empty((0, y.shape[1])))

    return make_dataset(train_idx), make_dataset(val_idx), make_dataset(test_idx)


def train_forecast_model(
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    input_size: int,
    config: dict[str, Any],
    model_dir: str,
) -> tuple[ForecastLSTM, float]:
    """Treina LSTM e salva melhor checkpoint por perda de validação."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastLSTM(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["forecast_horizon"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    best_val = float("inf")
    best_path = Path(model_dir) / "forecast_best.pt"

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        LOGGER.info("Epoch %s | train=%.4f val=%.4f", epoch + 1, mean_train, mean_val)

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_val


def train_autoencoder(train_array: np.ndarray, config: dict[str, Any], model_dir: str) -> tuple[PriceAutoencoder, float]:
    """Treina autoencoder com MSE de reconstrução e calcula limiar de anomalia."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PriceAutoencoder(input_size=train_array.shape[1], hidden_size=config["autoencoder_hidden"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    data_tensor = torch.tensor(train_array, dtype=torch.float32).to(device)
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        reconstructed = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            LOGGER.info("Autoencoder epoch %s loss=%.4f", epoch + 1, loss.item())

    model.eval()
    with torch.no_grad():
        rec = model(data_tensor)
        scores = ((rec - data_tensor) ** 2).mean(dim=1).cpu().numpy()

    threshold = float(np.percentile(scores, config["anomaly_percentile"]))
    torch.save(model.state_dict(), Path(model_dir) / "autoencoder.pt")

    with (Path(model_dir) / "anomaly_threshold.json").open("w", encoding="utf-8") as file:
        json.dump({"threshold": threshold}, file, indent=2)

    return model, threshold


def train_baseline_mae(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Baseline por média móvel de 7 dias (aproximação) para comparação."""
    if val_df.empty:
        return float("nan")
    temp = pd.concat([train_df, val_df]).sort_values("data")
    temp["baseline"] = temp.groupby("produto")["preco_normalizado"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    eval_df = temp[temp["data"].isin(val_df["data"])].dropna(subset=["baseline"])
    return mean_absolute_error(eval_df["preco_normalizado"], eval_df["baseline"])


def main() -> None:
    """Ponto de entrada do treinamento."""
    parser = argparse.ArgumentParser(description="Treinamento de previsão e anomalias")
    parser.add_argument("--config", required=True, help="Caminho para config JSON")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seed(config["random_seed"])

    ensure_dirs(config["model_dir"], config["report_dir"])

    df = load_and_clean_data(config["data_path"], config["municipio"])
    df = add_price_features(df)
    encoders = fit_label_encoders(df)

    train_df, val_df, test_df = temporal_split(df, config["val_days"], config["test_days"])
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split temporal resultou em conjunto vazio. Ajuste val_days/test_days ou dados.")

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    df.loc[:, FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    X_all, y_all, meta = build_sequences(df, FEATURE_COLUMNS, config["window_size"], config["forecast_horizon"])
    if len(X_all) == 0:
        raise ValueError("Não há sequências suficientes. Aumente dados ou reduza janela/horizonte.")

    val_start = val_df["data"].min()
    test_start = test_df["data"].min()
    train_dataset, val_dataset, _ = split_sequences_by_date(X_all, y_all, meta, val_start, test_start)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Sem sequências em treino/validação após split temporal.")

    forecast_model, best_val = train_forecast_model(
        train_dataset, val_dataset, input_size=len(FEATURE_COLUMNS), config=config, model_dir=config["model_dir"]
    )
    _ = forecast_model

    baseline_mae = train_baseline_mae(train_df, val_df)

    anomaly_train = scaler.transform(train_df[FEATURE_COLUMNS]).astype(np.float32)
    _, threshold = train_autoencoder(anomaly_train, config, config["model_dir"])

    with (Path(config["model_dir"]) / "artifacts.pkl").open("wb") as file:
        pickle.dump(
            {
                "encoders": encoders,
                "scaler": scaler,
                "feature_columns": FEATURE_COLUMNS,
                "municipio": config["municipio"],
            },
            file,
        )

    with (Path(config["model_dir"]) / "metrics_train.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_val_mae_forecast": best_val,
                "baseline_val_mae": baseline_mae,
                "anomaly_threshold": threshold,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
            },
            file,
            indent=2,
        )

    LOGGER.info("Treinamento concluído. Métricas salvas em models/metrics_train.json")


if __name__ == "__main__":
    main()
