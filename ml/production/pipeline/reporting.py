"""
pipeline/reporting.py

Guardado de todos los CSV de resultados y construcción del resumen
automático (mejor/peor horizonte, mejoras promedio, variable más
importante por horizonte, tiempos de ejecución).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .logging_utils import get_logger


def save_dataframe(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    get_logger().info(f"{label} guardado: {path} ({len(df)} filas)")


def build_summary(
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    backtesting_metrics_df: pd.DataFrame,
    top_feature_by_horizon: dict[int, str],
    timings: dict[str, float],
) -> dict:
    """
    Construye el resumen automático pedido: mejor/peor horizonte, mejores
    métricas, mejoras promedio vs cada naive, variable más importante por
    horizonte, y tiempos de ejecución.
    """
    summary: dict = {}

    if not metrics_df.empty:
        best_row = metrics_df.loc[metrics_df["mae"].idxmin()]
        worst_row = metrics_df.loc[metrics_df["mae"].idxmax()]
        summary["mejor_horizonte_mae"] = int(best_row["horizonte"])
        summary["peor_horizonte_mae"] = int(worst_row["horizonte"])
        summary["mejor_mae"] = float(metrics_df["mae"].min())
        summary["mejor_rmse"] = float(metrics_df["rmse"].min())
        summary["horizonte_mejor_rmse"] = int(
            metrics_df.loc[metrics_df["rmse"].idxmin(), "horizonte"]
        )
    else:
        summary.update({
            "mejor_horizonte_mae": None, "peor_horizonte_mae": None,
            "mejor_mae": None, "mejor_rmse": None, "horizonte_mejor_rmse": None,
        })

    if not comparison_df.empty:
        mae_rows = comparison_df[comparison_df["metrica"] == "mae"]
        summary["mejora_promedio_pct_vs_naive_semanal"] = (
            round(float(mae_rows["mejora_pct_vs_semanal"].mean()), 2)
            if not mae_rows.empty else None
        )
        summary["mejora_promedio_pct_vs_naive_anual"] = (
            round(float(mae_rows["mejora_pct_vs_anual"].mean()), 2)
            if not mae_rows.empty else None
        )
    else:
        summary["mejora_promedio_pct_vs_naive_semanal"] = None
        summary["mejora_promedio_pct_vs_naive_anual"] = None

    if not backtesting_metrics_df.empty:
        ml_bt = backtesting_metrics_df[backtesting_metrics_df["modelo"] == "ML"]
        if not ml_bt.empty:
            summary["backtesting_mae_promedio"] = round(float(ml_bt["mae"].mean()), 4)
            summary["backtesting_mejor_horizonte_mae"] = int(
                ml_bt.loc[ml_bt["mae"].idxmin(), "horizonte"]
            )

    summary["variable_mas_importante_por_horizonte"] = {
        str(h): feat for h, feat in sorted(top_feature_by_horizon.items())
    }

    summary["tiempos_segundos"] = {k: round(v, 3) for k, v in timings.items()}

    return summary


def save_summary(summary: dict, csv_path: Path, json_path: Path) -> None:
    """Guarda el resumen en JSON (estructura completa) y CSV (aplanado, para lectura rápida)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)
    get_logger().info(f"Resumen JSON guardado: {json_path}")

    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_rows.append({"clave": f"{key}.{sub_key}", "valor": sub_value})
        else:
            flat_rows.append({"clave": key, "valor": value})

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    get_logger().info(f"Resumen CSV guardado: {csv_path}")
