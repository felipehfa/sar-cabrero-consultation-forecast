"""
pipeline/plotting.py

Generación de todos los gráficos del reporte. Usa matplotlib puro (sin
seaborn) para minimizar dependencias. Cada función guarda su propio PNG y
cierra la figura al terminar (evita acumular memoria al generar decenas
de gráficos en una sola corrida).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo — necesario para correr sin display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .logging_utils import get_logger

_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_SQUARE = (7, 7)
_DPI = 120


def _save_and_close(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    get_logger().info(f"Gráfico guardado: {path}")


# ──────────────────────────────────────────────────────────────────────────
# Serie temporal real vs predicción, por horizonte
# ──────────────────────────────────────────────────────────────────────────

def plot_horizon_series(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    output_path: Path,
) -> None:
    """
    Serie real vs predicha para un horizonte, con el error absoluto en un
    panel inferior.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_true - y_pred)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(dates, y_true, label="Real", color="#1f77b4", linewidth=1.5)
    ax1.plot(dates, y_pred, label="Predicción", color="#d62728",
              linewidth=1.2, linestyle="--")
    ax1.set_title(f"Consultas respiratorias — Real vs Predicción (t+{horizon})")
    ax1.set_ylabel("Consultas")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.fill_between(dates, abs_error, color="#7f7f7f", alpha=0.6)
    ax2.set_ylabel("Error absoluto")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.3)

    fig.autofmt_xdate()
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Distribución de errores
# ──────────────────────────────────────────────────────────────────────────

def plot_error_histogram(errors: np.ndarray, horizon: int | str, output_path: Path) -> None:
    errors = np.asarray(errors, dtype=float)
    errors = errors[~np.isnan(errors)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Distribución de errores — {horizon}")
    ax.set_xlabel("Error (real - predicción)")
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.3)
    _save_and_close(fig, output_path)


def plot_error_boxplot_by_horizon(errors_by_horizon: dict[int, np.ndarray], output_path: Path) -> None:
    """Boxplot del error por horizonte, uno al lado del otro."""
    horizons = sorted(errors_by_horizon.keys())
    data = [errors_by_horizon[h][~np.isnan(errors_by_horizon[h])] for h in horizons]

    fig, ax = plt.subplots(figsize=(9, 5))
    # No se pasa `labels`/`tick_labels` a boxplot() — ese parámetro cambió
    # de nombre entre versiones de matplotlib (labels -> tick_labels desde
    # 3.9, y versiones más nuevas ya no aceptan `labels`). boxplot() ubica
    # las cajas en posiciones enteras 1..N por defecto en cualquier
    # versión, así que las etiquetas se asignan manualmente después.
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(horizons) + 1))
    ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Distribución del error por horizonte")
    ax.set_xlabel("Horizonte")
    ax.set_ylabel("Error (real - predicción)")
    ax.grid(alpha=0.3, axis="y")
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Scatter predicción vs real
# ──────────────────────────────────────────────────────────────────────────

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, horizon: int | str, output_path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    ax.scatter(y_true, y_pred, alpha=0.5, s=25, color="#4c72b0", edgecolors="none")

    lims = [
        min(y_true.min(), y_pred.min()) if len(y_true) else 0,
        max(y_true.max(), y_pred.max()) if len(y_true) else 1,
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Predicción")
    ax.set_title(f"Predicción vs Real — {horizon}")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Heatmap de error (fecha de ejecución x horizonte)
# ──────────────────────────────────────────────────────────────────────────

def plot_error_heatmap(backtest_df: pd.DataFrame, output_path: Path,
                        value_col: str = "error_absoluto") -> None:
    """
    Heatmap del error de backtesting: filas = fecha de ejecución,
    columnas = horizonte.
    """
    if backtest_df.empty:
        get_logger().warning("Heatmap: backtest_df vacío, se omite el gráfico.")
        return

    pivot = backtest_df.pivot_table(
        index="fecha_ejecucion", columns="horizonte", values=value_col, aggfunc="mean"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8, max(6, 0.18 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"t+{h}" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.strftime("%Y-%m-%d") for d in pivot.index], fontsize=6)

    ax.set_xlabel("Horizonte")
    ax.set_ylabel("Fecha de ejecución")
    ax.set_title(f"Heatmap de {value_col} — Backtesting operacional")

    fig.colorbar(im, ax=ax, label=value_col)
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Comparación de evolución del error: ML vs Naive semanal vs Naive anual
# ──────────────────────────────────────────────────────────────────────────

def plot_naive_comparison(backtesting_metrics_df: pd.DataFrame, output_path: Path,
                           metric: str = "mae") -> None:
    """
    Compara la métrica elegida (default MAE) entre ML, Naive semanal y
    Naive anual, para cada horizonte — un gráfico de líneas con 3 series.
    """
    if backtesting_metrics_df.empty:
        get_logger().warning("Comparación naive: métricas de backtesting vacías, se omite.")
        return

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    for modelo, color, style in [
        ("ML", "#1f77b4", "-o"),
        ("Naive_semanal", "#d62728", "--s"),
        ("Naive_anual", "#2ca02c", ":^"),
    ]:
        sub = backtesting_metrics_df[backtesting_metrics_df["modelo"] == modelo].sort_values("horizonte")
        if sub.empty:
            continue
        ax.plot(sub["horizonte"], sub[metric], style, color=color, label=modelo)

    ax.set_xlabel("Horizonte (días)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} por horizonte — ML vs Naive (backtesting)")
    ax.set_xticks(sorted(backtesting_metrics_df["horizonte"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Gráfico operacional — el más importante del reporte
# ──────────────────────────────────────────────────────────────────────────

def plot_operational_forecast(
    raw_series: pd.Series,
    backtest_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Serie real completa (último año) como línea principal, con las 7
    predicciones (T+1..T+7) de cada domingo de ejecución superpuestas como
    trazos cortos y semi-transparentes — permite ver visualmente cómo
    habría pronosticado el modelo semana a semana.
    """
    if backtest_df.empty:
        get_logger().warning("Gráfico operacional: backtest_df vacío, se omite.")
        return

    last_date = raw_series.index.max()
    window_start = backtest_df["fecha_ejecucion"].min()
    real_window = raw_series[(raw_series.index >= window_start) & (raw_series.index <= last_date)]

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(real_window.index, real_window.to_numpy(),
            color="black", linewidth=2, label="Real", zorder=3)

    n_exec = backtest_df["fecha_ejecucion"].nunique()
    cmap = plt.colormaps.get_cmap("viridis")

    for i, (exec_date, group) in enumerate(backtest_df.groupby("fecha_ejecucion")):
        group = group.sort_values("horizonte")
        color = cmap(i / max(n_exec - 1, 1))
        ax.plot(
            group["fecha_pronosticada"], group["prediccion"],
            marker="o", markersize=3, linewidth=1, alpha=0.55,
            color=color, zorder=2,
        )

    ax.set_title("Pronóstico operacional semanal (T+1..T+7) vs Real — último año")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Consultas respiratorias")
    ax.grid(alpha=0.3)

    # Leyenda simplificada (no una entrada por cada una de las ~52 semanas)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2, label="Real"),
        Line2D([0], [0], color=cmap(0.5), linewidth=1, marker="o",
               markersize=3, alpha=0.7, label="Pronóstico semanal (T+1..T+7)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.autofmt_xdate()
    _save_and_close(fig, output_path)


# ──────────────────────────────────────────────────────────────────────────
# Importancia de variables
# ──────────────────────────────────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, horizon: int, output_path: Path,
                             top_n: int = 15) -> None:
    """
    Gráfico tipo Pareto: barras con las top_n variables más importantes
    (importancia %) y línea de importancia acumulada sobre un eje
    secundario.
    """
    if fi_df.empty:
        get_logger().warning(f"h={horizon}: sin importancia de variables, se omite el gráfico.")
        return

    # Mismo slice y mismo orden para las barras y la línea de acumulada —
    # invertido una sola vez (iloc[::-1]) para que la variable más
    # importante quede arriba en el barh. Usar dos ordenamientos distintos
    # aquí desalinea la línea respecto a las barras.
    top = fi_df.head(top_n).iloc[::-1].reset_index(drop=True)
    positions = range(len(top))

    fig, ax1 = plt.subplots(figsize=(9, max(5, 0.35 * len(top))))
    ax1.barh(top["feature"], top["importance_pct"], color="#4c72b0")
    ax1.set_xlabel("Importancia (%)")
    ax1.set_title(f"Importancia de variables — horizonte t+{horizon}")

    ax2 = ax1.twiny()
    ax2.plot(top["cumulative_pct"], positions, color="#d62728",
             marker="o", markersize=3, linewidth=1.2)
    ax2.set_xlabel("Importancia acumulada (%)")
    ax2.set_xlim(0, 105)

    _save_and_close(fig, output_path)