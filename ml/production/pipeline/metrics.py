"""
pipeline/metrics.py

Cálculo de métricas de evaluación. Reutiliza smape() ya implementado en
model_xgboost.py (no se duplica) y añade MAPE, R² y Bias, que no existían
ahí.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from model_xgboost import smape as _smape_impl
except ImportError:  # pragma: no cover - fallback si el módulo no está en path
    def _smape_impl(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(100.0 * np.mean(
            2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
        ))


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_actual: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error.

    Los puntos donde |y_true| < min_actual se EXCLUYEN del cálculo, en vez
    de forzarles un piso artificial en el denominador (ej. eps=1e-6). Con
    datos de conteo (consultas por día), un valor real de 0 es genuino, no
    ruido de redondeo — dividir por un eps casi nulo dispara el error a
    millones de por ciento aunque el error absoluto sea perfectamente
    razonable (se observó esto en producción: MAPE=82,047,159% en un
    horizonte con sMAPE=21% en la misma fila, por un único día real=0 en
    el backtesting).

    sMAPE (que sí maneja bien los ceros, por diseño) es la métrica de
    referencia cuando MAPE se vuelve poco informativo — ver smape() /
    compute_all_metrics().

    Returns NaN si todos los puntos quedan excluidos (ningún |y_true| >= min_actual).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= min_actual
    if not np.any(mask):
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias (Mean Error) = mean(y_pred - y_true).

    Positivo → el modelo sobreestima en promedio.
    Negativo → el modelo subestima en promedio.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R². Con muestras pequeñas (ej. backtesting por horizonte con pocas
    observaciones) puede ser inestable o incluso negativo — es esperable,
    no es un bug.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


from .logging_utils import get_logger


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula el set completo de métricas pedido: MAE, RMSE, MAPE, sMAPE,
    R², Bias. y_pred se clippea a >= 0 antes de calcular (consultas no
    pueden ser negativas), consistente con el resto del pipeline.

    Filtra pares con NaN en y_true o y_pred antes de calcular — esto pasa
    legítimamente, por ejemplo, cuando el baseline naive anual no tiene
    dato disponible 365 días atrás (extremo inicial de la serie). Si se
    descarta algún par, se informa por logging con el conteo.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    n_discarded = int((~valid_mask).sum())
    if n_discarded > 0:
        get_logger().warning(
            f"compute_all_metrics: se descartaron {n_discarded} de "
            f"{len(y_true)} pares por NaN (ej. naive anual sin dato "
            f"disponible 365 días atrás)."
        )
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            "mae": float("nan"), "rmse": float("nan"), "mape": float("nan"),
            "smape": float("nan"), "r2": float("nan"), "bias": float("nan"),
            "n": 0,
        }

    return {
        "mae":   round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse":  round(rmse(y_true, y_pred), 4),
        "mape":  round(mape(y_true, y_pred), 4),
        "smape": round(_smape_impl(y_true, y_pred), 4),
        "r2":    round(r_squared(y_true, y_pred), 4) if not np.isnan(r_squared(y_true, y_pred)) else float("nan"),
        "bias":  round(bias(y_true, y_pred), 4),
        "n":     int(len(y_true)),
    }