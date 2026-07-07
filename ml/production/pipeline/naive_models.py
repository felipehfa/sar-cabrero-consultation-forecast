"""
pipeline/naive_models.py

Modelos baseline "naive" contra los que se compara el modelo ML:

    - Naive semanal: predicción(t) = valor observado en t-7
    - Naive anual:   predicción(t) = valor observado en t-365
                      (t-366 si el año de referencia es bisiesto y la
                      fecha cae después del 29 de febrero de ese año —
                      manejado vía aritmética calendario, no offset fijo,
                      para que sea correcto en ambos sentidos)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lookup_series(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Busca los valores de `series` (indexada por fecha) en las fechas
    pedidas. Si una fecha exacta no existe en el índice (hueco en la
    serie), cae al valor disponible más cercano hacia atrás (ffill).
    Devuelve NaN si no hay ningún valor disponible antes de esa fecha.

    Nota: `dates` puede traer valores repetidos de forma legítima — por
    ejemplo, el 28 y el 29 de febrero de un año bisiesto ambos mapean al
    "28 de febrero" del año anterior cuando ese año no es bisiesto (ver
    naive_annual_predict). pd.Index.union() NO deduplica el operando que
    recibe si éste ya trae duplicados, así que se deduplica `dates` ANTES
    de construir la unión para el ffill, y se usa el `dates` original
    (con sus posibles repetidos) solo en el reindex final — eso sí está
    soportado por pandas cuando la fuente es única.
    """
    series = series.sort_index()
    dates = pd.DatetimeIndex(dates)
    unique_dates = dates.unique()
    reindexed = series.reindex(series.index.union(unique_dates)).sort_index()
    reindexed = reindexed.ffill()
    return reindexed.reindex(dates).to_numpy()


def naive_weekly_predict(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Predicción naive semanal para cada fecha en `dates`: valor observado
    7 días antes.
    """
    shifted_dates = pd.DatetimeIndex(dates) - pd.Timedelta(days=7)
    return _lookup_series(series, shifted_dates)


def naive_annual_predict(series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Predicción naive anual para cada fecha en `dates`: valor observado el
    mismo día del año anterior. Usa aritmética de calendario
    (DateOffset(years=1)), que maneja correctamente 29 de febrero cayendo
    en un fallback al 28 de febrero cuando el año de referencia no es
    bisiesto.
    """
    dates = pd.DatetimeIndex(dates)
    shifted = []
    for d in dates:
        try:
            shifted.append(d - pd.DateOffset(years=1))
        except ValueError:
            # 29 de febrero sin equivalente en el año de referencia
            shifted.append(pd.Timestamp(d.year - 1, 2, 28))
    shifted_dates = pd.DatetimeIndex(shifted)
    return _lookup_series(series, shifted_dates)


def naive_value_at(series: pd.Series, target_date, kind: str) -> float:
    """
    Devuelve un único valor naive (semanal o anual) para `target_date`.
    Usado al generar el forecast puntual (no la evaluación en batch).

    kind: "weekly" o "annual"
    """
    target_date = pd.Timestamp(target_date)
    if kind == "weekly":
        arr = naive_weekly_predict(series, pd.DatetimeIndex([target_date]))
    elif kind == "annual":
        arr = naive_annual_predict(series, pd.DatetimeIndex([target_date]))
    else:
        raise ValueError(f"kind debe ser 'weekly' o 'annual', recibido: {kind}")
    return float(arr[0]) if len(arr) else float("nan")
