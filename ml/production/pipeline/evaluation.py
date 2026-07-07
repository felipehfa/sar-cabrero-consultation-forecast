"""
pipeline/evaluation.py

Evaluación del modelo ML sobre el conjunto de test (mismo split de
model_xgboost.split_data) y comparación contra dos baselines naive
(semanal y anual).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config
from .io_utils import predict_ensemble
from .metrics import compute_all_metrics
from .naive_models import naive_weekly_predict, naive_annual_predict
from .validation import validate_features
from .logging_utils import get_logger

from model_xgboost import split_data  # reutilizado, no reimplementado

# Métricas donde un valor MENOR es mejor -> la mejora % se calcula como
# (naive - ml) / |naive| * 100.
_LOWER_IS_BETTER = {"mae", "rmse", "mape", "smape"}


def _evaluate_horizon(
    df: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    models: list[xgb.Booster],
    features: list[str],
    raw_series: pd.Series,
) -> dict | None:
    """
    Evalúa un único horizonte sobre el conjunto de test. Devuelve None si
    la validación de features falla (se loguea el motivo) o si no quedan
    filas de test con target válido para este horizonte.
    """
    logger = get_logger()
    target_col = config.target_col_for(horizon)

    test_h = test.dropna(subset=[target_col])
    if test_h.empty:
        logger.warning(f"h={horizon}: sin filas de test con target válido, "
                        f"se omite la evaluación de este horizonte.")
        return None

    missing_in_df = [c for c in features if c not in test_h.columns]
    if missing_in_df:
        logger.error(f"h={horizon}: columnas {missing_in_df} no están en el "
                     f"dataset — se omite la evaluación.")
        return None

    X_test = test_h[features]
    y_true = test_h[target_col].to_numpy()

    result = validate_features(X_test, features, horizon=horizon)
    if not result.ok:
        logger.error(f"h={horizon}: validación falló al evaluar. {result.summary()}")
        return None

    y_pred_ml = predict_ensemble(models, X_test)

    # Fecha real que representa cada target_h{h} en esta fila (ver
    # convención en config.forecast_date_for): row_date + (h+1).
    forecast_dates = test_h.index + pd.Timedelta(days=horizon + 1)

    y_pred_weekly = naive_weekly_predict(raw_series, forecast_dates)
    y_pred_annual = naive_annual_predict(raw_series, forecast_dates)

    return {
        "horizon": horizon,
        "dates": forecast_dates,
        "y_true": y_true,
        "y_pred_ml": y_pred_ml,
        "y_pred_weekly": y_pred_weekly,
        "y_pred_annual": y_pred_annual,
        "metrics_ml":     compute_all_metrics(y_true, y_pred_ml),
        "metrics_weekly": compute_all_metrics(y_true, y_pred_weekly),
        "metrics_annual": compute_all_metrics(y_true, y_pred_annual),
    }


def run_evaluation(
    df: pd.DataFrame,
    models_by_horizon: dict[int, list[xgb.Booster]],
    features_by_horizon: dict[int, list[str]],
) -> dict:
    """
    Evalúa todos los horizontes disponibles sobre el conjunto de test y
    construye:
        - metrics_df:    métricas del modelo ML por horizonte (para
                          results/metrics.csv)
        - comparison_df: comparación ML vs Naive semanal vs Naive anual,
                          por horizonte y por métrica, con mejora % (para
                          results/comparison.csv)

    Returns: dict con metrics_df, comparison_df, y raw_results (detalle
    por horizonte, útil para plots posteriores sin tener que recalcular).
    """
    logger = get_logger()
    _, _, test = split_data(df)
    raw_series = df[config.RAW_TARGET_COL]

    raw_results = {}
    for h in sorted(models_by_horizon.keys()):
        if h not in features_by_horizon:
            continue
        res = _evaluate_horizon(
            df, test, h, models_by_horizon[h], features_by_horizon[h], raw_series
        )
        if res is not None:
            raw_results[h] = res
            logger.info(f"h={h}: evaluación OK (n={res['metrics_ml']['n']}) "
                        f"MAE_ml={res['metrics_ml']['mae']}")

    metrics_rows = []
    comparison_rows = []

    for h, res in raw_results.items():
        m_ml = res["metrics_ml"]
        m_wk = res["metrics_weekly"]
        m_an = res["metrics_annual"]

        metrics_rows.append({"horizonte": h, **m_ml})

        for metric_name in ["mae", "rmse", "mape", "smape", "r2", "bias"]:
            ml_val = m_ml[metric_name]
            wk_val = m_wk[metric_name]
            an_val = m_an[metric_name]

            mejora_wk = _improvement_pct(metric_name, ml_val, wk_val)
            mejora_an = _improvement_pct(metric_name, ml_val, an_val)

            comparison_rows.append({
                "horizonte": h,
                "metrica": metric_name,
                "modelo_ml": ml_val,
                "naive_semanal": wk_val,
                "naive_anual": an_val,
                "mejora_pct_vs_semanal": mejora_wk,
                "mejora_pct_vs_anual": mejora_an,
            })

    metrics_columns = ["horizonte", "mae", "rmse", "mape", "smape", "r2", "bias", "n"]
    comparison_columns = [
        "horizonte", "metrica", "modelo_ml", "naive_semanal", "naive_anual",
        "mejora_pct_vs_semanal", "mejora_pct_vs_anual",
    ]

    if not metrics_rows:
        logger.error("No se pudo evaluar ningún horizonte (revisa los "
                     "warnings/errores anteriores).")
        metrics_df = pd.DataFrame(columns=metrics_columns)
        comparison_df = pd.DataFrame(columns=comparison_columns)
    else:
        metrics_df = (
            pd.DataFrame(metrics_rows)[metrics_columns]
            .sort_values("horizonte")
            .reset_index(drop=True)
        )
        comparison_df = (
            pd.DataFrame(comparison_rows)[comparison_columns]
            .sort_values(["horizonte", "metrica"])
            .reset_index(drop=True)
        )

    return {
        "metrics_df": metrics_df,
        "comparison_df": comparison_df,
        "raw_results": raw_results,
    }


def _improvement_pct(metric_name: str, ml_val: float, naive_val: float) -> float:
    """
    Mejora porcentual del modelo ML respecto a un baseline naive, para una
    métrica dada.

    - Para métricas donde menor es mejor (mae, rmse, mape, smape):
          mejora% = (naive - ml) / |naive| * 100
      (positivo = el ML es mejor que el naive)

    - Para r2 (mayor es mejor): se reporta la diferencia en puntos
      (ml - naive), no un porcentaje — dividir por un r2 que puede ser
      negativo o cercano a 0 produce números sin sentido interpretativo.

    - Para bias (más cercano a 0 es mejor): se reporta la reducción en
      magnitud absoluta del sesgo (|naive| - |ml|), no un porcentaje del
      valor con signo.
    """
    if np.isnan(ml_val) or np.isnan(naive_val):
        return float("nan")

    if metric_name in _LOWER_IS_BETTER:
        if naive_val == 0:
            return float("nan")
        return round(100.0 * (naive_val - ml_val) / abs(naive_val), 2)

    if metric_name == "r2":
        return round(ml_val - naive_val, 4)

    if metric_name == "bias":
        return round(abs(naive_val) - abs(ml_val), 4)

    return float("nan")
