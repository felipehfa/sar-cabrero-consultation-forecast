"""
pipeline/forecasting.py

Genera la predicción futura (T+1..T+7) usando la información más reciente
disponible en el dataset engineered.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from . import config
from .io_utils import predict_ensemble
from .validation import validate_features
from .logging_utils import get_logger


def generate_forecast(
    df: pd.DataFrame,
    models_by_horizon: dict[int, list[xgb.Booster]],
    features_by_horizon: dict[int, list[str]],
) -> pd.DataFrame:
    """
    Genera el pronóstico para cada horizonte disponible, usando la última
    fila del dataset (la más reciente con features completas).

    Returns: DataFrame con columnas [horizonte, fecha_pronosticada,
    prediccion, modelo_utilizado], una fila por horizonte que pudo
    predecirse exitosamente.
    """
    logger = get_logger()

    last_date = df.index.max()
    last_row = df.loc[[last_date]]
    logger.info(f"Generando forecast a partir de la última fecha disponible: "
                f"{last_date.date()}")

    records = []

    for h in sorted(models_by_horizon.keys()):
        if h not in features_by_horizon:
            logger.warning(f"h={h}: modelo cargado pero sin lista de features "
                            f"disponible — se omite este horizonte.")
            continue

        expected_features = features_by_horizon[h]
        missing_in_df = [c for c in expected_features if c not in last_row.columns]
        if missing_in_df:
            logger.error(f"h={h}: el dataset engineered no tiene las columnas "
                         f"{missing_in_df} requeridas por este modelo — se omite.")
            continue

        X = last_row[expected_features]

        result = validate_features(X, expected_features, horizon=h)
        if not result.ok:
            logger.error(f"h={h}: validación falló, se omite la predicción "
                         f"de este horizonte. {result.summary()}")
            continue

        y_pred = predict_ensemble(models_by_horizon[h], X)
        forecast_date = config.forecast_date_for(last_date, h)

        records.append({
            "horizonte": h,
            "fecha_pronosticada": forecast_date,
            "prediccion": round(float(y_pred[0]), 4),
            "modelo_utilizado": f"{config.MODEL_NAME}_h{h}",
        })
        logger.info(f"h={h}: forecast para {forecast_date.date()} = "
                    f"{y_pred[0]:.2f}")

    columns = ["horizonte", "fecha_pronosticada", "prediccion", "modelo_utilizado"]

    if not records:
        logger.error("No se pudo generar forecast para ningún horizonte "
                     "(revisa los warnings/errores anteriores).")
        return pd.DataFrame(columns=columns)

    forecast_df = (
        pd.DataFrame(records)[columns]
        .sort_values("horizonte")
        .reset_index(drop=True)
    )
    return forecast_df
