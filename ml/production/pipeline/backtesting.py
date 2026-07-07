"""
pipeline/backtesting.py

Simula el comportamiento operacional del pipeline: para cada domingo del
último año disponible, genera las predicciones T+1..T+7 usando solo
información conocida hasta ese domingo, y las compara contra los valores
reales que efectivamente ocurrieron.

Nota de diseño — equivalencia de truncar vs re-ejecutar feature engineering:
Todas las features del proyecto son causales (lags, medias/std móviles
`shift(1).rolling(...)`, o el valor del propio día que ya se conoce al
momento de predecir) — ninguna mira hacia adelante. Por construcción, el
valor de una feature en la fila de fecha D, calculado sobre el dataset
completo, es idéntico al que se obtendría re-ejecutando feature engineering
solo con datos hasta D. Truncar el dataset ya generado en cada fecha de
corte es entonces matemáticamente equivalente a re-ejecutar feature
engineering desde cero en cada domingo, pero muchísimo más rápido (una sola
pasada de feature engineering para todo el backtesting, en vez de ~52).

Si se necesita verificar esta equivalencia de forma literal (re-ejecutando
feature engineering sobre datos crudos truncados), activar
config.STRICT_RERUN_FEATURE_ENGINEERING = True — más lento y con
requerimientos adicionales sobre las rutas de datos crudos (ver
_rerun_feature_engineering_up_to). Por default, y salvo que se detecte un
problema real de equivalencia, se recomienda dejarlo en False.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config
from .io_utils import predict_ensemble
from .validation import validate_features
from .logging_utils import get_logger


def _get_sundays(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Todos los domingos dentro del último año disponible en `df`."""
    last_date = df.index.max()
    one_year_ago = last_date - pd.Timedelta(days=364)
    all_sundays = pd.date_range(start=one_year_ago, end=last_date, freq="W-SUN")
    return all_sundays


def _rerun_feature_engineering_up_to(execution_date: pd.Timestamp) -> pd.DataFrame:
    """
    [STRICT_RERUN_FEATURE_ENGINEERING] Re-ejecuta feature_engineering.py
    literalmente, pero solo con datos crudos hasta `execution_date`.

    No modifica feature_engineering.py: filtra los CSV crudos a copias
    temporales, sobreescribe temporalmente las rutas de entrada del módulo
    (monkeypatch de atributos, no del código fuente), corre
    run_feature_engineering(), y restaura las rutas originales al salir.

    Es ~50x más lento que la versión por truncado (ver docstring del
    módulo) — usar solo para verificación puntual.
    """
    import feature_engineering as fe_module

    original_clima = fe_module.INPUT_CLIMA
    original_consultas = fe_module.INPUT_CONSULTAS
    original_output = fe_module.OUTPUT_DATASET

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        clima_df = pd.read_csv(original_clima)
        clima_df["fecha_hora"] = pd.to_datetime(clima_df["fecha_hora"])
        clima_trunc_path = tmp_dir_path / "clima_trunc.csv"
        clima_df[clima_df["fecha_hora"] <= execution_date].to_csv(
            clima_trunc_path, index=False
        )

        cons_df = pd.read_csv(original_consultas)
        date_col = "Unnamed: 0" if "Unnamed: 0" in cons_df.columns else "fecha"
        cons_df[date_col] = pd.to_datetime(cons_df[date_col])
        cons_trunc_path = tmp_dir_path / "consultas_trunc.csv"
        cons_df[cons_df[date_col] <= execution_date].to_csv(
            cons_trunc_path, index=False
        )

        tmp_output_path = tmp_dir_path / "feature_engineering_trunc.csv"

        try:
            fe_module.INPUT_CLIMA = str(clima_trunc_path)
            fe_module.INPUT_CONSULTAS = str(cons_trunc_path)
            fe_module.OUTPUT_DATASET = str(tmp_output_path)
            df_trunc = fe_module.run_feature_engineering()
            from .io_utils import normalize_date_index
            df_trunc = normalize_date_index(df_trunc)
        finally:
            fe_module.INPUT_CLIMA = original_clima
            fe_module.INPUT_CONSULTAS = original_consultas
            fe_module.OUTPUT_DATASET = original_output

    return df_trunc


def run_backtesting(
    df: pd.DataFrame,
    models_by_horizon: dict[int, list[xgb.Booster]],
    features_by_horizon: dict[int, list[str]],
) -> pd.DataFrame:
    """
    Ejecuta el backtesting operacional completo: por cada domingo del
    último año, predice T+1..T+7 usando solo datos conocidos hasta ese
    domingo, y compara contra los valores reales observados.

    Returns: DataFrame con columnas [fecha_ejecucion, horizonte,
    fecha_pronosticada, valor_real, prediccion, error, error_absoluto,
    error_porcentual].
    """
    logger = get_logger()
    raw_series = df[config.RAW_TARGET_COL]
    last_date = df.index.max()

    sundays = _get_sundays(df)
    logger.info(f"Backtesting: {len(sundays)} domingos a simular entre "
                f"{sundays.min().date()} y {sundays.max().date()}.")

    records = []

    for execution_date in sundays:
        if config.STRICT_RERUN_FEATURE_ENGINEERING:
            try:
                df_trunc = _rerun_feature_engineering_up_to(execution_date)
            except Exception as e:
                logger.error(f"Backtesting {execution_date.date()}: falló la "
                             f"re-ejecución estricta de feature engineering "
                             f"({e}) — se usa el método por truncado como "
                             f"fallback.")
                df_trunc = df[df.index <= execution_date]
        else:
            df_trunc = df[df.index <= execution_date]

        if df_trunc.empty:
            logger.warning(f"Backtesting {execution_date.date()}: sin datos "
                            f"disponibles hasta esa fecha, se omite.")
            continue

        exec_last_date = df_trunc.index.max()
        last_row = df_trunc.loc[[exec_last_date]]

        for h in sorted(models_by_horizon.keys()):
            if h not in features_by_horizon:
                continue

            forecast_date = config.forecast_date_for(exec_last_date, h)

            if forecast_date > last_date:
                # Todavía no tenemos el valor real de esta fecha futura —
                # esperado en las últimas semanas del backtesting.
                continue

            actual = raw_series.get(forecast_date, np.nan)
            if pd.isna(actual):
                logger.warning(f"Backtesting {execution_date.date()} h={h}: "
                               f"sin valor real disponible en "
                               f"{forecast_date.date()}, se omite.")
                continue

            expected_features = features_by_horizon[h]
            missing_in_df = [c for c in expected_features if c not in last_row.columns]
            if missing_in_df:
                logger.error(f"Backtesting {execution_date.date()} h={h}: "
                             f"columnas faltantes {missing_in_df}, se omite.")
                continue

            X = last_row[expected_features]
            result = validate_features(X, expected_features, horizon=h)
            if not result.ok:
                logger.error(f"Backtesting {execution_date.date()} h={h}: "
                             f"validación falló. {result.summary()}")
                continue

            pred = float(predict_ensemble(models_by_horizon[h], X)[0])
            error = float(actual - pred)
            abs_error = abs(error)
            pct_error = (error / actual * 100.0) if actual != 0 else float("nan")

            records.append({
                "fecha_ejecucion": execution_date,
                "horizonte": h,
                "fecha_pronosticada": forecast_date,
                "valor_real": float(actual),
                "prediccion": round(pred, 4),
                "error": round(error, 4),
                "error_absoluto": round(abs_error, 4),
                "error_porcentual": round(pct_error, 4) if not np.isnan(pct_error) else np.nan,
            })

    backtest_df = pd.DataFrame(records)
    if not backtest_df.empty:
        backtest_df = backtest_df.sort_values(
            ["fecha_ejecucion", "horizonte"]
        ).reset_index(drop=True)

    logger.info(f"Backtesting completado: {len(backtest_df)} predicciones "
                f"generadas ({backtest_df['horizonte'].nunique() if not backtest_df.empty else 0} "
                f"horizontes, {backtest_df['fecha_ejecucion'].nunique() if not backtest_df.empty else 0} "
                f"fechas de ejecución).")

    return backtest_df


def compute_backtesting_metrics(backtest_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula MAE, RMSE, MAPE, sMAPE, R², Bias por horizonte, para el modelo
    ML y para los dos baselines naive, usando las mismas fechas de
    pronóstico que se usaron en el backtesting (comparación justa,
    "manzanas con manzanas").
    """
    from .metrics import compute_all_metrics
    from .naive_models import naive_weekly_predict, naive_annual_predict
    from .logging_utils import get_logger

    logger = get_logger()

    if backtest_df.empty:
        logger.warning("Backtesting vacío — no se pueden calcular métricas.")
        return pd.DataFrame()

    raw_series = df[config.RAW_TARGET_COL]
    rows = []

    for h, group in backtest_df.groupby("horizonte"):
        y_true = group["valor_real"].to_numpy()
        y_pred_ml = group["prediccion"].to_numpy()
        forecast_dates = pd.DatetimeIndex(group["fecha_pronosticada"])

        y_pred_weekly = naive_weekly_predict(raw_series, forecast_dates)
        y_pred_annual = naive_annual_predict(raw_series, forecast_dates)

        for modelo, y_pred in [
            ("ML", y_pred_ml),
            ("Naive_semanal", y_pred_weekly),
            ("Naive_anual", y_pred_annual),
        ]:
            m = compute_all_metrics(y_true, y_pred)
            rows.append({"horizonte": h, "modelo": modelo, **m})

    return pd.DataFrame(rows).sort_values(["horizonte", "modelo"]).reset_index(drop=True)
