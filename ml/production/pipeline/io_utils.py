"""
pipeline/io_utils.py

Carga de artefactos ya generados por los módulos existentes (modelos,
features seleccionadas, dataset engineered). No modifica ni reimplementa
la lógica de entrenamiento/selección — solo la consume.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from . import config
from .logging_utils import get_logger

# Reutiliza la predicción de ensamble ya implementada en model_xgboost.py
# (promedio de las N semillas, clippeado a >=0) — no se duplica lógica.
from model_xgboost import predict_ensemble  # noqa: F401  (re-exportado)


def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que `df` quede indexado por fecha (DatetimeIndex).

    run_feature_engineering() devuelve 'fecha' como columna normal (con
    índice entero por default), consistente con cómo exporta el CSV
    (index=False) — pero el resto del pipeline (y split_data() en
    feature_selection.py / model_xgboost.py) asume df.index como fecha.
    Se normaliza aquí una sola vez, sin tocar feature_engineering.py.
    """
    if config.DATE_COL in df.columns:
        df = df.copy()
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
        df = df.set_index(config.DATE_COL)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_engineered_dataset(rerun: bool = True) -> pd.DataFrame:
    """
    Obtiene el dataset con todas las features + target_h1..target_h7.

    rerun=True (default): ejecuta feature_engineering.run_feature_engineering()
    para asegurar que se está usando la información más reciente disponible
    (requisito del pipeline: "utilizando la información más reciente").

    rerun=False: lee directamente el CSV ya generado
    (data/processed/feature_engineering.csv), útil para desarrollo/debug
    cuando no se quiere re-ejecutar todo el feature engineering.
    """
    logger = get_logger()

    if rerun:
        logger.info("Ejecutando feature_engineering.run_feature_engineering()...")
        from feature_engineering import run_feature_engineering
        df = run_feature_engineering()
        df = normalize_date_index(df)
    else:
        path = config.PROCESSED_DIR / "feature_engineering.csv"
        logger.info(f"Leyendo dataset engineered ya existente: {path}")
        df = pd.read_csv(path, index_col=config.DATE_COL, parse_dates=True)

    df = df.sort_index()
    return df


def load_selected_features(horizon: int) -> list[str]:
    """Carga la lista de features seleccionadas para un horizonte dado."""
    path = config.features_path_for(horizon)
    return pd.read_csv(path)["feature"].tolist()


def load_model(horizon: int) -> list[xgb.Booster] | None:
    """
    Carga el ensamble (lista de Booster) de un horizonte. Devuelve None
    si el archivo no existe (el llamador decide cómo manejarlo — ver
    load_all_models, que valida y loguea faltantes).
    """
    path = config.model_path_for(horizon)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        models = pickle.load(f)
    return models


def load_all_models(horizons: list[int] | None = None) -> dict[int, list[xgb.Booster]]:
    """
    Carga los modelos de todos los horizontes pedidos. Valida que todos
    existan; si falta alguno, lo informa por logging y lo omite del dict
    devuelto (el resto del pipeline debe seguir funcionando con los
    horizontes disponibles).

    Returns: dict {horizon: [Booster, ...]} solo con los horizontes cuyo
    .pkl existe y se pudo cargar.
    """
    logger = get_logger()
    horizons = horizons if horizons is not None else config.HORIZONS

    models_by_horizon: dict[int, list[xgb.Booster]] = {}
    missing = []

    for h in horizons:
        models = load_model(h)
        if models is None:
            missing.append(h)
            logger.warning(f"Modelo faltante para horizonte h={h}: "
                            f"{config.model_path_for(h)} no existe.")
            continue
        models_by_horizon[h] = models
        logger.info(f"Modelo h={h} cargado ({len(models)} miembros del ensamble).")

    if missing:
        logger.warning(f"Horizontes sin modelo disponible: {missing}. "
                        f"El pipeline continuará solo con los horizontes "
                        f"cargados: {sorted(models_by_horizon.keys())}.")
    else:
        logger.info(f"Los {len(horizons)} modelos se cargaron correctamente.")

    return models_by_horizon


def load_all_features(horizons: list[int] | None = None) -> dict[int, list[str]]:
    """
    Carga las listas de features seleccionadas para todos los horizontes
    pedidos. Si falta el CSV de algún horizonte, lo informa y lo omite.
    """
    logger = get_logger()
    horizons = horizons if horizons is not None else config.HORIZONS

    features_by_horizon: dict[int, list[str]] = {}
    for h in horizons:
        path = config.features_path_for(h)
        if not path.exists():
            logger.warning(f"Archivo de features faltante para horizonte h={h}: {path}")
            continue
        features_by_horizon[h] = load_selected_features(h)

    return features_by_horizon
