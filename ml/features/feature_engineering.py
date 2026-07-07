# ml/features/02_feature_engineering.py

from pathlib import Path

import numpy as np
import pandas as pd
import holidays

# =============================================================================
# CONFIG
# =============================================================================
#
# [MULTI-HORIZONTE] Este script corre UNA sola vez (no uno por horizonte),
# porque las features (clima, calendario, atenciones) son idénticas sin
# importar a cuántos días se quiera predecir — solo el TARGET depende del
# horizonte. En vez de generar 7 archivos casi idénticos, se genera un único
# feature_engineering.csv con una columna de target POR CADA horizonte
# (target_h1 ... target_h{max(HORIZONS)}), y cada script downstream
# (feature_selection.py, model_xgboost.py) selecciona la columna
# target_h{horizon} que le corresponde.
#
# Convención de shift (heredada del diseño original): el horizonte
# OPERACIONAL h (ej. "predecir t+1") corresponde a shift(-(h+1)) sobre
# total_atenciones, porque al correr el pipeline en la mañana de t+1 solo
# se conoce t completo — el mismo razonamiento que ya estaba documentado
# para TARGET_HORIZON=4 → "predecir t+3".

INPUT_CLIMA = "data/raw/datos_clima.csv"
INPUT_CONSULTAS = "data/processed/datos_consultas_corregidos.csv"

OUTPUT_DATASET = "data/processed/feature_engineering.csv"

# Horizontes operacionales para los que se genera una columna target_h{h}.
HORIZONS = list(range(1, 8))   # 1..7 días

ATENCIONES_LONG_WINDOW = 90
EPS_BASELINE = 1e-3   # evita división por cero en días de consulta muy baja

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_season(month):
    """
    Estaciones hemisferio sur.
    """
    if month in [12, 1, 2]:
        return "verano"
    elif month in [3, 4, 5]:
        return "otono"
    elif month in [6, 7, 8]:
        return "invierno"
    else:
        return "primavera"


def add_lags(df, cols, lags):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_moving_averages(df, cols, windows):
    for col in cols:
        for w in windows:
            df[f"{col}_ma{w}"] = (
                df[col]
                .shift(1)
                .rolling(w)
                .mean()
            )
    return df


def add_moving_std(df, cols, windows):
    for col in cols:
        for w in windows:
            df[f"{col}_std{w}"] = (
                df[col]
                .shift(1)
                .rolling(w)
                .std()
            )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal — encapsulado en función para poder importarlo desde
# otro script. No necesita serialización (pickle): produce un CSV plano
# que los scripts downstream leen directamente con pd.read_csv().
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(horizons: list = None) -> pd.DataFrame:
    """
    Corre el pipeline de feature engineering UNA sola vez y genera un único
    dataset con una columna target_h{h} por cada horizonte en `horizons`
    (default: HORIZONS = 1..7).

    A diferencia de feature_selection.py y model_xgboost.py, este script no
    necesita un parámetro `horizon` que cambie el comportamiento por corrida
    — las features son las mismas para todos los horizontes, solo cambia
    qué tan adelante mira cada columna target.

    Para usarlo desde un script driver:

        import importlib
        fe = importlib.import_module("02_feature_engineering")
        df = fe.run_feature_engineering()   # corre una sola vez

        fs = importlib.import_module("03_feature_selection")
        import model_xgboost as mx
        for h in range(1, 8):
            fs.run_feature_selection(df.rename(columns={f"target_h{h}": "target"}),
                                      horizon=h)
            ...

    (Ver nota al final sobre el nombre de columna target_h{h} vs "target"
    que usan actualmente feature_selection.py / model_xgboost.py.)

    Returns: el DataFrame final ya limpio de NaNs en las features (los
    target_h{h} conservan sus NaN naturales de cola, uno por horizonte —
    ver sección LIMPIEZA FINAL).
    """
    horizons = horizons if horizons is not None else HORIZONS

    # =========================================================================
    # CARGA CLIMA
    # =========================================================================

    print("Cargando datos clima...")

    df_clima = pd.read_csv(INPUT_CLIMA)

    df_clima["fecha_hora"] = pd.to_datetime(df_clima["fecha_hora"])
    df_clima["fecha"] = df_clima["fecha_hora"].dt.floor("D")

    # =========================================================================
    # AGREGACIÓN HORARIA -> DIARIA
    # =========================================================================

    print("Agregando clima horario a diario...")

    agg_dict = {
        "temperatura": ["mean", "max", "min", "std"],
        "humedad": ["mean", "max", "min", "std"],
        "precipitacion": ["sum", "max"],
        "pm2_5": ["mean", "max", "min", "std"],
        "pm10": ["mean", "max", "min", "std"],
        "no2": ["mean", "max", "min", "std"],
        "ozono": ["mean", "max", "min", "std"],
    }

    df_daily = (
        df_clima
        .groupby("fecha")
        .agg(agg_dict)
    )

    df_daily.columns = [
        f"{c[0]}_{c[1]}"
        for c in df_daily.columns
    ]

    df_daily.reset_index(inplace=True)

    # =========================================================================
    # HORAS SOBRE UMBRAL
    # =========================================================================

    print("Creando indicadores contaminación...")

    thresholds = {
        "pm2_5": [25, 50],
        "pm10": [50, 100],
        "no2": [50],
        "ozono": [100]
    }

    for pollutant, values in thresholds.items():

        for value in values:

            col_name = f"{pollutant}_h_gt_{value}"

            tmp = (
                df_clima
                .assign(flag=(df_clima[pollutant] > value).astype(int))
                .groupby("fecha")["flag"]
                .sum()
                .rename(col_name)
                .reset_index()
            )

            df_daily = df_daily.merge(
                tmp,
                on="fecha",
                how="left"
            )

    # =========================================================================
    # VARIABLES DERIVADAS CLIMA
    # =========================================================================

    df_daily["temp_range"] = (
        df_daily["temperatura_max"] -
        df_daily["temperatura_min"]
    )

    df_daily["pm25_pm10_ratio"] = (
        df_daily["pm2_5_mean"] /
        (df_daily["pm10_mean"] + 1e-6)
    )

    df_daily["contaminacion_total"] = (
        df_daily["pm2_5_mean"] +
        df_daily["pm10_mean"] +
        df_daily["no2_mean"] +
        df_daily["ozono_mean"]
    )

    df_daily = df_daily.sort_values("fecha")

    # =========================================================================
    # CARGA CONSULTAS
    # =========================================================================

    print("Cargando consultas...")

    df_cons = pd.read_csv(INPUT_CONSULTAS)

    df_cons.rename(
        columns={"Unnamed: 0": "fecha"},
        inplace=True
    )

    df_cons["fecha"] = pd.to_datetime(df_cons["fecha"])

    df_cons = df_cons.sort_values("fecha")

    # =========================================================================
    # INNER JOIN
    # =========================================================================

    print("Realizando inner join...")

    df = pd.merge(
        df_cons,
        df_daily,
        on="fecha",
        how="inner"
    )

    df = df.sort_values("fecha").reset_index(drop=True)

    # =========================================================================
    # CALENDARIO
    # =========================================================================

    print("Creando variables calendario...")

    df["dow"] = df["fecha"].dt.dayofweek

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dow"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dow"] / 7
    )

    df["month"] = df["fecha"].dt.month

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # =========================================================================
    # FERIADOS CHILE
    # =========================================================================

    print("Creando variables feriados...")

    cl_holidays = holidays.Chile()

    df["is_holiday"] = (
        df["fecha"]
        .dt.date
        .apply(lambda x: int(x in cl_holidays))
    )

    df["before_holiday"] = (
        df["is_holiday"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    df["after_holiday"] = (
        df["is_holiday"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    # =========================================================================
    # ESTACIONES
    # =========================================================================

    df["season"] = (
        df["fecha"]
        .dt.month
        .apply(get_season)
    )

    season_dummies = pd.get_dummies(
        df["season"],
        prefix="est",
        dtype=int
    )

    df = pd.concat(
        [df, season_dummies],
        axis=1
    )

    df.drop(columns=["season"], inplace=True)

    # =========================================================================
    # FEATURES CONSULTAS
    # =========================================================================

    print("Creando features consultas...")

    consulta_lags = [1, 2, 3, 5, 7, 10, 14, 21, 28]

    for lag in consulta_lags:
        df[f"atenciones_lag{lag}"] = (
            df["total_atenciones"].shift(lag)
        )

    for w in [3, 7, 10, 14, 21, 28]:

        df[f"atenciones_ma{w}"] = (
            df["total_atenciones"]
            .shift(1)
            .rolling(w)
            .mean()
        )

        df[f"atenciones_std{w}"] = (
            df["total_atenciones"]
            .shift(1)
            .rolling(w)
            .std()
        )

        df[f"atenciones_max{w}"] = (
            df["total_atenciones"]
            .shift(1)
            .rolling(w)
            .max()
        )

        df[f"atenciones_min{w}"] = (
            df["total_atenciones"]
            .shift(1)
            .rolling(w)
            .min()
        )

    df["atenciones_trend_3_7"] = (
        df["atenciones_ma3"] -
        df["atenciones_ma7"]
    )

    df["atenciones_trend_7_14"] = (
        df["atenciones_ma7"] -
        df["atenciones_ma14"]
    )

    df["atenciones_trend_14_28"] = (
        df["atenciones_ma14"] -
        df["atenciones_ma28"]
    )

    df["atenciones_vs_week"] = (
        df["total_atenciones"]
        - df["total_atenciones"].shift(7)
    )

    df["atenciones_std7"] = (
        df["total_atenciones"]
          .rolling(7)
          .std()
    )

    df["atenciones_pct_change_7"] = (
        df["total_atenciones"]
        .shift(1)
        .pct_change(7)
    )

    df["atenciones_pct_change_14"] = (
        df["total_atenciones"]
        .shift(1)
        .pct_change(14)
    )

    print("Creando variables normalizadas de atenciones (mitigación de drift)...")

    df["atenciones_ma_baseline"] = (
        df["total_atenciones"]
        .shift(1)
        .rolling(ATENCIONES_LONG_WINDOW)
        .mean()
    )

    df["total_atenciones_norm"] = (
        df["total_atenciones"] /
        (df["atenciones_ma_baseline"] + EPS_BASELINE)
    )

    for lag in consulta_lags:
        df[f"atenciones_lag{lag}_norm"] = (
            df[f"atenciones_lag{lag}"] /
            (df["atenciones_ma_baseline"] + EPS_BASELINE)
        )

    for w in [3, 7, 10, 14, 21, 28]:

        df[f"atenciones_ma{w}_norm"] = (
            df[f"atenciones_ma{w}"] /
            (df["atenciones_ma_baseline"] + EPS_BASELINE)
        )

        df[f"atenciones_std{w}_norm"] = (
            df[f"atenciones_std{w}"] /
            (df["atenciones_ma_baseline"] + EPS_BASELINE)
        )

        df[f"atenciones_max{w}_norm"] = (
            df[f"atenciones_max{w}"] /
            (df["atenciones_ma_baseline"] + EPS_BASELINE)
        )

        df[f"atenciones_min{w}_norm"] = (
            df[f"atenciones_min{w}"] /
            (df["atenciones_ma_baseline"] + EPS_BASELINE)
        )

    # =========================================================================
    # FEATURES CLIMA
    # =========================================================================

    print("Creando lags y medias móviles clima...")

    climate_features = [
        "temperatura_mean",
        "humedad_mean",
        "precipitacion_sum",
        "pm2_5_mean",
        "pm10_mean",
        "no2_mean",
        "ozono_mean",
        "contaminacion_total"
    ]

    df = add_lags(
        df,
        climate_features,
        lags=[1, 2, 3, 5, 7]
    )

    df = add_moving_averages(
        df,
        climate_features,
        windows=[3, 7, 14]
    )

    df = add_moving_std(
        df,
        climate_features,
        windows=[7, 14]
    )

    # =========================================================================
    # CAMBIOS DIARIOS
    # =========================================================================

    for col in [
        "temperatura_mean",
        "pm2_5_mean",
        "pm10_mean",
        "no2_mean",
        "ozono_mean"
    ]:
        df[f"{col}_delta1"] = df[col].diff(1)

    df = df.copy()

    # =========================================================================
    # TARGET — una columna por cada horizonte  [MULTI-HORIZONTE]
    # =========================================================================

    print(f"Creando targets para horizontes {horizons}...")

    target_cols = []
    for h in horizons:
        # Convención heredada: horizonte operacional h -> shift(-(h+1)),
        # porque al correr el pipeline en la mañana de t+1 solo se conoce
        # t completo (ver nota en CONFIG).
        col_name = f"target_h{h}"
        df[col_name] = df["total_atenciones"].shift(-(h + 1))
        target_cols.append(col_name)

    # =========================================================================
    # LIMPIEZA FINAL
    # =========================================================================
    #
    # [MULTI-HORIZONTE] El dropna() NO se aplica sobre las columnas target
    # (cada una tiene un número distinto de NaN al final de la serie, según
    # su horizonte). Se aplica solo sobre las FEATURES: así ninguna fila
    # válida para un horizonte corto (ej. h=1) se descarta solo porque un
    # horizonte largo (ej. h=7) todavía no tiene target definido ahí.
    # Cada script downstream (feature_selection.py, model_xgboost.py) debe
    # descartar por su cuenta las filas donde su target_h{horizon}
    # específico sea NaN, antes de usar el dataset.

    print("Eliminando NaNs (solo en features, targets conservan su cola natural)...")

    rows_before = len(df)

    feature_cols_for_dropna = [c for c in df.columns if c not in target_cols]
    df = df.dropna(subset=feature_cols_for_dropna)

    rows_after = len(df)

    # ordenar columnas

    cols = ["fecha", "total_atenciones"] + target_cols

    remaining = [
        c for c in df.columns
        if c not in cols
    ]

    df = df[
        cols + sorted(remaining)
    ]

    # =========================================================================
    # EXPORTAR
    # =========================================================================

    df.to_csv(
        OUTPUT_DATASET,
        index=False
    )

    # =========================================================================
    # RESUMEN
    # =========================================================================

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETADO")
    print("=" * 80)

    print(f"Filas iniciales: {rows_before:,}")
    print(f"Filas finales (features sin NaN): {rows_after:,}")
    print(f"Variables:       {df.shape[1]:,}")
    print(f"Horizontes generados: {horizons}  ({len(target_cols)} columnas target)")

    for h, col in zip(horizons, target_cols):
        n_validos = df[col].notna().sum()
        print(f"  {col}: {n_validos:,} filas con target válido "
              f"({len(df) - n_validos} NaN de cola por horizonte)")

    print("\nPrimeras columnas:")
    print(df.columns[:20].tolist())

    print(f"\nDataset guardado en:")
    print(OUTPUT_DATASET)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint (ejecución standalone)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()