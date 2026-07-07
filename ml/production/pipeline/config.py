"""
pipeline/config.py

Configuración central del pipeline de inferencia, evaluación y reporte.

Todas las rutas se calculan con pathlib de forma RELATIVA a la ubicación
de este archivo (no al directorio desde donde se ejecute python), para
que run_pipeline.py funcione sin importar desde dónde se lo invoque.

Estructura de proyecto asumida (ver PROJECT_ROOT más abajo):

    consultas-predict-cabrero/          <- PROJECT_ROOT
        data/
            processed/
            raw/
        ml/
            features/
                feature_engineering.py
                feature_selection.py
                feature_selection_outputs/
                    selected_features_h{h}.csv
                    feature_selection_report_h{h}.csv
                    metric_curve_k_features_h{h}.csv
                    family_redundancy_dropped_h{h}.csv
            models/
                model_xgboost.py
            production/
                run_pipeline.py          <- orquestador (único archivo suelto)
                pipeline/                <- este paquete
        save/
            model/
                xgboost_regression_h{h}.pkl
                xgboost_regression_h{h}_metrics.json

[IMPORTS] feature_engineering.py, feature_selection.py y model_xgboost.py
se importan por nombre simple (ej. `import model_xgboost`), no como
paquetes con __init__.py. Para que eso funcione, este módulo agrega
ml/features/ y ml/models/ a sys.path automáticamente al ser importado
(ver el bloque al final del archivo) — así cualquier módulo del pipeline
que haga `from . import config` antes de importar model_xgboost /
feature_engineering / feature_selection queda con el path ya resuelto.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Raíz del proyecto (calculada de forma relativa a este archivo)
# ──────────────────────────────────────────────────────────────────────────
#
# Este archivo vive en: <root>/ml/production/pipeline/config.py
#   parents[0] = ml/production/pipeline   (carpeta de este archivo)
#   parents[1] = ml/production
#   parents[2] = ml
#   parents[3] = <root>

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# ──────────────────────────────────────────────────────────────────────────
# Horizontes de predicción
# ──────────────────────────────────────────────────────────────────────────

HORIZONS: list[int] = list(range(1, 8))  # 1..7 días

# ──────────────────────────────────────────────────────────────────────────
# Rutas de entrada (ya generadas por los módulos existentes — NO se tocan)
# ──────────────────────────────────────────────────────────────────────────

MODEL_DIR: Path = PROJECT_ROOT / "save" / "model"
FEATURES_DIR: Path = PROJECT_ROOT / "ml" / "features" / "feature_selection_outputs"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

# Carpetas donde viven los módulos .py a importar por nombre simple
# (feature_engineering.py, feature_selection.py, model_xgboost.py).
FEATURES_MODULE_DIR: Path = PROJECT_ROOT / "ml" / "features"
MODELS_MODULE_DIR: Path = PROJECT_ROOT / "ml" / "models"

MODEL_NAME: str = "xgboost_regression"

RAW_CONSULTAS_PATH: Path = PROCESSED_DIR / "datos_consultas_corregidos.csv"

# ──────────────────────────────────────────────────────────────────────────
# Columnas del dataset engineered
# ──────────────────────────────────────────────────────────────────────────

DATE_COL: str = "fecha"
RAW_TARGET_COL: str = "total_atenciones"  # valor real observado


def target_col_for(horizon: int) -> str:
    """Nombre de la columna target para un horizonte dado."""
    return f"target_h{horizon}"


def model_path_for(horizon: int) -> Path:
    """Ruta al .pkl del ensamble de un horizonte dado."""
    return MODEL_DIR / f"{MODEL_NAME}_h{horizon}.pkl"


def features_path_for(horizon: int) -> Path:
    """Ruta al CSV de features seleccionadas de un horizonte dado."""
    return FEATURES_DIR / f"selected_features_h{horizon}.csv"


def forecast_date_for(last_available_date, horizon: int):
    """
    Fecha calendario que corresponde al horizonte `horizon`, dado que la
    última fecha con datos completos es `last_available_date`.

    Convención heredada del feature engineering: target_h{h} en la fila de
    fecha D = total_atenciones en D + (h + 1). Operacionalmente, el
    pipeline se corre la mañana siguiente a que D quede completo (D+1),
    así que el horizonte h corresponde a la fecha (D+1) + h.
    """
    import pandas as pd

    return pd.Timestamp(last_available_date) + pd.Timedelta(days=horizon + 1)


# ──────────────────────────────────────────────────────────────────────────
# Rutas de salida
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR: Path = PROJECT_ROOT / "ml" / "production" / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
FEATURE_IMPORTANCE_DIR: Path = PLOTS_DIR / "feature_importance"

FORECAST_CSV: Path = RESULTS_DIR / "forecast.csv"
METRICS_CSV: Path = RESULTS_DIR / "metrics.csv"
COMPARISON_CSV: Path = RESULTS_DIR / "comparison.csv"
BACKTEST_PRED_CSV: Path = RESULTS_DIR / "backtesting_predictions.csv"
BACKTEST_METRICS_CSV: Path = RESULTS_DIR / "backtesting_metrics.csv"
SUMMARY_CSV: Path = RESULTS_DIR / "summary.csv"
SUMMARY_JSON: Path = RESULTS_DIR / "summary.json"
LOG_PATH: Path = RESULTS_DIR / "pipeline.log"


def horizon_plot_dir(horizon: int) -> Path:
    return PLOTS_DIR / f"horizon_{horizon}"


def ensure_output_dirs() -> None:
    """Crea toda la estructura de carpetas de salida si no existe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        horizon_plot_dir(h).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Backtesting operacional
# ──────────────────────────────────────────────────────────────────────────

# Si True, en vez de truncar el dataset ya engineered en cada fecha de
# corte (rápido, matemáticamente equivalente porque todas las features son
# causales/backward-looking), se re-ejecuta feature_engineering.py desde
# cero sobre los datos crudos truncados en cada domingo (mucho más lento,
# útil solo como verificación de que ambos métodos coinciden).
STRICT_RERUN_FEATURE_ENGINEERING: bool = False


# ──────────────────────────────────────────────────────────────────────────
# [IMPORTS] Inyección de sys.path — se ejecuta automáticamente al importar
# este módulo, para que `import model_xgboost` / `import feature_engineering`
# / `import feature_selection` funcionen desde cualquier submódulo del
# pipeline sin necesitar __init__.py en ml/features/ ni ml/models/.
# ──────────────────────────────────────────────────────────────────────────

def _add_to_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_add_to_sys_path(FEATURES_MODULE_DIR)
_add_to_sys_path(MODELS_MODULE_DIR)
