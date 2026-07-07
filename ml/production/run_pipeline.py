"""
run_pipeline.py

Pipeline principal de inferencia, evaluación y reporte — se ejecuta
DESPUÉS del entrenamiento. Reutiliza (sin modificar) los módulos ya
existentes de feature engineering, selección de variables y entrenamiento.

Uso:
    python run_pipeline.py                 # entrena + genera todo el reporte
    python run_pipeline.py --skip-training # usa modelos ya guardados en disco

Supuestos de integración (ver pipeline/config.py para ajustar rutas):
    - feature_engineering.py expone run_feature_engineering() -> pd.DataFrame
    - feature_selection.py   expone run_feature_selection(df, horizon)
    - model_xgboost.py       expone run_all(df, horizon), split_data(df),
                              predict_ensemble(models, X), smape(...)
    - Modelos guardados en save/model/xgboost_regression_h{h}.pkl
      (lista de Booster — ensamble de bagging)
    - Features seleccionadas en ml/features/selected_features_h{h}.csv
    - Dataset con columna 'fecha' (índice), 'total_atenciones' (valor
      real) y target_h1..target_h7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pipeline import config
from pipeline.logging_utils import setup_logger, get_logger
from pipeline.io_utils import (
    load_engineered_dataset, load_all_models, load_all_features, normalize_date_index,
)
from pipeline.forecasting import generate_forecast
from pipeline.evaluation import run_evaluation
from pipeline.backtesting import run_backtesting, compute_backtesting_metrics
from pipeline.feature_importance import compute_feature_importance_all, top_feature_per_horizon
from pipeline import plotting
from pipeline.reporting import save_dataframe, build_summary, save_summary


def _run_training_stage(logger) -> pd.DataFrame | None:
    """
    Etapa 1: feature engineering + entrenamiento de los 7 modelos.

    Importa los módulos de entrenamiento existentes dinámicamente (no se
    modifican). Si no se pueden importar (ej. nombres de archivo distintos
    a los asumidos en config.py), se informa y se retorna None — el
    llamador decide si continuar solo con modelos ya guardados en disco.
    """
    try:
        import feature_engineering as fe_module
        import feature_selection as fs_module
        import model_xgboost as mx_module
    except ImportError as e:
        logger.error(
            f"No se pudieron importar los módulos de entrenamiento ({e}). "
            f"Verifica los nombres de archivo en pipeline/config.py / este "
            f"script, o usa --skip-training para omitir esta etapa."
        )
        return None

    logger.info("Ejecutando feature engineering...")
    df = fe_module.run_feature_engineering()
    df = normalize_date_index(df)

    for h in config.HORIZONS:
        logger.info(f"[h={h}] Selección de variables...")
        fs_module.run_feature_selection(df, horizon=h)
        logger.info(f"[h={h}] Entrenamiento del modelo...")
        mx_module.run_all(df, horizon=h)

    logger.info("Entrenamiento completado: los 7 modelos quedaron guardados en disco.")
    return df


def _generate_all_plots(
    df: pd.DataFrame,
    raw_results: dict,
    backtest_df: pd.DataFrame,
    backtesting_metrics_df: pd.DataFrame,
) -> None:
    """Genera todos los gráficos de la sección 9 del pipeline."""
    all_errors_by_horizon: dict[int, pd.Series] = {}

    for h, res in raw_results.items():
        plot_dir = config.horizon_plot_dir(h)

        plotting.plot_horizon_series(
            res["dates"], res["y_true"], res["y_pred_ml"], h,
            plot_dir / "serie_real_vs_prediccion.png",
        )
        plotting.plot_scatter(
            res["y_true"], res["y_pred_ml"], f"t+{h}", plot_dir / "scatter.png"
        )
        plotting.plot_error_histogram(
            res["y_true"] - res["y_pred_ml"], f"t+{h}", plot_dir / "error_histogram.png"
        )

        all_errors_by_horizon[h] = res["y_true"] - res["y_pred_ml"]

    if all_errors_by_horizon:
        plotting.plot_error_boxplot_by_horizon(
            all_errors_by_horizon, config.PLOTS_DIR / "boxplot_error_por_horizonte.png"
        )

    if not backtest_df.empty:
        plotting.plot_error_heatmap(backtest_df, config.PLOTS_DIR / "heatmap.png")
        plotting.plot_operational_forecast(
            df[config.RAW_TARGET_COL], backtest_df,
            config.PLOTS_DIR / "operational_forecast.png",
        )

    if not backtesting_metrics_df.empty:
        plotting.plot_naive_comparison(
            backtesting_metrics_df, config.PLOTS_DIR / "naive_comparison_mae.png", metric="mae"
        )


def _check_working_directory(logger) -> None:
    """
    feature_engineering.py, feature_selection.py y model_xgboost.py usan
    rutas relativas (ej. Path("save/model")) que se resuelven contra el
    directorio de trabajo actual (cwd) — NO contra la ubicación de esos
    archivos. Este pipeline, en cambio, calcula todo relativo a
    config.PROJECT_ROOT, así que si el cwd no coincide con la raíz del
    proyecto, los modelos/features que guardan esos módulos terminan en
    una ubicación distinta a la que este pipeline busca — fallando de
    forma silenciosa y confusa (aparenta que el entrenamiento funcionó,
    pero los .pkl no aparecen donde se esperan).

    Este chequeo solo avisa; no aborta la ejecución, porque en algunos
    setups (ej. symlinks) la comparación puede dar falso positivo.
    """
    cwd = Path.cwd().resolve()
    if cwd != config.PROJECT_ROOT:
        logger.warning(
            f"El directorio de trabajo actual ({cwd}) no coincide con la "
            f"raíz del proyecto detectada ({config.PROJECT_ROOT}). "
            f"feature_engineering.py / feature_selection.py / model_xgboost.py "
            f"usan rutas RELATIVAS AL DIRECTORIO DE TRABAJO (no a este "
            f"pipeline), así que si vas a re-entrenar, ejecuta este script "
            f"con el directorio de trabajo en la raíz del proyecto "
            f"(ej. `cd {config.PROJECT_ROOT}` antes de correrlo), o los "
            f"modelos/features nuevos podrían guardarse en un lugar "
            f"distinto al que este pipeline espera."
        )


def main(retrain: bool = True) -> None:
    config.ensure_output_dirs()
    logger = setup_logger(config.LOG_PATH)

    logger.info("=" * 70)
    logger.info("INICIO DEL PIPELINE DE INFERENCIA, EVALUACIÓN Y REPORTE")
    logger.info("=" * 70)

    _check_working_directory(logger)

    timings: dict[str, float] = {}
    t_start_total = time.time()

    # ── 1. Entrenamiento ──────────────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 1/8 — Entrenamiento")
    df = None
    if retrain:
        df = _run_training_stage(logger)
    if df is None:
        logger.info("Usando dataset engineered ya existente / recién recalculado "
                    "(sin re-entrenar modelos).")
        df = load_engineered_dataset(rerun=True)
    timings["1_entrenamiento"] = time.time() - t0

    # ── 2. Carga de modelos ───────────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 2/8 — Carga de modelos")
    models_by_horizon = load_all_models()
    features_by_horizon = load_all_features()
    if not models_by_horizon:
        logger.error("No se pudo cargar ningún modelo — abortando el pipeline.")
        return
    timings["2_carga_modelos"] = time.time() - t0

    # ── 3. Predicción futura ──────────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 3/8 — Predicción futura (forecast T+1..T+7)")
    forecast_df = generate_forecast(df, models_by_horizon, features_by_horizon)
    save_dataframe(forecast_df, config.FORECAST_CSV, "Forecast")
    t_forecast = time.time() - t0
    timings["3_forecast"] = t_forecast
    timings["tiempo_promedio_por_prediccion"] = (
        t_forecast / len(forecast_df) if len(forecast_df) else float("nan")
    )

    # ── 4. Evaluación + comparación con Naive ─────────────────────────
    t0 = time.time()
    logger.info("Etapa 4/8 — Evaluación del modelo + comparación con Naive")
    eval_results = run_evaluation(df, models_by_horizon, features_by_horizon)
    metrics_df = eval_results["metrics_df"]
    comparison_df = eval_results["comparison_df"]
    raw_results = eval_results["raw_results"]
    save_dataframe(metrics_df, config.METRICS_CSV, "Métricas de evaluación")
    save_dataframe(comparison_df, config.COMPARISON_CSV, "Tabla comparativa vs Naive")
    timings["4_evaluacion"] = time.time() - t0

    # ── 5. Backtesting operacional ────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 5/8 — Backtesting operacional (simulación semanal)")
    backtest_df = run_backtesting(df, models_by_horizon, features_by_horizon)
    save_dataframe(backtest_df, config.BACKTEST_PRED_CSV, "Predicciones de backtesting")
    backtesting_metrics_df = compute_backtesting_metrics(backtest_df, df)
    save_dataframe(backtesting_metrics_df, config.BACKTEST_METRICS_CSV, "Métricas de backtesting")
    timings["5_backtesting"] = time.time() - t0

    # ── 6. Gráficos ────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 6/8 — Generación de gráficos")
    _generate_all_plots(df, raw_results, backtest_df, backtesting_metrics_df)
    timings["6_graficos"] = time.time() - t0

    # ── 7. Importancia de variables ───────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 7/8 — Importancia de variables")
    fi_by_horizon = compute_feature_importance_all(models_by_horizon)
    for h, fi_df in fi_by_horizon.items():
        save_dataframe(
            fi_df, config.FEATURE_IMPORTANCE_DIR / f"feature_importance_h{h}.csv",
            f"Importancia de variables h={h}",
        )
        plotting.plot_feature_importance(
            fi_df, h, config.FEATURE_IMPORTANCE_DIR / f"feature_importance_h{h}.png"
        )
    top_feature_by_horizon = top_feature_per_horizon(fi_by_horizon)
    timings["7_importancia_variables"] = time.time() - t0

    # ── 8. Resumen automático ─────────────────────────────────────────
    t0 = time.time()
    logger.info("Etapa 8/8 — Resumen automático")
    timings["total"] = time.time() - t_start_total
    summary = build_summary(
        metrics_df, comparison_df, backtesting_metrics_df,
        top_feature_by_horizon, timings,
    )
    save_summary(summary, config.SUMMARY_CSV, config.SUMMARY_JSON)
    timings["8_resumen"] = time.time() - t0

    logger.info("=" * 70)
    logger.info(f"PIPELINE FINALIZADO — tiempo total: {timings['total']:.1f}s")
    logger.info(f"Resultados en: {config.RESULTS_DIR.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de inferencia, evaluación y reporte (post-entrenamiento)."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Omite feature engineering + entrenamiento; usa los modelos y "
             "features ya existentes en disco (más rápido para iterar sobre "
             "el reporte sin reentrenar).",
    )
    args = parser.parse_args()
    main(retrain=not args.skip_training)
