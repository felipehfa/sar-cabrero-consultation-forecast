import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class NumpyEncoder(json.JSONEncoder):
    """Permite serializar tipos numpy (int64, float64, ndarray) en JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
#
# [HORIZON] Ya NO es una constante global — es un parámetro explícito de
# run_all() (y de las funciones internas que lo necesitan), con default=1.
# Esto permite importar este módulo desde otro script y correr los 7
# horizontes en un loop, ej.:
#
#     import model_xgboost as mx
#     for h in range(1, 8):
#         df = pd.read_csv(f"data/processed/feature_engineering_h{h}.csv", ...)
#         mx.run_all(df, horizon=h)
#
# El horizonte determina: el gap del TimeSeriesSplit (evita fuga del target
# hacia adelante), qué archivo de features se carga
# (ml/features/selected_features_h{horizon}.csv), los textos impresos, y el
# nombre del .pkl exportado (save/model/xgboost_regression_h{horizon}.pkl).

FEATURES_DIR = Path("ml/features/feature_selection_outputs")
PROCESSED_DIR  = Path("data/processed")
SAVE_MODEL_DIR = Path("save/model")

# [MULTI-HORIZONTE] Ya no hay una columna "target" fija — cada horizonte
# usa su propia columna target_h{horizon}, calculada dentro de run_all().
SEED        = 42
MODEL_NAME  = "xgboost_regression"

# Número de modelos del ensamble de bagging en el entrenamiento final.
# Cada modelo usa los mismos hiperparámetros (best_params) pero una semilla
# distinta (SEED, SEED+1, ..., SEED+N-1). Las predicciones finales son el
# promedio del ensamble. N_SEEDS_BAGGING=1 → un solo modelo.
N_SEEDS_BAGGING = 5

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def load_selected_features(horizon: int = 1) -> list:
    """
    Carga ml/features/selected_features_h{horizon}.csv — cada horizonte
    tiene su propio archivo de features seleccionadas (la selección de
    variables se corre por separado para cada horizonte).
    """
    path = FEATURES_DIR / f"selected_features_h{horizon}.csv"
    features = pd.read_csv(path)["feature"].tolist()
    print(f"\nFeatures seleccionadas para horizonte h={horizon} "
          f"({len(features)}): {features}")
    return features


def split_data(df: pd.DataFrame):
    """
    Split temporal:

        Train : resto del historial
        Val   : 365 días
        Test  : últimos 365 días

    """

    df = df.sort_index()

    last_date = df.index.max()

    # Último año = Test
    test_start = last_date - pd.Timedelta(days=364)

    # Año anterior = Validación
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=364)

    train = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index <= val_end)]
    test = df[df.index >= test_start]

    print("\nSplit temporal:")
    print(f"  Train: {len(train)} | {train.index.min().date()} → {train.index.max().date()}")
    print(f"  Val:   {len(val)} | {val.index.min().date()} → {val.index.max().date()}")
    print(f"  Test:  {len(test)} | {test.index.min().date()} → {test.index.max().date()}")

    return train, val, test

# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    ))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    mae_v   = mean_absolute_error(y_true, y_pred)
    smape_v = smape(y_true, y_pred)
    rmse_v  = rmse(y_true, y_pred)
    prefix  = f"[{label}] " if label else ""
    print(f"  {prefix}MAE={mae_v:.3f}  sMAPE={smape_v:.2f}%  RMSE={rmse_v:.3f}")
    return {"mae": round(mae_v,4), "smape": round(smape_v,4), "rmse": round(rmse_v,4)}

# ──────────────────────────────────────────────────────────────────────────────
# PSI
# ──────────────────────────────────────────────────────────────────────────────

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos muestras de una
    misma variable continua, típicamente "expected" = distribución de
    referencia (ej. train) y "actual" = distribución a comparar (ej. val
    o test).

    El PSI mide cuánto cambió la DISTRIBUCIÓN de una variable entre dos
    periodos — es decir, drift de los datos de entrada, no del error del
    modelo. Un PSI alto en una feature no implica que el modelo esté fallando,
    pero sí que esa variable se está comportando de forma distinta a como
    lo hacía cuando el modelo aprendió de ella, lo cual es una señal de
    riesgo de generalización.

    Algoritmo:
        1. Se divide el rango de `expected` en `buckets` intervalos usando
           sus propios percentiles (cuantiles), no anchos fijos. Esto
           asegura que cada bucket tenga aproximadamente la misma cantidad
           de observaciones de referencia, dando más resolución donde la
           densidad de `expected` es mayor.
        2. Los bordes externos se reemplazan por -inf/+inf para que ningún
           valor de `actual` quede fuera de rango y sea ignorado, aunque
           `actual` tenga mínimos/máximos distintos a `expected`.
        3. Se calcula la proporción de observaciones de cada muestra que
           cae en cada bucket.
        4. Proporciones en 0 se reemplazan por un valor pequeño (1e-4) para
           evitar división por cero o log(0) en el paso siguiente — sin
           esto, un bucket vacío en cualquiera de las dos muestras
           produciría PSI = inf o NaN.
        5. PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)
           sobre todos los buckets i. Es una divergencia tipo
           Kullback-Leibler simetrizada: penaliza tanto que un bucket haya
           ganado peso en `actual` respecto a `expected` como lo contrario.

    Args:
        expected: array de la variable en la muestra de referencia
                   (ej. valores de esa columna en train).
        actual:   array de la misma variable en la muestra a comparar
                   (ej. valores de esa columna en val o test).
        buckets:  número de intervalos (cuantiles) de `expected` a usar.
                   10 es el valor convencional en la literatura de riesgo
                   crediticio de donde viene esta métrica.

    Returns:
        PSI (float, >= 0). Convención de interpretación:
            PSI < 0.10  → sin cambio relevante
            0.10 - 0.25 → cambio moderado (revisar / "watch")
            PSI > 0.25  → cambio significativo (drift, "alerta")
        Retorna 0.0 si `expected` no tiene suficiente variabilidad para
        definir al menos 2 bordes de bucket distintos (ej. columna casi
        constante), caso en que el PSI no está bien definido.

    Nota: los buckets se definen SIEMPRE a partir de `expected`, nunca de
    `actual` — el PSI es direccional (compara actual contra una referencia
    fija), no una medida simétrica de distancia entre dos distribuciones
    arbitrarias.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    # Bordes de bucket = percentiles de `expected` (cuantiles, no ancho fijo).
    # np.unique() colapsa percentiles duplicados (ej. si expected tiene
    # muchos valores repetidos), lo que puede resultar en menos de
    # `buckets` intervalos reales.
    breakpoints = np.unique(np.nanpercentile(expected, np.linspace(0, 100, buckets + 1)))

    # Con menos de 2 bordes no se puede formar ni un solo bucket válido
    # (ej. expected es constante o casi constante) → PSI no definido, 0.0
    # por convención (sin evidencia de drift, no "drift infinito").
    if len(breakpoints) < 2:
        return 0.0

    # Extender los bordes externos a infinito para que ningún valor de
    # `actual` quede fuera de rango (ej. actual tiene un máximo mayor al de
    # expected) y por tanto excluido silenciosamente del cálculo.
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def proportions(data, breaks):
        counts = np.histogram(data, bins=breaks)[0]
        props  = counts / len(data)
        # Evita log(0) / división por 0 en el cálculo de PSI cuando un
        # bucket queda vacío en alguna de las dos muestras.
        return np.where(props == 0, 1e-4, props)

    exp_p = proportions(expected, breakpoints)
    act_p = proportions(actual,   breakpoints)

    # Divergencia tipo KL simetrizada, sumada sobre todos los buckets.
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

# ──────────────────────────────────────────────────────────────────────────────
# [IC] Conformal Prediction — Intervalos de predicción con garantía de cobertura
# ──────────────────────────────────────────────────────────────────────────────

def conformal_quantile(val_residuals: np.ndarray, alpha: float) -> float:
    """
    [IC] Calcula el cuantil conformal para nivel de cobertura 1-alpha.

    Fórmula: q = quantil(|residuos_val|, ceil((1-α)(n+1))/n)
    El factor (n+1)/n es la corrección finita que garantiza cobertura
    marginal ≥ 1-α en expectativa sobre la aleatoriedad de val.

    Args:
        val_residuals: residuos del modelo en val (y_real - y_pred)
        alpha: nivel de error (0.10 → IC90%, 0.20 → IC80%)

    Returns:
        q: radio del intervalo de predicción (simétrico)
    """
    n       = len(val_residuals)
    scores  = np.abs(val_residuals)
    level   = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, level))


def build_conformal_intervals(
    y_pred:        np.ndarray,
    val_residuals: np.ndarray,
) -> dict:
    """
    [IC] Construye ICs conformales para IC80% y IC90% sobre y_pred.

    Retorna dict con arrays lower/upper para cada nivel.
    La cobertura esperada es ≥ al nivel nominal sin supuestos distribucionales.
    """
    y_pred = np.asarray(y_pred, dtype=float)

    q80 = conformal_quantile(val_residuals, alpha=0.20)
    q90 = conformal_quantile(val_residuals, alpha=0.10)

    print(f"\n── Conformal Prediction  [IC] ───────────────────────────")
    print(f"  n_val (scores): {len(val_residuals)}")
    print(f"  Cuantil IC80%:  ±{q80:.2f} consultas")
    print(f"  Cuantil IC90%:  ±{q90:.2f} consultas")
    print(f"  (IC simétrico: ŷ ± q)")

    return {
        "ic80": {
            "lower": np.clip(y_pred - q80, 0, None),
            "upper": y_pred + q80,
            "q":     q80,
            "alpha": 0.20,
        },
        "ic90": {
            "lower": np.clip(y_pred - q90, 0, None),
            "upper": y_pred + q90,
            "q":     q90,
            "alpha": 0.10,
        },
    }


def evaluate_conformal_coverage(
    y_true: np.ndarray,
    intervals: dict,
) -> dict:
    """
    [IC] Evalúa cobertura real vs nominal de los ICs conformales.
    También calcula el ancho medio del intervalo (eficiencia).
    """
    y_true  = np.asarray(y_true, dtype=float)
    results = {}

    print(f"\n  Cobertura conformal (val):")
    for nivel, data in intervals.items():
        lower    = data["lower"]
        upper    = data["upper"]
        nominal  = (1 - data["alpha"]) * 100
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper))) * 100
        width    = float(np.mean(upper - lower))
        status   = "✓" if coverage >= nominal - 2 else "⚠"
        print(f"    {nivel}: nominal={nominal:.0f}%  real={coverage:.1f}%  "
              f"ancho_medio={width:.1f}  {status}")
        results[nivel] = {
            "cobertura_nominal": nominal,
            "cobertura_real":    round(coverage, 2),
            "ancho_medio":       round(width, 2),
            "q":                 round(data["q"], 4),
        }
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Optimización de hiperparámetros
# ──────────────────────────────────────────────────────────────────────────────

def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon:   int = 1,
    n_trials:  int = 50,
    cv_splits: int = 5,
) -> tuple:
    """
    Optuna minimiza MAE promedio de un TimeSeriesSplit con gap=horizon,
    calculado EXCLUSIVAMENTE sobre X_train/y_train.

    val queda reservado para early stopping del modelo final y para la
    evaluación reportada — nunca influye en qué hiperparámetros se eligen.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=horizon)

    def objective_fn(trial):
        objective = trial.suggest_categorical("objective", ["count:poisson", "reg:tweedie"])
        params = {
            "objective":        objective,
            "verbosity":        0,
            "seed":             SEED,
            "n_jobs":           -1,
            "learning_rate":    trial.suggest_float("learning_rate",    0.01, 0.10, log=True),
            "max_depth":        trial.suggest_int(  "max_depth",        2, 6),
            "max_leaves":       trial.suggest_int(  "max_leaves",       8, 64),
            "min_child_weight": trial.suggest_int(  "min_child_weight", 1, 20),
            "subsample":        trial.suggest_float("subsample",        0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "alpha":            trial.suggest_float("alpha",            1e-2, 20.0, log=True),
            "lambda":           trial.suggest_float("lambda",           1e-2, 20.0, log=True),
            "gamma":            trial.suggest_float("gamma",            0.0,  5.0),
        }
        if objective == "reg:tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.0, 1.9
            )
        fold_maes = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr = X_train.iloc[tr_idx].fillna(0)
            X_va = X_train.iloc[va_idx].fillna(0)
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval   = xgb.DMatrix(X_va, label=y_va)
            model  = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
            )
            y_pred = np.clip(model.predict(dval), 0, None)
            fold_maes.append(mean_absolute_error(y_va, y_pred))
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective_fn, n_trials=n_trials)
    best_trial = study.best_trial
    print(f"\n  Mejor MAE CV (solo train, sin val): {best_trial.value:.4f}  (trial {best_trial.number})")
    print(f"  Objective:    {best_trial.params.get('objective','?')}")
    print(f"  Params:       {best_trial.params}")
    optim_info = {
        "best_mae_cv":        round(best_trial.value, 6),
        "best_trial":         best_trial.number,
        "best_params":        best_trial.params,
        "n_trials":           n_trials,
        "cv_splits":          cv_splits,
        "horizon_gap":        horizon,
        "cv_solo_train":      True,
    }
    return best_trial.params, optim_info

# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento — Bagging de semillas
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    best_params: dict,
    horizon: int = 1,
    n_seeds_bagging: int = N_SEEDS_BAGGING,
) -> list:
    """
    Entrena un ensamble de n_seeds_bagging modelos con los mismos
    best_params pero semillas distintas (SEED, SEED+1, ..., SEED+n-1).
    Cada modelo usa X_val/y_val para su propio early stopping.

    El .pkl exportado incluye el horizonte en el nombre del archivo
    (ej. xgboost_regression_h3.pkl) para que corridas con distinto
    horizonte no se sobrescriban entre sí.

    Returns: lista de xgb.Booster entrenados (el ensamble completo).
    """
    dval = xgb.DMatrix(X_val.fillna(0), label=y_val)

    models = []
    print(f"\nEntrenando ensamble de {n_seeds_bagging} modelo(s) "
          f"({best_params.get('objective','?')})  |  horizonte t+{horizon}...")

    for i in range(n_seeds_bagging):
        seed_i = SEED + i
        params = {**best_params, "verbosity": 0, "seed": seed_i, "n_jobs": -1}
        dtrain = xgb.DMatrix(X_train.fillna(0), label=y_train)

        print(f"\n  ── Modelo {i+1}/{n_seeds_bagging} (seed={seed_i}) ──")
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100 if i == 0 else False,   # log detallado solo del primero
        )
        print(f"    Mejor iteración: {model.best_iteration}  |  "
              f"Mejor val score: {model.best_score:.4f}")
        models.append(model)

    SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_h{horizon}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)   # se guarda la LISTA del ensamble
    print(f"\n  Ensamble guardado ({len(models)} modelos): {model_path}")
    return models


def predict_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    """
    Predicción del ensamble: promedio de las predicciones
    individuales (cada una ya clippeada a >=0 antes de promediar).
    """
    dmatrix = xgb.DMatrix(X.fillna(0))
    preds = np.stack([
        np.clip(m.predict(dmatrix), 0, None) for m in models
    ])
    return preds.mean(axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluación exhaustiva (solo impresión en consola, sin exportar a disco)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    index:     pd.DatetimeIndex,
    label:     str,
    intervals: dict = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN — {label}")
    print("="*60)

    print("\n  Métricas globales:")
    global_metrics = compute_metrics(y_true, y_pred)

    naive_pred    = np.roll(y_true, 1); naive_pred[0] = y_true[0]
    naive_mae     = mean_absolute_error(y_true[1:], naive_pred[1:])
    print(f"  Baseline naive MAE: {naive_mae:.3f}  "
          f"→ Ganancia: {naive_mae - global_metrics['mae']:+.3f}")

    print("\n  Métricas por año:")
    years          = index.year
    yearly_metrics = {}
    for yr in sorted(np.unique(years)):
        mask = years == yr
        if mask.sum() < 5: continue
        yearly_metrics[int(yr)] = compute_metrics(y_true[mask], y_pred[mask], label=str(yr))

    residuals = y_true - y_pred
    from scipy import stats as scipy_stats
    skew_val = float(scipy_stats.skew(residuals))
    kurt_val = float(scipy_stats.kurtosis(residuals))
    acf_lag1 = float(pd.Series(residuals).autocorr(lag=1))
    acf_lag7 = float(pd.Series(residuals).autocorr(lag=7))

    print(f"\n  Residuos:")
    print(f"    Media={residuals.mean():+.3f}  Std={residuals.std():.3f}  "
          f"Skew={skew_val:.3f}  Kurt={kurt_val:.3f}")
    print(f"    ACF lag-1={acf_lag1:.3f} {'⚠' if abs(acf_lag1)>0.2 else 'OK'}  "
          f"ACF lag-7={acf_lag7:.3f} {'⚠' if abs(acf_lag7)>0.2 else 'OK'}")

    months      = index.month
    month_names = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                   7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    print("\n  MAE por mes:")
    monthly_mae = {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() < 3: continue
        m_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        bar   = "█" * int(m_mae / global_metrics["mae"] * 10)
        print(f"    {month_names[m]}: {m_mae:.2f}  {bar}")
        monthly_mae[m] = round(m_mae, 4)

    # Cobertura conformal
    coverage_results = None
    if intervals is not None:
        coverage_results = evaluate_conformal_coverage(y_true, intervals)

    return {
        "metricas_globales":    global_metrics,
        "mae_baseline_naive":   round(naive_mae, 4),
        "ganancia_vs_baseline": round(naive_mae - global_metrics["mae"], 4),
        "metricas_por_año":     yearly_metrics,
        "residuos": {
            "media":    round(float(residuals.mean()), 4),
            "std":      round(float(residuals.std()),  4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "acf_lag1": round(acf_lag1, 4),
            "acf_lag7": round(acf_lag7, 4),
        },
        "mae_por_mes":          monthly_mae,
        "cobertura_conformal":  coverage_results,
    }

# ──────────────────────────────────────────────────────────────────────────────
# PSI
# ──────────────────────────────────────────────────────────────────────────────

def run_psi(features, X_train, X_val, X_test):
    print(f"\n{'='*60}")
    print("DRIFT — Population Stability Index")
    print("="*60)
    comparisons    = [
        ("Train → Val",  X_train, X_val),
        ("Val  → Test",  X_val,   X_test),
        ("Train → Test", X_train, X_test),
    ]
    psi_results    = {}
    alert_features = []
    for comp_label, ref, curr in comparisons:
        print(f"\n  [{comp_label}]")
        psi_results[comp_label] = {}
        for col in features:
            psi_val = calculate_psi(ref[col].values, curr[col].values)
            status  = ("🔴 ALERTA" if psi_val > 0.25 else
                       "🟡 WATCH"  if psi_val > 0.10 else "✅ OK")
            if psi_val > 0.25 and comp_label == "Train → Test":
                alert_features.append(col)
            print(f"    {col:40} PSI={psi_val:.4f} {status}")
            psi_results[comp_label][col] = round(psi_val, 4)
    if alert_features:
        print(f"\n  ⚠ Drift alto (Train→Test): {alert_features}")
    return psi_results, alert_features

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Corre el pipeline completo (selección de features ya hecha externamente
    -> optimización -> entrenamiento -> evaluación -> PSI -> SHAP) para un
    horizonte de predicción dado.

    horizon determina:
      - qué archivo de features se carga: selected_features_h{horizon}.csv
      - el gap del TimeSeriesSplit (Optuna)
      - los textos impresos
      - el nombre del .pkl exportado: xgboost_regression_h{horizon}.pkl
      - el nombre del JSON de métricas: xgboost_regression_h{horizon}_metrics.json
        (guardado en la misma carpeta que el modelo, save/model/, para
        análisis y presentación posterior)

    Para correr los 7 horizontes desde otro script:

        import model_xgboost as mx
        for h in range(1, 8):
            df = pd.read_csv(f"data/processed/feature_engineering_h{h}.csv",
                              index_col="fecha", parse_dates=True).sort_index()
            resultados_h = mx.run_all(df, horizon=h)
    """
    SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES = load_selected_features(horizon=horizon)
    missing  = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes:\n{missing}")

    print("\n" + "="*60)
    print(f"MODELO — XGBoost | Consultas respiratorias (t+{horizon})")
    print("="*60)

    train, val, test = split_data(df)

    # [MULTI-HORIZONTE] target_h{horizon} trae NaN de cola (magnitud según
    # el horizonte). Se descartan aquí, antes de armar X/y, para no
    # entrenar/evaluar con target faltante.
    target_col = f"target_h{horizon}"
    train = train.dropna(subset=[target_col])
    val   = val.dropna(subset=[target_col])
    test  = test.dropna(subset=[target_col])

    X_train = train[FEATURES].fillna(0)
    y_train = train[target_col].astype(float)
    X_val   = val[FEATURES].fillna(0)
    y_val   = val[target_col].astype(float)
    X_test  = test[FEATURES].fillna(0)
    y_test  = test[target_col].astype(float)

    print("\n── Estadísticas del target ──")
    for nombre, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {nombre}: media={y.mean():.1f}  std={y.std():.1f}  "
              f"min={y.min():.0f}  max={y.max():.0f}  zeros={( y==0).mean()*100:.1f}%")

    ratio_var_mean = float(y_train.var() / y_train.mean()) if y_train.mean() > 0 else 0
    print(f"  Ratio var/media (train): {ratio_var_mean:.2f}")

    # ── Optimización solo sobre X_train ──────────────
    print(f"\n{'='*60}")
    print(f"OPTIMIZACIÓN — Optuna (50 trials, gap={horizon})  [CV solo train]")
    print("="*60)
    best_params, optim_info = optimize_hyperparams(
        X_train, y_train, horizon=horizon, n_trials=50, cv_splits=5,
    )

    # ── Entrenamiento ensamble de bagging ────────────
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO FINAL  [Bagging x{N_SEEDS_BAGGING}]  [horizonte t+{horizon}]")
    print("="*60)
    models = train_model(X_train, y_train, X_val, y_val, best_params,
                          horizon=horizon, n_seeds_bagging=N_SEEDS_BAGGING)

    # ── Predicciones (promedio del ensamble) ─────────
    y_val_pred  = predict_ensemble(models, X_val)
    y_test_pred = predict_ensemble(models, X_test)

    # ── Intervalos conformales ────────────────────────
    # Los scores de calibración son los residuos absolutos de VAL.
    # VAL nunca se usó para entrenar ni para elegir hiperparámetros.
    print(f"\n{'='*60}")
    print("INTERVALOS DE PREDICCIÓN — Conformal Prediction  [IC]")
    print("="*60)

    val_residuals   = y_val.values - y_val_pred          # residuos val (con signo)
    intervals_val   = build_conformal_intervals(y_val_pred,  val_residuals)
    intervals_test  = build_conformal_intervals(y_test_pred, val_residuals)

    # Verificar cobertura en val (debe ser ≥ nominal)
    print("\n  Verificación de cobertura en val (calibración):")
    _ = evaluate_conformal_coverage(y_val.values, intervals_val)

    # ── Evaluación (solo impresión, sin exportar) ────────────
    eval_val  = evaluate_model(y_val.values,  y_val_pred,  val.index,  "Val",
                               intervals=intervals_val)
    eval_test = evaluate_model(y_test.values, y_test_pred, test.index, "Test",
                               intervals=intervals_test)

    # ── PSI (solo impresión) ──────────────────────────────────
    psi_results, alert_features = run_psi(FEATURES, X_train, X_val, X_test)

    # ── SHAP (solo impresión) ──────────────────────────────────
    # Con un ensamble, se reporta el SHAP promedio entre los modelos
    # del ensamble (no solo del primero), para que la importancia reportada
    # sea representativa del ensamble completo, no de una sola semilla.
    print(f"\n{'='*60}")
    print(f"SHAP — Feature Importance (Test, promedio de {len(models)} modelos)")
    print("="*60)
    shap_importance = {}
    try:
        import shap
        shap_abs_sum = None
        for m in models:
            explainer = shap.TreeExplainer(m)
            shap_vals = explainer(X_test).values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 0]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            shap_abs_sum = mean_abs if shap_abs_sum is None else shap_abs_sum + mean_abs
        shap_abs_mean = shap_abs_sum / len(models)
        shap_importance = dict(
            pd.Series(shap_abs_mean, index=X_test.columns)
            .sort_values(ascending=False).round(6)
        )
        print(pd.Series(shap_importance).head(15).to_string())
    except ImportError:
        print("  shap no disponible.")
    except Exception as e:
        print(f"  Error SHAP: {e}")

    # ── Exportar métricas (JSON) — junto al modelo, identificado por horizonte ─
    conformal_info = {
        "metodo":      "Split Conformal Prediction",
        "descripcion": "IC basado en residuos absolutos de val. "
                       "Garantía marginal de cobertura ≥ nominal sin supuestos distribucionales.",
        "n_calibracion": len(val_residuals),
        "ic80": {
            "q":               round(intervals_test["ic80"]["q"], 4),
            "cobertura_val":   eval_val["cobertura_conformal"]["ic80"]["cobertura_real"]
                               if eval_val.get("cobertura_conformal") else None,
            "cobertura_test":  eval_test["cobertura_conformal"]["ic80"]["cobertura_real"]
                               if eval_test.get("cobertura_conformal") else None,
            "ancho_medio_test":eval_test["cobertura_conformal"]["ic80"]["ancho_medio"]
                               if eval_test.get("cobertura_conformal") else None,
        },
        "ic90": {
            "q":               round(intervals_test["ic90"]["q"], 4),
            "cobertura_val":   eval_val["cobertura_conformal"]["ic90"]["cobertura_real"]
                               if eval_val.get("cobertura_conformal") else None,
            "cobertura_test":  eval_test["cobertura_conformal"]["ic90"]["cobertura_real"]
                               if eval_test.get("cobertura_conformal") else None,
            "ancho_medio_test":eval_test["cobertura_conformal"]["ic90"]["ancho_medio"]
                               if eval_test.get("cobertura_conformal") else None,
        },
    }

    eval_data = {
        "model":               MODEL_NAME,
        "horizon":             horizon,                 # [día de predicción, t+horizon]
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features":            FEATURES,
        "objetivo_xgb":        best_params.get("objective", "?"),
        "hiperparametros":     best_params,
        "optimizacion":        optim_info,
        "n_seeds_bagging":     N_SEEDS_BAGGING,
        "ratio_var_media":     round(ratio_var_mean, 4),
        "evaluacion_val":      eval_val,
        "evaluacion_test":     eval_test,
        "conformal_prediction":conformal_info,
        "psi":                 psi_results,
        "shap_importance":     {k: float(v) for k, v in shap_importance.items()},
    }

    metrics_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_h{horizon}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n  Métricas exportadas: {metrics_path}")

    # ── Resumen final ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Horizonte:        t+{horizon}")
    print(f"  Objective:        {best_params.get('objective','?')}")
    print(f"  Mejor MAE CV:     {optim_info['best_mae_cv']:.4f}  (solo train)")
    print(f"  Bagging:          {N_SEEDS_BAGGING} modelo(s)")
    print(f"  VAL  → MAE={eval_val['metricas_globales']['mae']:.3f}  "
          f"sMAPE={eval_val['metricas_globales']['smape']:.2f}%")
    print(f"  TEST → MAE={eval_test['metricas_globales']['mae']:.3f}  "
          f"sMAPE={eval_test['metricas_globales']['smape']:.2f}%")
    print(f"  Ganancia vs naive: {eval_test['ganancia_vs_baseline']:+.3f} MAE")
    cov = eval_test.get("cobertura_conformal", {})
    if cov:
        print(f"  IC80 cobertura test: {cov['ic80']['cobertura_real']:.1f}%  "
              f"(ancho ±{intervals_test['ic80']['q']:.1f})")
        print(f"  IC90 cobertura test: {cov['ic90']['cobertura_real']:.1f}%  "
              f"(ancho ±{intervals_test['ic90']['q']:.1f})")
    print(f"  Drift features:    {len(alert_features)}")
    print(f"\n  Modelo exportado:   {SAVE_MODEL_DIR / f'{MODEL_NAME}_h{horizon}.pkl'}")
    print(f"  Métricas exportadas: {SAVE_MODEL_DIR / f'{MODEL_NAME}_h{horizon}_metrics.json'}")

    return {
        "horizon":        horizon,
        "models":         models,
        "best_params":    best_params,
        "X_test":         X_test,
        "y_test":         y_test,
        "y_val_pred":     y_val_pred,
        "y_test_pred":    y_test_pred,
        "intervals_test": intervals_test,
        "features":       FEATURES,
        "eval_val":       eval_val,
        "eval_test":      eval_test,
        "psi_results":    psi_results,
        "shap_importance":shap_importance,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)
    df = pd.read_csv(
        PROCESSED_DIR / "feature_engineering.csv",
        index_col="fecha",
        parse_dates=True,
    ).sort_index()
    print(f"Dataset: {len(df)} filas | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    results = run_all(df)   # horizon=1 por default