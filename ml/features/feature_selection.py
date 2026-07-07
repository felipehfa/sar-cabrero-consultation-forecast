import re
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
#
# [HORIZON] Ya NO es una constante global — es un parámetro explícito de
# run_feature_selection() (default=1), igual que en model_xgboost.py. Esto
# permite importar este módulo desde un script driver y correr los 7
# horizontes en un loop, ej.:
#
#     import 03_feature_selection as fs   (o el nombre de módulo que uses)
#     for h in range(1, 8):
#         df = pd.read_csv(f"data/processed/feature_engineering_h{h}.csv",
#                           index_col="fecha", parse_dates=True).sort_index()
#         fs.run_feature_selection(df, horizon=h)
#
# El horizonte determina: el gap del TimeSeriesSplit (evita fuga del target
# hacia adelante), los textos impresos, y el sufijo de todos los archivos
# exportados (selected_features_h{horizon}.csv, etc.) — para que corridas
# de distinto horizonte no se sobrescriban entre sí.

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("ml/features/feature_selection_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_WINDOW_DAYS  = 365
TEST_WINDOW_DAYS = 365

# Selección por estabilidad
N_MODELS         = 30
CV_FOLDS         = 5
RANDOM_SEED_BASE = 42

# Importancia SHAP (recomendado para variables en distintas escalas)
IMPORTANCE_TYPE = "shap"

# Umbral de correlación DENTRO de cada familia, antes de stability selection
FAMILY_CORR_THRESHOLD = 0.90

# Umbral de correlación post-selección (red de seguridad, sobre el set final)
CORR_THRESHOLD_FINAL = 0.95

# Varianza mínima sobre X_train
CONSTANT_VAR_THRESHOLD = 1e-6

# Presencia mínima: % de modelos donde importancia > 0
PRESENCE_THRESHOLD = 0.6

# PSI — drift train→val
PSI_BINS        = 10
HARD_PSI_LIMIT  = 0.5   # excluye candidatas con drift extremo
PSI_EPS         = 1e-4

# Diversificación por dominio
MIN_PER_DOMAIN = 1

# Piso de score para que la diversificación no fuerce candidatas muy
# débiles solo para llenar la cuota de un dominio poco poblado. Se expresa
# como fracción del mejor adjusted_score observado entre las candidatas
# elegibles (ej. 0.5 = no forzar nada por debajo de la mitad del mejor score).
MIN_SCORE_RATIO_FOR_DIVERSITY = 0.5

# Rango de k para la curva MAE vs k
K_MIN = 3
K_MAX = None   # se determina dinámicamente

# Número de semillas sobre las que se promedia cada punto de la curva
# MAE/sMAPE vs k.
N_SEEDS_K_CURVE = 5

# Parámetros del modelo rápido usado SOLO para rankear
# candidatas dentro de una misma familia (más liviano que XGB_PARAMS porque
# las familias suelen ser pequeñas y no necesita ser tan profundo).
FAMILY_MODEL_PARAMS = dict(
    n_estimators     = 100,
    max_depth        = 3,
    learning_rate    = 0.08,
    subsample        = 0.8,
    colsample_bytree = 1.0,
    n_jobs           = -1,
    verbosity        = 0,
    eval_metric      = "mae",
)

# Parámetros del XGBRegressor para selección de variables
XGB_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    n_jobs           = -1,
    verbosity        = 0,
    eval_metric      = "mae",
)

# [MULTI-HORIZONTE] Ya no se usa una lista fija de exclusión — FEATURES se
# calcula dinámicamente excluyendo TODAS las columnas target_h* (ver
# run_feature_selection), porque ahora el dataset trae una columna de
# target por cada horizonte, no solo "target".

# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    sMAPE = 100 * mean( 2*|y - ŷ| / (|y| + |ŷ| + ε) )

    ε=1.0 estabiliza el denominador cuando ambos son 0 (0 consultas reales
    y 0 predichas → contribución 0, no NaN).
    """
    numerator   = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(numerator / denominator))


# ──────────────────────────────────────────────────────────────────────────────
# Familias de features — para reducir redundancia intra-familia
# ──────────────────────────────────────────────────────────────────────────────

# Sufijos añadidos por el feature engineering. Se buscan anclados al final
# del nombre; el primero que haga match determina la raíz de la familia.
_SUFFIX_PATTERNS = [
    r"_trend_\d+_\d+$",     # atenciones_trend_14_28
    r"_pct_change_\d+$",    # atenciones_pct_change_7
    r"_vs_week$",           # atenciones_vs_week
    r"_lag\d+$",            # *_lag1, *_lag28
    r"_ma\d+$",             # *_ma3, *_ma28
    r"_std\d+$",            # *_std7, *_std14
    r"_max\d+$",            # atenciones_max10 (NO pm10_max: sin dígito final)
    r"_min\d+$",            # atenciones_min10
    r"_delta\d+$",          # *_delta1
]


def get_family_root(feature: str) -> str:
    """
    Extrae la raíz de familia de una feature, eliminando sufijos de
    feature engineering (lag/ma/std/max/min/delta/trend/pct_change/vs_week).

    Ej: 'pm2_5_mean_ma7' -> 'pm2_5_mean'
        'contaminacion_total_std7' -> 'contaminacion_total'
        'atenciones_trend_14_28' -> 'atenciones'
        'dow_cos' -> 'dow_cos' (sin sufijo reconocido, queda igual)
    """
    for pattern in _SUFFIX_PATTERNS:
        m = re.search(pattern, feature)
        if m:
            return feature[: m.start()]
    return feature


# ──────────────────────────────────────────────────────────────────────────────
# Dominios de features — diversificación en selección de k
# ──────────────────────────────────────────────────────────────────────────────

def get_domain(feature: str) -> str:
    """
    Clasifica una feature en un dominio amplio, usado solo para
    garantizar diversidad en el pool de candidatas de la curva k, no para
    el cálculo de stability_score.
    """
    fam = get_family_root(feature)

    # "atenciones" puede aparecer como prefijo
    # (atenciones_lag1, atenciones_ma7, ...) o como sufijo en la variable
    # base cruda (total_atenciones). Detectar solo el prefijo dejaba a
    # total_atenciones caer en "otros" como único miembro de ese dominio,
    # forzándola a entrar por la cuota MIN_PER_DOMAIN sin importar su score
    # o su drift.
    if "atenciones" in fam:
        return "atenciones"

    calendario_exact = {
        "dow", "dow_sin", "dow_cos",
        "month", "month_sin", "month_cos",
        "is_holiday", "before_holiday", "after_holiday",
    }
    if fam in calendario_exact or fam.startswith("est_"):
        return "calendario"

    if fam.startswith(("pm2_5", "pm10", "no2", "ozono", "contaminacion_total")):
        return "clima_contaminacion"

    if fam.startswith(("temperatura", "humedad", "precipitacion", "temp_range", "pm25_pm10_ratio")):
        return "clima_meteo"

    return "otros"


# ──────────────────────────────────────────────────────────────────────────────
# PSI — Population Stability Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_psi(train_vals, val_vals, bins: int = PSI_BINS, eps: float = PSI_EPS) -> float:
    """
    Calcula PSI entre la distribución de train y la de val para una
    variable continua, usando cuantiles de train como bordes de bins.

    PSI < 0.10 : sin cambio relevante
    PSI < 0.25 : cambio moderado (watch)
    PSI >= 0.25: cambio mayor (drift)

    Returns NaN si no hay suficiente variabilidad para definir bins.
    """
    train_vals = pd.Series(train_vals).dropna()
    val_vals   = pd.Series(val_vals).dropna()

    if train_vals.nunique() < 2 or len(val_vals) == 0:
        return np.nan

    quantiles = np.unique(np.quantile(train_vals, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return np.nan

    quantiles = quantiles.astype(float)
    quantiles[0]  = -np.inf
    quantiles[-1] = np.inf

    train_bins = pd.cut(train_vals, bins=quantiles)
    val_bins   = pd.cut(val_vals, bins=quantiles)

    train_dist = train_bins.value_counts(normalize=True).sort_index()
    val_dist   = val_bins.value_counts(normalize=True).sort_index()

    psi = 0.0
    for b in train_dist.index:
        p = float(train_dist.get(b, 0.0)) + eps
        q = float(val_dist.get(b, 0.0)) + eps
        psi += (p - q) * np.log(p / q)

    return float(psi)


# ──────────────────────────────────────────────────────────────────────────────
# Split temporal por ventana fija (reemplaza el split por porcentaje)
# ──────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple:
    """
    Split temporal por ventana fija de un año calendario completo:

        Train : resto del historial
        Val   : 365 días (año anterior al test)
        Test  : últimos 365 días

    Esto asegura que Val y Test cubran un ciclo estacional completo cada
    uno, evitando que el PSI train→val quede inflado artificialmente por
    comparar una mezcla de estaciones (train) contra una ventana que no
    representa el ciclo anual completo (val).

    Returns: (train, val, test) — Test se calcula por consistencia con el
    script de entrenamiento pero NO se usa en este script de selección.
    """
    df = df.sort_index()

    last_date = df.index.max()

    test_start = last_date - pd.Timedelta(days=TEST_WINDOW_DAYS - 1)

    val_end   = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=VAL_WINDOW_DAYS - 1)

    train = df[df.index < val_start]
    val   = df[(df.index >= val_start) & (df.index <= val_end)]
    test  = df[df.index >= test_start]

    print("\nSplit temporal (ventana fija de 365 días)")
    print(f"  Train: {len(train)} filas | {train.index.min().date()} → {train.index.max().date()}")
    print(f"  Val:   {len(val)} filas | {val.index.min().date()} → {val.index.max().date()}")
    print(f"  Test:  {len(test)} filas | {test.index.min().date()} → {test.index.max().date()}  "
          f"(calculado por consistencia, NO usado en este script)")

    return train, val, test


def remove_constant_features(X: pd.DataFrame,
                              threshold: float = CONSTANT_VAR_THRESHOLD) -> tuple:
    """
    Elimina features con varianza cero o cuasi-cero sobre X_train.

    Returns: (selected, dropped)
    """
    variances = X.var(axis=0, skipna=True)
    low_var   = variances[variances <= threshold].index.tolist()
    nan_feat  = variances[variances.isna()].index.tolist()
    to_drop   = sorted(set(low_var + nan_feat))
    selected  = [c for c in X.columns if c not in to_drop]
    return selected, to_drop


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Selección por estabilidad (helper de importancia, usado también
# por el criterio de campeón de familia)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_importance(model: XGBRegressor,
                        X_fold: pd.DataFrame,
                        imp_type: str) -> dict:
    """
    Calcula importancia SHAP mean abs sobre X_fold.
    Fallback a 'weight' si shap no está disponible.
    """
    booster = model.get_booster()

    if imp_type == "shap":
        try:
            import shap
            explainer = shap.TreeExplainer(booster)
            sv = explainer(X_fold).values
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            mean_abs = np.abs(sv).mean(axis=0)
            return dict(zip(X_fold.columns, mean_abs.tolist()))
        except ImportError:
            print("    ⚠ shap no disponible — usando 'weight' como fallback")
            imp_type = "weight"

    raw = booster.get_score(importance_type=imp_type)
    return {feat: float(raw.get(feat, 0.0)) for feat in X_fold.columns}


# ──────────────────────────────────────────────────────────────────────────────
# Paso 1.5 — Reducción de redundancia intra-familia
# ──────────────────────────────────────────────────────────────────────────────

def _compute_family_relevance(
    X_family: pd.DataFrame,
    y_train:  pd.Series,
    importance_type: str = IMPORTANCE_TYPE,
) -> dict:
    """
    Calcula relevancia de cada variable DENTRO de una familia
    entrenando un único modelo XGBoost sobre solo esas variables y tomando
    su importancia SHAP.

    Es una estimación en una sola muestra (no CV) porque su único propósito
    es rankear candidatas DENTRO de la familia antes de deduplicar por
    correlación — la estabilidad real se sigue midiendo después, en el
    Paso 2, sobre las que sobrevivan.
    """
    params = {**FAMILY_MODEL_PARAMS, "random_state": RANDOM_SEED_BASE}
    model  = XGBRegressor(**params)
    model.fit(X_family, y_train, verbose=False)
    return _compute_importance(model, X_family, importance_type)


def reduce_family_redundancy(
    X_train:         pd.DataFrame,
    y_train:         pd.Series,
    threshold:       float = FAMILY_CORR_THRESHOLD,
    importance_type: str = IMPORTANCE_TYPE,
) -> tuple:
    """
    Dentro de cada familia (misma raíz tras quitar sufijos de feature
    engineering), elimina variables muy correlacionadas entre sí,
    quedándose con la de mayor importancia SHAP según un modelo entrenado
    solo con esa familia.

    Esto se hace ANTES de stability selection para que ningún bloque de
    features (ej. contaminación, con ~10 variantes por contaminante) se
    fragmente en tantas variantes que ninguna acumule presencia suficiente
    frente a bloques menos redundantes (ej. atenciones_*).

    Returns: (keep, dropped_by_family) donde dropped_by_family es un dict
    {familia: [features eliminadas]} para trazabilidad en el reporte.
    """
    families = defaultdict(list)
    for col in X_train.columns:
        families[get_family_root(col)].append(col)

    keep = []
    dropped_by_family = {}

    for fam, cols in families.items():
        if len(cols) == 1:
            keep.extend(cols)
            continue

        sub  = X_train[cols].fillna(0)
        corr = sub.corr().abs()

        # Relevancia por importancia SHAP de un modelo entrenado solo con
        # las variables de esta familia, en vez de correlación de Pearson
        # simple con el target.
        relevance = _compute_family_relevance(sub, y_train, importance_type)

        ordered = sorted(cols, key=lambda c: relevance.get(c, 0.0), reverse=True)

        fam_keep = []
        fam_dropped = []
        for c in ordered:
            if c in fam_dropped:
                continue
            fam_keep.append(c)
            highly_corr = [
                o for o in ordered
                if o != c and o not in fam_dropped and o not in fam_keep
                and corr.loc[c, o] > threshold
            ]
            fam_dropped.extend(highly_corr)

        keep.extend(fam_keep)
        if fam_dropped:
            dropped_by_family[fam] = fam_dropped

    return keep, dropped_by_family


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Selección por estabilidad (solo sobre X_train)
# ──────────────────────────────────────────────────────────────────────────────

def select_features_by_stability(
    X_train:            pd.DataFrame,
    y_train:            pd.Series,
    n_models:           int   = N_MODELS,
    cv_folds:           int   = CV_FOLDS,
    random_seed_base:   int   = RANDOM_SEED_BASE,
    presence_threshold: float = PRESENCE_THRESHOLD,
    importance_type:    str   = IMPORTANCE_TYPE,
    horizon:            int   = 1,
) -> pd.DataFrame:
    """
    Calcula estabilidad de features entrenando N modelos con TimeSeriesSplit
    sobre X_train exclusivamente. `horizon` define el gap del split (evita
    fuga del target hacia adelante).

    stability_score = (1 / (1 + CV)) * presence_pct
    Importancia: SHAP mean abs

    Returns: DataFrame ordenado por stability_score descendente.
    """
    tscv = TimeSeriesSplit(n_splits=cv_folds, gap=horizon)

    all_importances: dict[str, list] = {feat: [] for feat in X_train.columns}
    all_mae:   list[float] = []
    all_smape: list[float] = []

    print(f"\n  Entrenando {n_models} modelos (importancia='{importance_type}', "
          f"gap={horizon} días)...")

    for i in range(n_models):
        seed      = random_seed_base + i
        fold_imps = {feat: [] for feat in X_train.columns}
        fold_mae_list   = []
        fold_smape_list = []

        for _, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            X_tr = X_train.iloc[tr_idx].fillna(0)
            X_va = X_train.iloc[va_idx].fillna(0)
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            if len(y_va) == 0:
                continue

            params = {**XGB_PARAMS, "random_state": seed}
            model  = XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)

            y_pred = model.predict(X_va)
            y_pred = np.clip(y_pred, 0, None)

            fold_mae_list.append(mean_absolute_error(y_va, y_pred))
            fold_smape_list.append(smape(y_va.values, y_pred))

            imp_dict = _compute_importance(model, X_va, importance_type)
            for feat in X_train.columns:
                fold_imps[feat].append(imp_dict.get(feat, 0.0))

        if not fold_mae_list:
            continue

        all_mae.append(float(np.mean(fold_mae_list)))
        all_smape.append(float(np.mean(fold_smape_list)))

        for feat in X_train.columns:
            vals = fold_imps[feat]
            all_importances[feat].append(float(np.mean(vals)) if vals else 0.0)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Modelo {i+1:>2}/{n_models}  "
                  f"MAE_fold={all_mae[-1]:.2f}  "
                  f"sMAPE_fold={all_smape[-1]:.2f}%")

    results = []
    for feat in X_train.columns:
        gains    = np.array(all_importances[feat])
        mean_g   = float(np.mean(gains))
        std_g    = float(np.std(gains))
        median_g = float(np.median(gains))
        presence = float(np.mean(gains > 0))

        if mean_g > 0:
            cv    = std_g / mean_g
            score = (1.0 / (1.0 + cv)) * presence
        else:
            cv    = np.nan
            score = 0.0

        results.append({
            "feature":         feat,
            "domain":          get_domain(feat),
            "mean_imp":        round(mean_g,   6),
            "std_imp":         round(std_g,    6),
            "median_imp":      round(median_g, 6),
            "cv_imp":          round(cv, 4) if not np.isnan(cv) else np.nan,
            "presence_pct":    round(presence, 4),
            "stability_score": round(score,    6),
        })

    stability_df = (
        pd.DataFrame(results)
        .sort_values("stability_score", ascending=False)
        .reset_index(drop=True)
    )
    stability_df["passes_presence"] = stability_df["presence_pct"] >= presence_threshold

    print(f"\n  MAE medio global (folds internos):   {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f}")
    print(f"  sMAPE medio global (folds internos): {np.mean(all_smape):.2f}% ± {np.std(all_smape):.2f}%")
    print(f"  Features con presence >= {presence_threshold}: "
          f"{stability_df['passes_presence'].sum()} / {len(stability_df)}")

    return stability_df


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2.5 — PSI y adjusted_score
# ──────────────────────────────────────────────────────────────────────────────

def add_psi_and_adjusted_score(
    stability_df:    pd.DataFrame,
    X_train:         pd.DataFrame,
    X_val:           pd.DataFrame,
    hard_psi_limit:  float = HARD_PSI_LIMIT,
) -> pd.DataFrame:
    """
    Añade PSI train→val y adjusted_score (penalizado por drift) a
    stability_df. Solo se calcula sobre features que pasan el filtro de
    presencia, para no gastar cómputo en descartadas.

    adjusted_score = stability_score / (1 + PSI)

    Features con PSI >= hard_psi_limit quedan marcadas como excluidas por
    drift extremo (excluded_hard_drift=True), independientemente de su
    stability_score.
    """
    df = stability_df.copy()
    df["psi_train_val"] = np.nan

    candidates_mask = df["passes_presence"] & df["feature"].isin(X_train.columns) & df["feature"].isin(X_val.columns)

    print(f"\n  Calculando PSI train→val sobre {candidates_mask.sum()} candidatas...")

    for idx in df[candidates_mask].index:
        feat = df.at[idx, "feature"]
        psi  = compute_psi(X_train[feat], X_val[feat])
        df.at[idx, "psi_train_val"] = psi

    df["psi_train_val"] = df["psi_train_val"].fillna(0.0)
    df["adjusted_score"] = df["stability_score"] / (1.0 + df["psi_train_val"])
    df["excluded_hard_drift"] = df["psi_train_val"] >= hard_psi_limit

    n_hard = int(df["excluded_hard_drift"].sum())
    if n_hard:
        print(f"  ⚠ {n_hard} candidatas excluidas por drift extremo (PSI >= {hard_psi_limit}):")
        for f in df.loc[df["excluded_hard_drift"], "feature"]:
            psi_val = df.loc[df["feature"] == f, "psi_train_val"].iloc[0]
            print(f"    {f:<45} PSI={psi_val:.3f}")

    return df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Diversificación por dominio
# ──────────────────────────────────────────────────────────────────────────────

def diversify_candidates(
    stability_df:          pd.DataFrame,
    min_per_domain:        int   = MIN_PER_DOMAIN,
    min_score_ratio:       float = MIN_SCORE_RATIO_FOR_DIVERSITY,
) -> list:
    """
    Reordena las candidatas (ya filtradas por presencia y drift duro) para
    garantizar representación mínima de cada dominio al frente de la lista,
    preservando el orden por adjusted_score dentro de cada pasada.

    La cuota por dominio (min_per_domain) solo se llena con candidatas cuyo
    adjusted_score sea >= min_score_ratio * mejor_score_elegible. Sin este
    piso, un dominio con una sola variable débil y con drift alto se cuela
    al set final solo por ocupar una cuota vacía, no por mérito.

    Primera pasada: toma, en orden de adjusted_score, hasta min_per_domain
    features de cada dominio no representado todavía, siempre que superen
    el piso de score.
    Segunda pasada: añade el resto en orden de adjusted_score global.
    """
    ordered_all = stability_df["feature"].tolist()
    domain_map  = dict(zip(stability_df["feature"], stability_df["domain"]))
    score_map   = dict(zip(stability_df["feature"], stability_df["adjusted_score"]))

    best_score  = stability_df["adjusted_score"].max() if len(stability_df) else 0.0
    score_floor = best_score * min_score_ratio

    seen = set()
    result = []
    domain_count = defaultdict(int)

    for f in ordered_all:
        d = domain_map[f]
        if domain_count[d] < min_per_domain and score_map[f] >= score_floor:
            result.append(f)
            seen.add(f)
            domain_count[d] += 1

    for f in ordered_all:
        if f not in seen:
            result.append(f)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 — Selección del número óptimo de features usando Validation MAE
# ──────────────────────────────────────────────────────────────────────────────

def select_k_features_via_val(
    stability_df: pd.DataFrame,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_val:        pd.DataFrame,
    y_val:        pd.Series,
    k_min:        int = K_MIN,
    k_max:        int = None,
    seed:         int = RANDOM_SEED_BASE,
    n_seeds:      int = N_SEEDS_K_CURVE,
) -> tuple:
    """
    Determina el número óptimo de features evaluando MAE + sMAPE en X_val,
    sobre un pool de candidatas ya diversificado por dominio y filtrado por
    presencia + drift duro.

    Cada punto de la curva (cada k) se promedia sobre n_seeds semillas
    distintas.

    SE = std(MAE entre semillas EN EL k DEL MÍNIMO) / sqrt(n_seeds)
    threshold = best_mae + SE
    → k más pequeño cuyo MAE_promedio <= threshold

    Returns:
        best_k         : número óptimo de features
        selected_feats : lista de features seleccionadas
        metric_curve   : DataFrame con la curva MAE/sMAPE vs k (promediada,
                          con std entre semillas por k)
    """
    eligible_df = stability_df[
        stability_df["passes_presence"] & ~stability_df["excluded_hard_drift"]
    ]

    candidates = diversify_candidates(
        eligible_df,
        min_per_domain=MIN_PER_DOMAIN,
        min_score_ratio=MIN_SCORE_RATIO_FOR_DIVERSITY,
    )
    candidates = [f for f in candidates if f in X_train.columns and f in X_val.columns]

    if not candidates:
        print("  ⚠ Sin candidatas después del filtro de presencia/drift.")
        return 0, [], pd.DataFrame()

    k_max_eff = min(k_max or len(candidates), len(candidates))
    k_range   = range(k_min, k_max_eff + 1)
    seeds     = [seed + i for i in range(n_seeds)]

    print(f"\n  Evaluando curva MAE/sMAPE vs k features (k={k_min}..{k_max_eff}, "
          f"promediado sobre {n_seeds} semillas)...")
    print(f"  Orden de candidatas diversificado por dominio (primeras 10): {candidates[:10]}")

    records = []
    for k in k_range:
        feats = candidates[:k]

        mae_per_seed   = []
        smape_per_seed = []
        for s in seeds:
            params = {**XGB_PARAMS, "random_state": s, "n_estimators": 300}
            model  = XGBRegressor(**params)
            model.fit(
                X_train[feats].fillna(0),
                y_train,
                eval_set=[(X_val[feats].fillna(0), y_val)],
                verbose=False,
            )

            y_pred = np.clip(model.predict(X_val[feats].fillna(0)), 0, None)
            mae_per_seed.append(mean_absolute_error(y_val, y_pred))
            smape_per_seed.append(smape(y_val.values, y_pred))

        mae_k     = float(np.mean(mae_per_seed))
        mae_k_std = float(np.std(mae_per_seed))
        smape_k   = float(np.mean(smape_per_seed))

        records.append({
            "k": k,
            "mae_val": round(mae_k, 4),
            "mae_val_std_seeds": round(mae_k_std, 4),
            "smape_val": round(smape_k, 4),
        })

        if k % 5 == 0 or k == k_min or k == k_max_eff:
            print(f"    k={k:>3}  MAE={mae_k:.2f} (±{mae_k_std:.2f} entre semillas)  "
                  f"sMAPE={smape_k:.2f}%")

    metric_curve = pd.DataFrame(records)

    # ── Regla de 1 error estándar, correctamente calculada ─────────────
    argmin_idx = metric_curve["mae_val"].idxmin()
    argmin_k   = int(metric_curve.loc[argmin_idx, "k"])
    best_mae   = float(metric_curve.loc[argmin_idx, "mae_val"])
    std_at_best = float(metric_curve.loc[argmin_idx, "mae_val_std_seeds"])
    se_at_best  = std_at_best / np.sqrt(n_seeds)
    threshold   = best_mae + se_at_best

    parsimonious = metric_curve[metric_curve["mae_val"] <= threshold]
    best_k       = int(parsimonious.iloc[0]["k"])
    best_k_row   = metric_curve[metric_curve["k"] == best_k].iloc[0]

    print(f"\n  MAE mínimo: {best_mae:.2f} (k={argmin_k}, std_entre_semillas={std_at_best:.3f})")
    print(f"  1-SE (std_en_mínimo/√{n_seeds}): {se_at_best:.3f}  →  threshold: {threshold:.3f}")
    print(f"  k óptimo (parsimonia, 1-SE rule): {best_k}  "
          f"MAE={best_k_row['mae_val']:.2f}  sMAPE={best_k_row['smape_val']:.2f}%")

    selected_feats = candidates[:best_k]
    return best_k, selected_feats, metric_curve


# ──────────────────────────────────────────────────────────────────────────────
# Paso 4 — Eliminación de correlación post-selección (red de seguridad)
# ──────────────────────────────────────────────────────────────────────────────

def remove_correlated_after_selection(
    X_train:      pd.DataFrame,
    selected:     list,
    stability_df: pd.DataFrame,
    threshold:    float = CORR_THRESHOLD_FINAL,
    score_col:    str = "adjusted_score",
) -> tuple:
    """
    Elimina correlación entre las features YA seleccionadas.
    Con la deduplicación por familia aplicada antes, esto debería eliminar
    poco o nada — queda como red de seguridad para redundancias que crucen
    familias distintas (ej. una variable de atenciones correlacionada con
    una climática).

    Returns: (keep, removed)
    """
    if not selected:
        return [], []

    avail     = [f for f in selected if f in X_train.columns]
    X_sub     = X_train[avail].fillna(0)
    score_map = stability_df.set_index("feature")[score_col].to_dict()
    ordered   = sorted(avail, key=lambda f: score_map.get(f, 0.0), reverse=True)

    corr    = X_sub[ordered].corr().abs()
    keep    = []
    removed = []

    for feat in ordered:
        if feat in removed:
            continue
        keep.append(feat)
        highly_corr = [
            other for other in ordered
            if other != feat
            and other not in removed
            and corr.loc[feat, other] > threshold
        ]
        removed.extend(highly_corr)

    return keep, removed


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal — encapsulado en función para poder importarlo y
# correr los 7 horizontes desde un script driver.
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_selection(df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Corre el pipeline completo de selección de variables para un horizonte
    de predicción dado.

    horizon determina:
      - el gap del TimeSeriesSplit (Paso 2 y en la lógica de CV)
      - los textos impresos
      - el sufijo de TODOS los archivos exportados en ml/features/:
            selected_features_h{horizon}.csv
            feature_selection_report_h{horizon}.csv
            metric_curve_k_features_h{horizon}.csv
            family_redundancy_dropped_h{horizon}.csv

    Para correr los 7 horizontes desde otro script:

        import importlib
        fs = importlib.import_module("03_feature_selection")
        for h in range(1, 8):
            df = pd.read_csv(f"data/processed/feature_engineering_h{h}.csv",
                              index_col="fecha", parse_dates=True).sort_index()
            fs.run_feature_selection(df, horizon=h)

    Returns: dict con final_features, stability_df, metric_curve y demás
    resultados intermedios, por si se quiere inspeccionar programáticamente
    en vez de solo leer los CSV exportados.
    """
    pd.set_option("future.no_silent_downcasting", True)

    n_filas = len(df)
    print(f"Dataset total: {n_filas} filas")

    # [MULTI-HORIZONTE] target_h{horizon} es la columna de este horizonte;
    # se excluyen TODAS las columnas target_h* de las features candidatas
    # (no solo la de este horizonte) para evitar fuga cruzada entre
    # horizontes.
    target_col = f"target_h{horizon}"
    FEATURES = [c for c in df.columns if not c.startswith("target_h")]
    print(f"Features iniciales (excluyendo columnas target_h*): {len(FEATURES)}")

    # ── Split temporal por ventana fija de 365 días ────────────────
    train, val, _test = split_data(df)   # test no se usa en este script

    # [MULTI-HORIZONTE] target_h{horizon} trae NaN de cola (la magnitud del
    # NaN depende del horizonte — a mayor horizonte, más filas finales sin
    # target válido). Se descartan aquí, antes de armar X/y, para no
    # entrenar/evaluar con target faltante.
    train = train.dropna(subset=[target_col])
    val   = val.dropna(subset=[target_col])

    X_train_full = train[FEATURES]
    y_train_full = train[target_col]
    X_val        = val[FEATURES]
    y_val        = val[target_col]

    # ── Paso 1: Eliminar constantes ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASO 1 — Eliminación de features constantes (sobre X_train)  [h={horizon}]")
    print("="*60)

    selected_not_const, dropped_const = remove_constant_features(
        X_train_full, threshold=CONSTANT_VAR_THRESHOLD
    )
    print(f"  Eliminadas (varianza ≈ 0): {len(dropped_const)}")
    if dropped_const:
        print(f"    {dropped_const}")
    print(f"  Features restantes: {len(selected_not_const)}")

    X_train = X_train_full[selected_not_const]

    # ── Paso 1.5: Reducción de redundancia intra-familia ──
    print(f"\n{'='*60}")
    print("PASO 1.5 — Reducción de redundancia intra-familia (sobre X_train)")
    print(f"  Umbral de correlación intra-familia: {FAMILY_CORR_THRESHOLD}")
    print(f"  Criterio de campeón: importancia SHAP de modelo por familia (no Pearson)")
    print("="*60)

    deduped_features, dropped_by_family = reduce_family_redundancy(
        X_train, y_train_full, threshold=FAMILY_CORR_THRESHOLD
    )

    n_dropped_family = sum(len(v) for v in dropped_by_family.values())
    print(f"  Familias con reducción: {len(dropped_by_family)}")
    print(f"  Features eliminadas por redundancia intra-familia: {n_dropped_family}")
    for fam, cols in sorted(dropped_by_family.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    [{fam}] -{len(cols)}: {cols}")
    print(f"  Features restantes tras deduplicación: {len(deduped_features)}")

    X_train = X_train[deduped_features]

    # ── Paso 2: Selección por estabilidad ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASO 2 — Selección por estabilidad (sobre X_train)  [horizonte t+{horizon}]")
    print(f"  N_MODELS={N_MODELS}, CV_FOLDS={CV_FOLDS}, gap={horizon} días")
    print(f"  Importancia: '{IMPORTANCE_TYPE}'")
    print(f"  Métricas evaluación: MAE + sMAPE")
    print("="*60)

    stability_df = select_features_by_stability(
        X_train,
        y_train_full,
        n_models           = N_MODELS,
        cv_folds            = CV_FOLDS,
        random_seed_base    = RANDOM_SEED_BASE,
        presence_threshold  = PRESENCE_THRESHOLD,
        importance_type     = IMPORTANCE_TYPE,
        horizon             = horizon,
    )

    print(f"\nTop 20 features por stability_score (pre-drift):")
    print(
        stability_df[["feature", "domain", "mean_imp", "cv_imp",
                       "presence_pct", "stability_score"]]
        .head(20)
        .to_string(index=False)
    )

    # ── Paso 2.5: PSI y adjusted_score ──────────────────────
    print(f"\n{'='*60}")
    print("PASO 2.5 — PSI train→val y adjusted_score")
    print(f"  HARD_PSI_LIMIT: {HARD_PSI_LIMIT}")
    print("="*60)

    stability_df = add_psi_and_adjusted_score(
        stability_df, X_train, X_val[deduped_features],
        hard_psi_limit=HARD_PSI_LIMIT,
    )

    print(f"\nTop 20 features por adjusted_score (post-drift):")
    print(
        stability_df[["feature", "domain", "stability_score",
                       "psi_train_val", "adjusted_score", "excluded_hard_drift"]]
        .head(20)
        .to_string(index=False)
    )

    # ── Paso 3: k óptimo via Validation MAE, con diversificación ───
    print(f"\n{'='*60}")
    print("PASO 3 — Selección de k óptimo via Validation MAE")
    print("="*60)

    best_k, selected_feats, metric_curve = select_k_features_via_val(
        stability_df = stability_df,
        X_train      = X_train,
        y_train      = y_train_full,
        X_val        = X_val[deduped_features],
        y_val        = y_val,
        k_min        = K_MIN,
        k_max        = K_MAX,
        seed         = RANDOM_SEED_BASE,
        n_seeds      = N_SEEDS_K_CURVE,
    )

    print(f"\n  Features seleccionadas (k={best_k}):")
    for f in selected_feats:
        row = stability_df[stability_df["feature"] == f].iloc[0]
        print(f"    {f:<45} domain={row['domain']:<20} adj_score={row['adjusted_score']:.4f}  "
              f"psi={row['psi_train_val']:.3f}  presence={row['presence_pct']:.2f}")

    # ── Paso 4: Eliminación de correlación post-selección (red de seguridad) ─
    print(f"\n{'='*60}")
    print(f"PASO 4 — Eliminación de correlación post-selección (red de seguridad)")
    print(f"  Umbral correlación: {CORR_THRESHOLD_FINAL}")
    print("="*60)

    final_features, removed_corr = remove_correlated_after_selection(
        X_train      = X_train,
        selected     = selected_feats,
        stability_df = stability_df,
        threshold    = CORR_THRESHOLD_FINAL,
        score_col    = "adjusted_score",
    )

    print(f"  Eliminadas por correlación: {len(removed_corr)}")
    if removed_corr:
        print(f"    {removed_corr}")
    print(f"  Features finales: {len(final_features)}")

    # ── Paso 5: Exportar resultados (nombres con sufijo _h{horizon}) ─────────
    print(f"\n{'='*60}")
    print(f"PASO 5 — Exportar resultados  [h={horizon}]")
    print("="*60)

    selected_path      = OUTPUT_DIR / f"selected_features_h{horizon}.csv"
    report_path        = OUTPUT_DIR / f"feature_selection_report_h{horizon}.csv"
    metric_curve_path  = OUTPUT_DIR / f"metric_curve_k_features_h{horizon}.csv"
    family_dropped_path = OUTPUT_DIR / f"family_redundancy_dropped_h{horizon}.csv"

    if final_features:
        pd.DataFrame({"feature": final_features}).to_csv(selected_path, index=False)
        print(f"  ✓ Features seleccionadas: {selected_path}")

        export_df = stability_df.copy()
        export_df["selected_final"] = export_df["feature"].isin(final_features)
        export_df.to_csv(report_path, index=False)
        print(f"  ✓ Reporte detallado:      {report_path}")

        if not metric_curve.empty:
            metric_curve.to_csv(metric_curve_path, index=False)
            print(f"  ✓ Curva MAE/sMAPE vs k:   {metric_curve_path}")

        # Trazabilidad de deduplicación intra-familia
        family_rows = [
            {"family": fam, "dropped_feature": col}
            for fam, cols in dropped_by_family.items()
            for col in cols
        ]
        if family_rows:
            pd.DataFrame(family_rows).to_csv(family_dropped_path, index=False)
            print(f"  ✓ Redundancia intra-familia eliminada: {family_dropped_path}")

        print(f"\n  ℹ El feature_engineering de origen NO fue modificado.")
        print(f"    En model_xgboost.py, para usar estas features (horizon={horizon}):")
        print(f"    FEATURES = pd.read_csv('{selected_path}')['feature'].tolist()")

    else:
        print("  ⚠ No se seleccionaron features — revisa PRESENCE_THRESHOLD, "
              "HARD_PSI_LIMIT y K_MIN.")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESUMEN  [horizonte t+{horizon}]")
    print("="*60)
    print(f"  Dataset total:                              {n_filas}")
    print(f"  Features iniciales:                         {len(FEATURES)}")
    print(f"  Después de eliminar constantes:             {len(selected_not_const)}")
    print(f"  Después de deduplicación intra-familia:     {len(deduped_features)}")
    print(f"  Candidatas (presence >= {PRESENCE_THRESHOLD}):            "
          f"{stability_df['passes_presence'].sum()}")
    print(f"  Excluidas por drift extremo:                "
          f"{int(stability_df['excluded_hard_drift'].sum())}")
    print(f"  k óptimo (1-SE rule, Val MAE):                {best_k}")
    print(f"  Eliminadas por correlación final:            {len(removed_corr)}")
    print(f"  Features finales:                            {len(final_features)}")
    print()
    print(f"  Parámetros:")
    print(f"    HORIZON (gap):                 {horizon}")
    print(f"    IMPORTANCE_TYPE:               {IMPORTANCE_TYPE}")
    print(f"    N_MODELS:                      {N_MODELS}")
    print(f"    CV_FOLDS:                      {CV_FOLDS}")
    print(f"    PRESENCE_THRESHOLD:            {PRESENCE_THRESHOLD}")
    print(f"    FAMILY_CORR_THRESHOLD:         {FAMILY_CORR_THRESHOLD}")
    print(f"    CORR_THRESHOLD_FINAL:          {CORR_THRESHOLD_FINAL}")
    print(f"    HARD_PSI_LIMIT:                {HARD_PSI_LIMIT}")
    print(f"    MIN_PER_DOMAIN:                {MIN_PER_DOMAIN}")
    print(f"    MIN_SCORE_RATIO_FOR_DIVERSITY: {MIN_SCORE_RATIO_FOR_DIVERSITY}")
    print(f"    N_SEEDS_K_CURVE:               {N_SEEDS_K_CURVE}")
    print(f"    VAL_WINDOW_DAYS:               {VAL_WINDOW_DAYS}")
    print(f"    TEST_WINDOW_DAYS:              {TEST_WINDOW_DAYS}")
    print()
    if final_features:
        print("  FEATURES FINALES:")
        for f in final_features:
            row = stability_df[stability_df["feature"] == f].iloc[0]
            print(f"    {f:<45} domain={row['domain']:<20} "
                  f"adj_score={row['adjusted_score']:.4f}  psi={row['psi_train_val']:.3f}")

    return {
        "horizon":            horizon,
        "final_features":     final_features,
        "stability_df":       stability_df,
        "metric_curve":       metric_curve,
        "dropped_by_family":  dropped_by_family,
        "removed_corr":       removed_corr,
        "best_k":             best_k,
        "selected_path":      selected_path,
        "report_path":        report_path,
        "metric_curve_path":  metric_curve_path,
        "family_dropped_path":family_dropped_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint (ejecución standalone con horizon=1 por default)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando dataset...")
    df = pd.read_csv(
        PROCESSED_DIR / "feature_engineering.csv",
        index_col="fecha",
        parse_dates=True,
    ).sort_index()

    run_feature_selection(df, horizon=1)