"""
pipeline/feature_importance.py

Extrae importancia de variables (gain) de cada modelo del ensamble,
promediada entre las semillas, con ranking e importancia acumulada — para
los gráficos y CSVs de la sección de importancia de variables.

Se usa importancia "gain" nativa de XGBoost (no SHAP) porque no requiere
volver a pasar datos de test por el modelo: es intrínseca al booster ya
entrenado, disponible siempre sin dependencias opcionales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_feature_importance(models: list[xgb.Booster]) -> pd.DataFrame:
    """
    Importancia de variables promediada entre los modelos del ensamble.

    Returns: DataFrame con columnas [feature, importance, rank,
    importance_pct, cumulative_pct], ordenado de mayor a menor
    importancia.
    """
    scores_per_model = [m.get_score(importance_type="gain") for m in models]
    all_features: set[str] = set()
    for s in scores_per_model:
        all_features.update(s.keys())

    avg_importance = {
        feat: float(np.mean([s.get(feat, 0.0) for s in scores_per_model]))
        for feat in all_features
    }

    df = pd.DataFrame({
        "feature": list(avg_importance.keys()),
        "importance": list(avg_importance.values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total * 100.0) if total > 0 else 0.0
    df["cumulative_pct"] = df["importance_pct"].cumsum()

    return df


def compute_feature_importance_all(
    models_by_horizon: dict[int, list[xgb.Booster]],
) -> dict[int, pd.DataFrame]:
    """Calcula la importancia de variables para todos los horizontes disponibles."""
    return {h: compute_feature_importance(models) for h, models in models_by_horizon.items()}


def top_feature_per_horizon(fi_by_horizon: dict[int, pd.DataFrame]) -> dict[int, str]:
    """La variable más importante de cada horizonte — usado en el resumen final."""
    result = {}
    for h, fi_df in fi_by_horizon.items():
        if not fi_df.empty:
            result[h] = fi_df.iloc[0]["feature"]
    return result
