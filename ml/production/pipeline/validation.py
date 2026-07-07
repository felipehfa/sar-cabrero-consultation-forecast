"""
pipeline/validation.py

Validaciones que corren antes de cualquier predicción, para detectar
problemas de datos temprano y con mensajes claros, en vez de que XGBoost
falle con un error críptico o (peor) prediga silenciosamente con datos
corruptos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_utils import get_logger


@dataclass
class ValidationResult:
    horizon: int
    ok: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    order_mismatch: bool = False
    nan_columns: list[str] = field(default_factory=list)
    inf_columns: list[str] = field(default_factory=list)
    dtype_issues: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return f"h={self.horizon}: OK"
        parts = []
        if self.missing_columns:
            parts.append(f"faltantes={self.missing_columns}")
        if self.extra_columns:
            parts.append(f"adicionales={self.extra_columns}")
        if self.order_mismatch:
            parts.append("orden de columnas distinto al esperado")
        if self.nan_columns:
            parts.append(f"NaN en={self.nan_columns}")
        if self.inf_columns:
            parts.append(f"inf en={self.inf_columns}")
        if self.dtype_issues:
            parts.append(f"dtype inválido={self.dtype_issues}")
        return f"h={self.horizon}: PROBLEMAS -> " + "; ".join(parts)


def validate_features(
    X: pd.DataFrame,
    expected_features: list[str],
    horizon: int,
) -> ValidationResult:
    """
    Valida un DataFrame de features antes de predecir con el modelo de un
    horizonte dado.

    Chequeos:
        - columnas faltantes respecto a expected_features
        - columnas adicionales no usadas por el modelo (informativo, no
          bloqueante — XGBoost las ignoraría, pero puede indicar un
          desalineamiento con el feature engineering)
        - orden de columnas (relevante solo se compara si no hay faltantes
          ni adicionales, para que el mensaje sea interpretable)
        - valores NaN en columnas esperadas
        - valores infinitos
        - columnas no numéricas (dtype inválido para XGBoost)
    """
    logger = get_logger()

    actual_cols = list(X.columns)
    expected_set = set(expected_features)
    actual_set = set(actual_cols)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    order_mismatch = False
    if not missing and not extra:
        order_mismatch = actual_cols != list(expected_features)

    nan_cols = []
    inf_cols = []
    dtype_issues = {}

    present_expected = [c for c in expected_features if c in X.columns]
    if present_expected:
        sub = X[present_expected]

        nan_mask = sub.isna().any(axis=0)
        nan_cols = nan_mask[nan_mask].index.tolist()

        numeric_cols = sub.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            inf_mask = np.isinf(sub[numeric_cols].to_numpy()).any(axis=0)
            inf_cols = [c for c, is_inf in zip(numeric_cols, inf_mask) if is_inf]

        non_numeric = sub.select_dtypes(exclude=[np.number]).columns
        for c in non_numeric:
            dtype_issues[c] = str(sub[c].dtype)

    ok = not (missing or order_mismatch or nan_cols or inf_cols or dtype_issues)
    # `extra` es informativo, no invalida el resultado por sí solo.

    result = ValidationResult(
        horizon=horizon,
        ok=ok,
        missing_columns=missing,
        extra_columns=extra,
        order_mismatch=order_mismatch,
        nan_columns=nan_cols,
        inf_columns=inf_cols,
        dtype_issues=dtype_issues,
    )

    if not ok:
        logger.error(result.summary())
    elif extra:
        logger.info(f"h={horizon}: columnas adicionales presentes (ignoradas "
                     f"por el modelo): {extra}")

    return result
