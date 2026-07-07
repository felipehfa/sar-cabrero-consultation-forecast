# CLAUDE.md

## Preferencias de comunicación

- Responde siempre en español.
- Sé directo y conciso — evita explicaciones largas salvo que se pidan explícitamente.
- No repitas código completo si solo cambia una línea o función; muestra solo el fragmento relevante con contexto mínimo.
- Evita resúmenes largos al final de cada respuesta salvo que se pidan.

Contexto del proyecto para Claude Code. Léelo antes de tocar código en este repo.

## Qué hace este proyecto

Predice consultas respiratorias diarias en un pueblo de Chile (Cabrero), a horizontes
de 1 a 7 días, usando clima (temperatura, humedad, precipitación, PM2.5, PM10, NO2,
ozono), calendario/feriados, y la propia serie histórica de consultas. Modelo: XGBoost
(regresión, objetivo `count:poisson` o `reg:tweedie` elegido por Optuna), con un
ensamble de bagging de 5 semillas por horizonte.

Entorno: conda env `data-science` (Windows, PowerShell). Activar con:
```
conda activate data-science
```

## Estructura del proyecto

```
consultas-predict-cabrero/
├── .vscode/settings.json          # python.analysis.extraPaths (ver sección Pylance)
├── data/
│   ├── raw/                       # datos_clima.csv y fuentes crudas
│   └── processed/                 # feature_engineering.csv, datos_consultas_corregidos.csv
├── ml/
│   ├── data/
│   │   ├── fetch_clima.py
│   │   └── fetch_consultas.py
│   ├── features/
│   │   ├── feature_engineering.py     # genera data/processed/feature_engineering.csv
│   │   ├── feature_selection.py       # selecciona variables por horizonte
│   │   └── feature_selection_outputs/
│   │       ├── selected_features_h{h}.csv
│   │       ├── feature_selection_report_h{h}.csv
│   │       ├── metric_curve_k_features_h{h}.csv
│   │       └── family_redundancy_dropped_h{h}.csv
│   ├── models/
│   │   └── model_xgboost.py       # entrena y guarda el ensamble por horizonte
│   └── production/
│       ├── run_pipeline.py        # ORQUESTADOR — único entrypoint de esta etapa
│       ├── pipeline/              # config, validation, forecasting, evaluation,
│       │                          # backtesting, naive_models, metrics,
│       │                          # feature_importance, plotting, reporting,
│       │                          # io_utils, logging_utils
│       └── results/
│           ├── forecast.csv, metrics.csv, comparison.csv
│           ├── backtesting_predictions.csv, backtesting_metrics.csv
│           ├── summary.csv, summary.json   # resumen automático (etapa 8/8)
│           ├── pipeline.log
│           └── plots/                      # horizon_{h}/, feature_importance/, etc.
├── save/model/
│   ├── xgboost_regression_h{h}.pkl          # lista de 5 xgb.Booster (ensamble)
│   └── xgboost_regression_h{h}_metrics.json
└── visulization-streamlit/
    └── app.py                      # dashboard Streamlit — DESACTUALIZADO, ver nota abajo
```

## Orden de ejecución

1. `ml/features/feature_engineering.py` → genera `data/processed/feature_engineering.csv`
   con TODAS las features (clima, calendario, atenciones) + columnas `target_h1`..`target_h7`.
   Corre **una sola vez**, no una vez por horizonte (las features no dependen del horizonte,
   solo el target).
2. `ml/features/feature_selection.py` → corre **una vez por horizonte** (`run_feature_selection(df, horizon=h)`),
   genera `selected_features_h{h}.csv` en `feature_selection_outputs/`.
3. `ml/models/model_xgboost.py` → corre **una vez por horizonte** (`run_all(df, horizon=h)`),
   entrena el ensamble de 5 modelos y guarda `.pkl` + `_metrics.json`.
4. `ml/production/run_pipeline.py` → orquesta 1-3 para los 7 horizontes (etapas 1-2/8),
   y además genera forecast (3/8), evaluación + comparación vs naive (4/8), backtesting
   operacional (5/8), gráficos (6/8), importancia de variables (7/8) y un resumen
   automático en `results/summary.csv` / `summary.json` (8/8). Ver `--skip-training`
   para omitir el reentrenamiento (etapas 1-2 reusan modelos/features ya guardados).

**Ejecutar siempre desde la raíz del proyecto** (`cd consultas-predict-cabrero` antes de
correr cualquier script). `feature_engineering.py`, `feature_selection.py` y
`model_xgboost.py` usan rutas **relativas al directorio de trabajo**, no a su propia
ubicación — si corres desde otro cwd (ej. botón "Run" de algún IDE con cwd distinto),
van a leer/escribir en el lugar equivocado sin avisar. `run_pipeline.py` sí valida esto
y loguea un warning si el cwd no coincide con la raíz del proyecto.

## Convención de horizonte y target

```python
target_h{h} = total_atenciones.shift(-(h + 1))
```

El "+1" existe porque el pipeline corre la mañana siguiente a que un día quede completo:
si la última fecha con datos completos es `D`, se ejecuta en `D+1`, y el horizonte `h`
corresponde a la fecha `D + 1 + h`. Ver `config.forecast_date_for()` en
`ml/production/pipeline/config.py`.

`feature_engineering.py` genera las 7 columnas `target_h1..target_h7` en una sola pasada.
El `dropna()` final se aplica **solo sobre las features**, nunca sobre las columnas
target — cada una conserva su propia cola de NaN (a mayor horizonte, más filas NaN al
final de la serie). Cada script downstream debe hacer su propio
`df.dropna(subset=[f"target_h{h}"])` antes de entrenar/evaluar ese horizonte específico.

## Variables de atenciones — nivel absoluto vs normalizado

`feature_engineering.py` exporta, para `atenciones`, tanto la versión cruda como la
normalizada de cada lag/ma/std/max/min: la cruda (`atenciones_lag{n}`, `atenciones_ma{w}`,
`atenciones_std{w}`, `atenciones_max{w}`, `atenciones_min{w}`) y su par `_norm`
(dividida por una línea base móvil de 90 días, `ATENCIONES_LONG_WINDOW`). También
existen formas relativas: `atenciones_trend_3_7/7_14/14_28`, `atenciones_vs_week`,
`atenciones_pct_change_7/14`. La línea base (`_baseline`, interna) **nunca se exporta
como columna**.

Motivo de tener ambas versiones: las variables de nivel absoluto de `atenciones` son
históricamente las de peor drift (el nivel base de consultas se desplaza con el
tiempo), así que las `_norm` existen como alternativa más estable para que
`feature_selection.py` las prefiera cuando corresponda. **No hay una lista fija de
"solo N excepciones" ni descarte por nombre de columna** — `feature_selection.py` no
tiene ninguna regla hardcodeada para esto; el filtro es el mecanismo genérico de PSI
(`HARD_PSI_LIMIT=0.5` + `adjusted_score`, ver abajo) más la dedup intra-familia.
Revisando `feature_selection_outputs/` actual, el PSI de varias crudas (lags, `ma3`,
`std3/7/10/14`) es hoy bajo (~0.04–0.33, lejos de 0.5), así que sobreviven junto con
`total_atenciones` en varios horizontes — no solo 2-3 columnas. Si el PSI de alguna
vuelve a dispararse, el mecanismo ya existente la penalizaría/excluiría sin cambios de
código.

## Pipeline de selección de variables (`feature_selection.py`)

- Split por ventana fija de 365 días (`VAL_WINDOW_DAYS`, `TEST_WINDOW_DAYS`), no por
  porcentaje — evita que val/test queden desalineados estacionalmente.
- Deduplicación intra-familia (`FAMILY_CORR_THRESHOLD=0.90`): dentro de cada familia
  (mismo nombre base tras quitar sufijos `_lag*/_ma*/_std*/_max*/_min*/_delta*`), se
  descartan variables muy correlacionadas, quedándose con la de mayor importancia SHAP
  de un modelo entrenado **solo con esa familia** (no correlación de Pearson simple —
  Pearson elegía mal el "campeón" de la familia en la práctica).
- PSI train→val (`HARD_PSI_LIMIT=0.5`): excluye candidatas con drift extremo;
  `adjusted_score = stability_score / (1 + PSI)` penaliza el resto.
- Diversificación por dominio (`atenciones`, `clima_meteo`, `clima_contaminacion`,
  `calendario`) con piso de score (`MIN_SCORE_RATIO_FOR_DIVERSITY=0.5`) para que la
  cuota de diversidad no fuerce candidatas débiles solo por llenar un dominio poco
  poblado.
- Selección de `k` óptimo por regla de 1 error estándar (1-SE), promediada sobre
  `N_SEEDS_K_CURVE=5` semillas — el SE se calcula en el punto del mínimo, no sobre la
  dispersión de toda la curva (error metodológico corregido en una iteración anterior).

## Pipeline de entrenamiento (`model_xgboost.py`)

- CV de Optuna corre **solo sobre train** (no train+val) — antes había fuga porque val
  se usaba tanto para elegir hiperparámetros como para early stopping/evaluación.
- Bagging: `N_SEEDS_BAGGING=5` modelos con mismos hiperparámetros, semillas distintas;
  predicción final = promedio del ensamble (`predict_ensemble`).
- `HORIZON` es parámetro de `run_all(df, horizon=1)`, default=1. `FEATURES_DIR` debe
  apuntar a `ml/features/feature_selection_outputs/` (bug ya corregido — antes apuntaba
  a `ml/features/` directo y causaba `FileNotFoundError`).
- Conformal prediction (split conformal) para intervalos IC80/IC90, calibrado con
  residuos de val.

## `ml/production/` — pipeline de inferencia, evaluación y reporte

Consume (sin modificar) los tres módulos de arriba. Ver docstring de
`ml/production/pipeline/config.py` para el detalle de cada ruta.

- `PROJECT_ROOT` se calcula con `Path(__file__).resolve().parents[3]` — relativo a la
  ubicación de `config.py`, no al cwd. Por eso `run_pipeline.py` es robusto al
  directorio desde el que se invoque (aunque los módulos que consume, no — ver más
  arriba).
- `config.py` agrega `ml/features/` y `ml/models/` a `sys.path` automáticamente al
  importarse, para que `import model_xgboost` / `import feature_engineering` /
  `import feature_selection` funcionen sin `__init__.py` en esas carpetas.
- **`run_feature_engineering()` devuelve `fecha` como columna normal, no como índice**
  (consistente con cómo exporta el CSV, `index=False`). `io_utils.normalize_date_index()`
  lo corrige en memoria cada vez que se llama a esa función — sin este fix,
  `df.index.max()` devuelve un entero y todo lo que reste un `Timedelta` revienta.
- Naive baselines: semanal (`t-7`) y anual (`t-365`, calendario-aware — 28/29 de
  febrero de un año bisiesto ambos mapean al "28 de febrero" del año anterior cuando
  ese año no es bisiesto; esto genera duplicados legítimos que `naive_models.py` maneja
  explícitamente).
- Backtesting operacional: simula cada domingo del último año. Por default trunca el
  dataset ya generado en cada fecha de corte (matemáticamente equivalente a re-ejecutar
  feature engineering desde cero, porque todas las features son causales/backward-looking) —
  mucho más rápido. `config.STRICT_RERUN_FEATURE_ENGINEERING=True` fuerza la
  re-ejecución literal si se necesita verificar la equivalencia.
- Métricas: MAE, RMSE, MAPE, sMAPE, R², Bias (`pipeline/metrics.py`). `compute_all_metrics`
  filtra pares NaN antes de calcular (pasa legítimamente cuando el naive anual no tiene
  dato 365 días atrás, ej. al inicio de la serie).

## Streamlit (`visulization-streamlit/app.py`) — DESACTUALIZADO

Lee `save/evaluation/xgboost_predictions_total_conformal.csv`, ruta que **no existe**
en el proyecto actual (no hay carpeta `save/evaluation/`; el pipeline vigente escribe
en `ml/production/results/`) y espera columnas (`ic80_upper/lower`, `ic90_upper/lower`)
que no coinciden con las de `forecast.csv` (`horizonte, fecha_pronosticada, prediccion,
modelo_utilizado`). Este dashboard quedó desconectado de `ml/production/run_pipeline.py`
y no funciona corriéndolo tal cual — antes de tocarlo hay que decidir si se reconecta
a los CSV actuales o se reemplaza.

## Pylance / VSCode

`import model_xgboost`, `import feature_engineering`, `import feature_selection` se
resuelven en tiempo de ejecución (vía el `sys.path` que agrega `config.py`), pero
Pylance los marca en amarillo porque analiza el código sin ejecutarlo. Ya está resuelto
en `.vscode/settings.json`:

```jsonc
{
    "python.analysis.extraPaths": [
        "ml/features",
        "ml/models"
    ]
}
```

## Historial de bugs ya corregidos (no reintroducir)

- **Fuga train→val en Optuna**: el CV de hiperparámetros debe correr solo sobre train.
- **Regla 1-SE mal calculada**: usar SE en el punto del mínimo, no std de toda la curva.
- **Split por porcentaje** desalineaba val/test estacionalmente → se cambió a ventana
  fija de 365 días.
- **Criterio de campeón de familia por Pearson** elegía mal (perdía `atenciones_ma7`
  frente a variantes menos útiles) → se cambió a SHAP de modelo por familia.
- **`total_atenciones` clasificada en dominio "otros"** por error de substring →
  se cuela por la cuota de diversificación sin competir por mérito. Corregido en
  `get_domain()`.
- **Colisión Feb 28/29** en `naive_annual_predict` + `pd.Index.union()` no deduplica si
  el operando ya trae duplicados → corregido en `naive_models.py`.
- **Desalineación en gráfico de importancia acumulada**: barras y línea usaban dos
  ordenamientos distintos → corregido en `plotting.py` (usar el mismo slice/orden para
  ambos).
- **`fecha` como columna, no índice**, al volver de `run_feature_engineering()` →
  corregido con `io_utils.normalize_date_index()`.
- **`FEATURES_DIR` en `model_xgboost.py`** apuntaba a `ml/features/` en vez de
  `ml/features/feature_selection_outputs/` → corregido.

## Qué NO hacer

- No modificar `feature_engineering.py`, `feature_selection.py` ni `model_xgboost.py`
  desde `ml/production/` — ese paquete solo los consume (importa y llama funciones).
  Si hace falta un cambio de comportamiento en esos tres, se edita ahí directamente,
  no se parchea desde afuera.
- No asumir que las variables ganadoras de un horizonte sirven para otro — la
  selección de variables corre independiente por cada `h`.
- No agregar más variables de nivel absoluto de `atenciones` sin verificar PSI
  train→test primero — es el patrón de drift más recurrente detectado en este proyecto.