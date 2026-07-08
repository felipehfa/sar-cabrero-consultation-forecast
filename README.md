# Predicción de Consultas Respiratorias — SAR Cabrero

Sistema de pronóstico de demanda de consultas respiratorias en el Servicio de
Atención de Urgencia Rural (SAR) de Cabrero, Chile. Predice el número de
consultas diarias a horizontes de **1 a 7 días**, combinando variables
climáticas (temperatura, humedad, precipitación, PM2.5, PM10, NO2, ozono),
calendario/feriados y el historial propio de consultas.

**[Ver dashboard](https://sar-cabrero-consultation-forecast.streamlit.app/)**

---

## Estructura del proyecto

```
consultas-predict-cabrero/
├── data/
│   ├── raw/                  # Datos crudos (clima, contaminación, consultas)
│   └── processed/             # Datasets procesados listos para modelar
├── ml/
│   ├── data/                  # Scripts de obtención de datos (clima, consultas)
│   ├── features/               # Feature engineering y selección de variables
│   ├── models/                 # Entrenamiento del modelo XGBoost (ensamble)
│   └── production/             # Pipeline de inferencia, evaluación y reporte
│       └── results/            # Forecasts, métricas, backtesting y gráficos
├── save/model/                 # Modelos entrenados (ensambles .pkl) por horizonte
└── visulization-streamlit/     # Dashboard interactivo (Streamlit)
```

- **`ml/data/`** — descarga y consolida los datos crudos de clima/contaminación y de consultas del SAR.
- **`ml/features/`** — construye todas las variables predictivas (feature engineering) y selecciona el subconjunto óptimo por horizonte.
- **`ml/models/`** — entrena el modelo XGBoost con búsqueda de hiperparámetros y ensamble de bagging.
- **`ml/production/`** — orquesta todo el flujo de punta a punta: genera el pronóstico, evalúa contra baselines, hace backtesting operacional y produce reportes y gráficos.
- **`visulization-streamlit/`** — dashboard web para explorar pronósticos y resultados de forma interactiva.

---

## Metodología

1. **Datos**: series diarias de variables meteorológicas y de calidad del aire
   (temperatura, humedad, precipitación, PM2.5, PM10, NO2, ozono), calendario
   chileno (día de la semana, feriados, estación del año) y el historial de
   consultas del SAR.
2. **Feature engineering**: se generan rezagos (*lags*), promedios y
   desviaciones móviles, tendencias y variables normalizadas a partir del
   historial de consultas, además de variables derivadas del clima y el
   calendario.
3. **Selección de variables**: por cada horizonte (1 a 7 días) se elige un
   subconjunto de variables mediante deduplicación por familia, control de
   drift (PSI train→val) y diversificación por dominio (consultas, clima,
   contaminación, calendario).
4. **Entrenamiento**: modelo **XGBoost** de regresión (objetivo Poisson o
   Tweedie, elegido automáticamente), con búsqueda de hiperparámetros vía
   **Optuna** y un **ensamble de bagging de 5 semillas** por horizonte para
   mayor estabilidad. Incluye intervalos de predicción (conformal prediction).
5. **Evaluación**: el modelo se compara contra dos baselines *naive*
   (semanal y anual, ajustado a calendario) usando MAE, RMSE, R² y sMAPE.
6. **Backtesting operacional**: se simula el desempeño del pipeline como si
   se hubiera ejecutado en producción cada domingo del último año, para
   validar su comportamiento en condiciones realistas.

---

## Cómo correrlo localmente

### 1. Requisitos

- Python 3.10+ (entorno conda `data-science`)
- Dependencias principales: `pandas`, `numpy`, `xgboost`, `optuna`,
  `scikit-learn`, `holidays`, `matplotlib`, `streamlit`, `plotly`

### 2. Activar el entorno

```bash
conda activate data-science
```

### 3. Ejecutar el pipeline completo

Desde la raíz del proyecto:

```bash
cd consultas-predict-cabrero
python ml/production/run_pipeline.py
```

Esto entrena (o reentrena) los modelos para los 7 horizontes, genera el
pronóstico, evalúa contra los baselines, corre el backtesting y produce
gráficos y un resumen en `ml/production/results/`.

Para iterar más rápido sobre el reporte sin reentrenar (reutilizando
modelos y features ya generados):

```bash
python ml/production/run_pipeline.py --skip-training
```

### 4. Levantar el dashboard

```bash
streamlit run visulization-streamlit/app.py
```

## Resultados destacados

En evaluación sobre el conjunto de test, el modelo XGBoost supera de forma
consistente a ambos baselines *naive* en los 7 horizontes de pronóstico:

| Horizonte | MAE modelo | MAE naive semanal | MAE naive anual | Mejora vs. semanal | Mejora vs. anual |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 día | 7.41 | 8.94 | 10.60 | 17.1% | 30.1% |
| 3 días | 7.39 | 8.92 | 10.62 | 17.2% | 30.5% |
| 5 días | 7.17 | 9.02 | 10.69 | 20.6% | 33.0% |
| 7 días | 7.74 | 8.97 | 10.65 | 13.7% | 27.3% |

En promedio sobre todos los horizontes, el modelo reduce el error (MAE) en
**~16.7%** respecto al baseline semanal y **~29.9%** respecto al baseline
anual. El mejor desempeño se obtiene en el horizonte de 5 días (MAE 7.17,
RMSE 9.20). El backtesting operacional, que simula ejecuciones semanales a
lo largo de un año completo, confirma un MAE promedio de ~7.1 consultas/día,
consistente con la evaluación en test.

*(Cifras extraídas de `ml/production/results/comparison.csv`,
`backtesting_metrics.csv` y `summary.json`.)*

---

## Datos de consultas médicas

Los datos utilizados en este proyecto fueron obtenidos desde el portal de **Datos Abiertos del Departamento de Estadísticas e Información de Salud (DEIS)** del Ministerio de Salud de Chile.

Con el objetivo de mantener un repositorio liviano, **las bases de datos originales no se incluyen en este repositorio**, debido a su tamaño.

Los archivos pueden descargarse desde el siguiente enlace:

**https://deis.minsal.cl/#datosabiertos**

Dentro del portal, seleccione la opción **"Atenciones de Urgencia"**, donde encontrará las bases de datos correspondientes a cada año. La base de datos del año en curso es actualizada diariamente por el DEIS.

Una vez descargados los archivos, estos deben ubicarse en la siguiente carpeta del proyecto:

```text
data/raw/datos_deis/

---

## Tecnologías

- **Python** — lenguaje principal del proyecto
- **XGBoost** — modelo de regresión (gradient boosting)
- **Optuna** — búsqueda automática de hiperparámetros
- **scikit-learn** — validación cruzada temporal y métricas
- **pandas / numpy** — procesamiento de datos y features
- **Streamlit** — dashboard interactivo
- **Plotly / Matplotlib** — visualización de resultados y gráficos del pipeline
- **holidays** — calendario de feriados chilenos