import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 1. CONFIGURACIÓN GLOBAL
st.set_page_config(
    page_title="Pronóstico de Consultas Respiratorias",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS: jerarquía tipográfica + pestañas + tarjetas de métricas.
#
# Los selectores están deliberadamente acotados (nada de "span"/"label" a
# secas): un selector tan amplio termina aplicándose también a los <span>
# internos que Streamlit inserta dentro de los propios encabezados
# (st.title/st.header/st.subheader), pisando su tamaño y dejándolos más
# chicos que el texto de cuerpo. Por eso los títulos se targetean por tag
# (h1..h4) con alta prioridad, y el cuerpo solo por su contenedor
# (stMarkdownContainer p / stWidgetLabel), sin tocar spans genéricos.
st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stWidgetLabel"] p {
        font-size: 1.1rem !important;
    }

    h1 { font-size: 2.6rem !important; font-weight: 700 !important; }
    h2 { font-size: 2.1rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.7rem !important; font-weight: 700 !important; }
    h4 { font-size: 1.4rem !important; font-weight: 600 !important; }

    /* Selector de sección (reemplaza st.tabs, ver nota en el código) */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 1.75rem !important;
    }
    div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] { font-size: 1.9rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 1.05rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Función para determinar la ruta correcta de cualquier archivo en el proyecto
def get_project_path(relative_path):
    # Si existe directamente (ej. ejecutado desde la raíz)
    if os.path.exists(relative_path):
        return relative_path
    # Si hay que subir un nivel (ej. ejecutado desde la subcarpeta visulization-streamlit)
    alternative_path = os.path.join("..", relative_path)
    if os.path.exists(alternative_path):
        return alternative_path
    return relative_path


def metric_card(col, label, value, help=None):
    """Renderiza un st.metric dentro de un contenedor con borde (tarjeta visual)."""
    with col:
        with st.container(border=True):
            st.metric(label, value, help=help)


def get_plotly_template() -> str:
    """
    Plantilla de Plotly acorde al tema activo de Streamlit (☰ → Settings →
    Theme, o [theme] base en .streamlit/config.toml). "plotly_dark"/
    "plotly_white" ya traen colores de texto, grilla y leyenda pensados
    para fondo oscuro/claro respectivamente — evita hardcodear un solo
    color de grilla/texto que se vea bien en un tema y mal en el otro.
    Se llama en el sitio de la llamada (no dentro de una función
    @st.cache_data) para que cambiar el tema invalide el cache y
    reconstruya la figura con la plantilla correcta.
    """
    return "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"


# ──────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_backtesting_predictions():
    """ml/production/results/backtesting_predictions.csv — todos los horizontes mezclados."""
    csv_path = get_project_path("ml/production/results/backtesting_predictions.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["fecha_pronosticada"] = pd.to_datetime(df["fecha_pronosticada"])
    df["fecha_ejecucion"] = pd.to_datetime(df["fecha_ejecucion"])
    return df


@st.cache_data
def load_real_series():
    """
    Serie real completa de total_atenciones, tomada de
    data/processed/feature_engineering.csv (misma fuente que usa
    plot_operational_forecast en ml/production/pipeline/plotting.py).
    Se usa como línea de fondo continua — evita los huecos que aparecen
    al reconstruir la serie solo desde valor_real en los bordes de cada
    ventana de backtesting.
    """
    csv_path = get_project_path("data/processed/feature_engineering.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path, usecols=["fecha", "total_atenciones"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha")


@st.cache_data
def load_backtesting_metrics():
    """ml/production/results/backtesting_metrics.csv — ML vs Naive_semanal vs Naive_anual."""
    csv_path = get_project_path("ml/production/results/backtesting_metrics.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    return pd.read_csv(csv_path)


# ──────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE FIGURAS (cacheadas)
#
# Streamlit re-ejecuta el script completo ante CUALQUIER interacción, en
# CUALQUIER pestaña — eso es normal. El problema aparece cuando una figura
# Plotly pesada (ej. ~50 trazos del abanico semanal) se reconstruye como un
# objeto Python nuevo en cada rerun aunque sus datos de entrada no hayan
# cambiado: st.plotly_chart trata eso como "contenido nuevo" y vuelve a
# montar el componente, incluso en una pestaña oculta — lo que puede
# manifestarse como un parpadeo/"sangrado" visual de esa pestaña antes de
# que el CSS la vuelva a ocultar. Cachear la construcción de la figura hace
# que reruns disparados por OTRA pestaña (ej. el selector de "Resultados")
# reciban el mismo objeto ya cacheado, sin reconstrucción ni remount.
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def build_operational_figure(df_window: pd.DataFrame, df_real: pd.DataFrame,
                              corte: pd.Timestamp, fecha_max: pd.Timestamp,
                              template: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=(
            "Consultas respiratorias: real vs. pronóstico semanal",
            "Error diario (real − predicción)",
        ),
    )

    # [HOVER] hovermode="x unified" agrupa, por subplot, todos los traces que
    # tengan un punto cerca del cursor — pero "cerca" es una búsqueda del
    # punto más cercano EN PÍXELES por cada trace, no una coincidencia
    # exacta de fecha. Con ~50 trazos finos (uno por semana de ejecución),
    # varias semanas distintas quedaban "cerca" del cursor y se mezclaban
    # bajo un mismo encabezado (confirmado: eso NO se arregla cambiando el
    # tipo de eje, el buscador de píxel más cercano es independiente del
    # tipo de eje). La solución real es separar estilo de hover:
    #   - Los ~50 trazos del abanico quedan solo VISUALES: hoverinfo="skip"
    #     los excluye por completo del sistema de hover (no compiten por el
    #     tooltip).
    #   - Un único trace adicional, invisible (opacity 0), con EXACTAMENTE
    #     un punto por fecha de la ventana completa (no hay fechas donde dos
    #     horizontes/semanas se superpongan en los datos), concentra todo el
    #     hover — sin ambigüedad posible entre trazos.
    primer_abanico = True
    for exec_date, group in df_window.groupby("fecha_ejecucion"):
        group = group.sort_values("horizonte")
        fig.add_trace(
            go.Scatter(
                x=group["fecha_pronosticada"], y=group["prediccion"],
                mode="lines+markers",
                line=dict(color="rgba(31,119,180,0.35)", width=1),
                marker=dict(size=4, color="rgba(31,119,180,0.5)"),
                name="Pronóstico semanal (t+1..t+7)",
                legendgroup="fan_pred",
                showlegend=primer_abanico,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=group["fecha_pronosticada"], y=group["error"],
                mode="lines+markers",
                line=dict(color="rgba(214,39,40,0.35)", width=1),
                marker=dict(size=4, color="rgba(214,39,40,0.5)"),
                name="Error diario (real − predicción)",
                legendgroup="fan_err",
                showlegend=primer_abanico,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )
        primer_abanico = False

    # ── Trace único de hover para las predicciones (fila 1) e igual para
    # el error (fila 2) — invisible, un solo punto por fecha, sin ambigüedad.
    df_hover = df_window.sort_values("fecha_pronosticada")
    fig.add_trace(
        go.Scatter(
            x=df_hover["fecha_pronosticada"], y=df_hover["prediccion"],
            mode="markers",
            marker=dict(size=6, opacity=0),
            name="Pronóstico semanal (t+1..t+7)",
            legendgroup="fan_pred",
            showlegend=False,
            customdata=df_hover[["horizonte"]].to_numpy(),
            hovertemplate=(
                # Sin "Fecha:" (ya es el encabezado del cuadro unificado) ni
                # "Valor real:" (ya lo muestra, con su propio color, el trace
                # "Valor real" — repetirlo aquí duplicaba la línea en el
                # cuadro consolidado).
                "<b>Predicción:</b> %{y:.2f}<br>"
                "<b>Horizonte:</b> t+%{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_hover["fecha_pronosticada"], y=df_hover["error"],
            mode="markers",
            marker=dict(size=6, opacity=0),
            name="Error diario (real − predicción)",
            legendgroup="fan_err",
            showlegend=False,
            customdata=df_hover[["horizonte"]].to_numpy(),
            hovertemplate=(
                "<b>Error (real − predicción):</b> %{y:+.1f}<br>"
                "<b>Horizonte:</b> t+%{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    # ── Serie real (fila 1, continua, por encima de los abanicos) ──
    # Preferir la serie completa de total_atenciones (sin huecos en los
    # bordes de cada ventana); si no está disponible, reconstruir desde
    # valor_real del propio backtest (puede tener pequeños huecos).
    if not df_real.empty:
        real_slice = df_real[(df_real["fecha"] >= corte) & (df_real["fecha"] <= fecha_max)]
        real_x, real_y = real_slice["fecha"], real_slice["total_atenciones"]
    else:
        real_slice = (
            df_window[["fecha_pronosticada", "valor_real"]]
            .drop_duplicates()
            .sort_values("fecha_pronosticada")
        )
        real_x, real_y = real_slice["fecha_pronosticada"], real_slice["valor_real"]

    fig.add_trace(
        go.Scatter(
            x=real_x, y=real_y,
            mode="lines",
            line=dict(color="#ff7f0e", width=2.5),
            name="Valor real",
            hovertemplate=(
                "<b>Valor real:</b> %{y:.0f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ── Línea de referencia error=0 (fila 2) ──
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # hovermode="x unified": agrupa en un solo cuadro los traces con hover
    # habilitado que comparten la fecha del cursor (por subplot). Con los
    # abanicos en hoverinfo="skip" y un solo trace de hover por fila, cada
    # cuadro consolidado trae como máximo 2 entradas (real + predicción, o
    # solo error), sin mezclas entre semanas.
    # template="plotly_dark"/"plotly_white" (según el tema activo de
    # Streamlit) le da a texto, grilla y leyenda colores con buen contraste
    # para ese fondo. plot/paper_bgcolor quedan transparentes para que se
    # vea el fondo real de la app (que ya cambia con el tema) detrás del
    # gráfico, en vez de un color de fondo propio de la plantilla.
    fig.update_layout(
        template=template,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=700,
    )
    fig.update_xaxes(showgrid=True, title_text="Fecha", row=2, col=1)
    fig.update_yaxes(showgrid=True, title_text="Consultas respiratorias", row=1, col=1)
    fig.update_yaxes(showgrid=True, title_text="Error", row=2, col=1)

    return fig


@st.cache_data
def build_comparison_figure(metrics_df: pd.DataFrame, metric_sel: str, metric_label: str,
                             template: str) -> go.Figure:
    colors = {"ML": "#1f77b4", "Naive_semanal": "#ff7f0e", "Naive_anual": "#2ca02c"}
    modelo_labels = {"ML": "Machine Learning", "Naive_semanal": "Naive_semanal", "Naive_anual": "Naive_anual"}
    modelos_orden = ["ML", "Naive_semanal", "Naive_anual"]

    fig = go.Figure()
    for modelo in modelos_orden:
        sub = metrics_df[metrics_df["modelo"] == modelo].sort_values("horizonte")
        fig.add_trace(go.Bar(
            x=[f"t+{h}" for h in sub["horizonte"]],
            y=sub[metric_sel],
            name=modelo_labels[modelo],
            marker_color=colors.get(modelo),
        ))

    fig.update_layout(
        template=template,
        barmode="group",
        xaxis_title="Horizonte",
        yaxis_title=metric_label,
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# TÍTULO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────

st.title("🩺 Pronóstico de Consultas Médicas por Enfermedades Respiratorias")
st.markdown("### Desarrollo de modelos predictivos de consultas médicas respiratorias en el SAR Cabrero")
st.write("---")

SECCIONES = ["📈 Backtesting", "📚 Metodología", "📊 Resultados"]

# [TABS BUG] st.tabs() se reemplazó por este selector porque, con datos
# reales, interactuar con un widget dentro de "Resultados" hacía aparecer
# el contenido completo de "Backtesting" renderizado por encima (confirmado
# visualmente, no era solo un parpadeo — persistía incluso cacheando la
# construcción de las figuras). st.tabs() mantiene todas las pestañas
# "montadas" y las oculta vía CSS; ese mecanismo es responsabilidad interna
# del componente, no del código de esta app, y en este caso no encapsulaba
# correctamente el contenido. Con un st.radio + if/elif, en cada rerun se
# ejecuta y se renderiza ÚNICAMENTE el bloque de la sección elegida — el
# código de las otras dos secciones ni siquiera corre, así que no hay
# ningún mecanismo por el cual su contenido pueda aparecer.
seccion = st.radio(
    "Sección",
    SECCIONES,
    horizontal=True,
    label_visibility="collapsed",
    key="seccion_activa",
)

# ──────────────────────────────────────────────────────────────────────────
# 1. BACKTESTING — pronóstico operacional semanal (T+1..T+7) vs Real
# ──────────────────────────────────────────────────────────────────────────

if seccion == "📈 Backtesting":
    st.subheader("📈 Backtesting operacional: valor real vs. predicho")

    st.markdown(
        """
Este proyecto busca **anticipar cuántas personas van a consultar por
enfermedades respiratorias** (resfríos, bronquitis, neumonía, entre otras)
en Cabrero, día a día y con hasta **7 días de anticipación**. Para eso, un
modelo de machine learning (XGBoost) combina el historial de consultas
médicas con datos de clima (temperatura, humedad, lluvia), calidad del aire
(PM2.5, PM10, NO2, ozono) y calendario (día de la semana, feriados,
estación del año).

¿Para qué sirve? Anticipar los picos de demanda respiratoria — típicamente
en invierno — permite planificar personal e insumos con más anticipación
que reaccionando día a día a lo que ya está ocurriendo.

Este dashboard resume el proyecto completo: en **esta pestaña** se muestra
qué tan bien habría funcionado el modelo si se hubiera usado durante el
último año (un *backtesting* honesto — cada domingo el modelo solo conocía
lo que había pasado hasta ese día); en **Metodología** se explica paso a
paso cómo se construye; y en **Resultados** se compara su desempeño contra
métodos de referencia simples (naive).
"""
    )

    df_bt = load_backtesting_predictions()

    if df_bt.empty:
        st.error(
            "No se encontró `ml/production/results/backtesting_predictions.csv`. "
            "Corre `ml/production/run_pipeline.py` para generarlo."
        )
    else:
        st.info(
            "**Cómo leer este gráfico:** la línea naranja (arriba) muestra las "
            "consultas reales día a día. Cada trazo azul representa las 7 "
            "predicciones que el modelo hizo un domingo específico para la semana "
            "siguiente (t+1 a t+7) — mientras más cerca esté un trazo azul de la "
            "línea naranja, mejor fue esa predicción. Abajo, cada trazo rojo muestra "
            "el error de esas mismas predicciones (valor real menos predicción): "
            "cerca de cero es acierto, positivo significa que el modelo subestimó la "
            "demanda real, y negativo que la sobreestimó."
        )

        # Ventana de últimos 365 días, por fecha de ejecución (cada domingo)
        ultima_ejecucion = df_bt["fecha_ejecucion"].max()
        corte = ultima_ejecucion - pd.Timedelta(days=365)
        df_window = df_bt[df_bt["fecha_ejecucion"] >= corte].copy()
        fecha_max = df_window["fecha_pronosticada"].max()

        df_real = load_real_series()
        fig = build_operational_figure(df_window, df_real, corte, fecha_max, get_plotly_template())
        st.plotly_chart(fig, width='stretch')

        col1, col2, col3 = st.columns(3)
        mae_promedio = df_window["error_absoluto"].mean()
        metric_card(col1, "MAE promedio (t+1..t+7)", f"{mae_promedio:.2f}" if pd.notna(mae_promedio) else "s/d")
        metric_card(col2, "Domingos en la ventana", f"{df_window['fecha_ejecucion'].nunique()}")
        metric_card(
            col3,
            "Rango de fechas",
            f"{df_window['fecha_pronosticada'].min().date()} → {df_window['fecha_pronosticada'].max().date()}",
        )

# ──────────────────────────────────────────────────────────────────────────
# 2. METODOLOGÍA
# ──────────────────────────────────────────────────────────────────────────

elif seccion == "📚 Metodología":
    st.subheader("📚 Metodología del proyecto")
    st.markdown(
        "El proyecto predice consultas respiratorias diarias en Cabrero a horizontes "
        "de 1 a 7 días, combinando clima, contaminación, calendario y la propia serie "
        "histórica de consultas. A continuación se detalla cada etapa del pipeline."
    )

    st.code(
        "Open-Meteo (clima + calidad del aire)        DEIS (consultas SAR Cabrero)\n"
        "        │  ml/data/fetch_clima.py                     │  ml/data/fetch_consultas.py\n"
        "        │                                              │  (incluye limpieza de outliers)\n"
        "        ▼                                              ▼\n"
        "data/raw/datos_clima.csv          data/processed/datos_consultas_corregidos.csv\n"
        "        └───────────────────────┬─────────────────────┘\n"
        "                                 ▼\n"
        "        ml/features/feature_engineering.py   (corre 1 vez: features + target_h1..h7)\n"
        "                                 ▼\n"
        "        ml/features/feature_selection.py     (corre 1 vez por horizonte h)\n"
        "                                 ▼\n"
        "        ml/models/model_xgboost.py           (corre 1 vez por horizonte h)\n"
        "                                 ▼\n"
        "        ml/production/run_pipeline.py        (forecast, backtesting, métricas, reporte)",
        language=None,
    )

    with st.expander("1. Origen de los datos", expanded=True):
        st.markdown(
            """
**Clima y contaminación** (`ml/data/fetch_clima.py`): consulta dos APIs de
Open-Meteo para las coordenadas de Cabrero (lat -37.034, lon -72.405):

- El archivo histórico (`archive-api`), con **temperatura, humedad relativa
  y precipitación** horarias.
- La API de calidad del aire (`air-quality-api`), con **PM2.5, PM10, NO2 y
  ozono** horarios.

Ambas fuentes se consolidan por hora en `data/raw/datos_clima.csv`, y luego
`feature_engineering.py` las agrega a nivel diario (ver punto 3).

**Consultas médicas** (`ml/data/fetch_consultas.py`): procesa los CSV crudos
que publica **DEIS** (Departamento de Estadísticas e Información en Salud)
en `data/raw/datos_deis/`:

- Filtra por establecimiento — actualmente **SAR Cabrero** (el script
  soporta una lista de establecimientos, comentada, por si se quiere volver
  a incluir otros centros de la red, como el Hospital Regional de
  Concepción).
- Filtra por causa — **CIE-10 J00-J98, enfermedades del sistema
  respiratorio**.
- Convierte fechas (`dayfirst=True`) y descarta filas con fecha o total
  inválidos.
- Si el mismo día aparece repetido entre archivos o establecimientos, se
  **suman** todas las atenciones de ese día.

El resultado (antes de limpiar outliers) se resume en consola: rango de
fechas cubierto y total de consultas, y luego se corrige (ver punto 2) y se
guarda en `data/processed/datos_consultas_corregidos.csv` — la fuente que
lee `feature_engineering.py` como `INPUT_CONSULTAS`.
"""
        )

    with st.expander("2. Limpieza de outliers"):
        st.markdown(
            """
La limpieza ocurre **dentro de `fetch_consultas.py`**, en la función
`corregir_outliers()`, inmediatamente antes de exportar el CSV — no existe
un script de corrección separado ni un archivo intermedio "sin corregir".

La función recorre la serie diaria y, para cada día (excepto el primero y
el último), reemplaza el valor por el **promedio entre el día anterior y el
siguiente** en dos situaciones:

1. **El valor es 0** — típicamente un día sin registro cargado, no un día
   real sin consultas.
2. **El valor es ≥ 100** (`umbral_alto`) **y** el día anterior fue **< 40**
   (`umbral_anterior`) — un salto abrupto de esa magnitud es mucho más
   compatible con un error de digitación/carga en la fuente DEIS que con un
   pico real de un día para otro.

Cada corrección queda registrada (fecha, valor original, valor nuevo) y se
imprime en consola al correr el script, para trazabilidad — no es una
limpieza silenciosa.

Es una regla **determinística y basada en umbrales fijos**, no un método
estadístico (ej. z-score o IQR): se prefirió así porque es fácil de auditar
y de explicar a alguien no técnico revisando los datos de salud pública.
"""
        )

    with st.expander("3. Ingeniería de variables (`feature_engineering.py`)"):
        st.markdown(
            """
Corre **una sola vez** (no por horizonte): las features de clima, calendario
y atenciones no dependen del horizonte de predicción, solo cambia a qué
columna de target apunta cada horizonte. El script genera de una sola
pasada un único `data/processed/feature_engineering.csv` con las 7 columnas
`target_h1..target_h7`.

**Variables de clima**: agregación horaria → diaria (media, máximo, mínimo
y desviación estándar para temperatura y humedad; suma y máximo para
precipitación; media/máx/mín/std para PM2.5, PM10, NO2 y ozono), conteo de
horas sobre umbrales de contaminación (ej. PM2.5 > 25 y > 50 µg/m³), y
variables derivadas: rango térmico diario, ratio PM2.5/PM10 y un índice de
"contaminación total" (suma de los 4 contaminantes). Además, lags (1 a 7
días) y medias/desviaciones móviles (ventanas de 3, 7 y 14 días) de las
variables climáticas principales, y el cambio día a día (`_delta1`) de
temperatura y los 4 contaminantes.

**Variables de calendario**: día de la semana y mes en forma cíclica
(seno/coseno, para que el modelo entienda que diciembre y enero están
"cerca"), feriados de Chile vía la librería `holidays`, flags de día previo
y posterior a feriado, y dummies de estación del año (hemisferio sur:
verano = dic-feb, ..., primavera = sep-nov).

**Variables de atenciones**: lags de 1 a 28 días (1, 2, 3, 5, 7, 10, 14, 21,
28), medias/desviaciones/máximos/mínimos móviles (ventanas de 3 a 28 días),
tendencias entre ventanas cortas y largas (`atenciones_trend_3_7`,
`_7_14`, `_14_28`), variación semanal (`atenciones_vs_week`) y variación
porcentual (`pct_change_7/14`).

**Normalización anti-drift**: las variables de *nivel absoluto* de
atenciones son, históricamente, las que más drift presentan — el volumen
base de consultas se desplaza con el tiempo por razones ajenas al clima
(cambios poblacionales, de cobertura del centro, etc.). Para mitigarlo,
cada variable de nivel (lags, medias, std, máx, mín, y `total_atenciones`
mismo) tiene una versión **`_norm`**, dividida por una **línea base móvil
de 90 días** (`ATENCIONES_LONG_WINDOW = 90`, un promedio móvil con
`shift(1)` para no filtrar el día actual), con un epsilon (`EPS_BASELINE =
1e-3`) para evitar división por cero en rachas de consultas muy bajas. La
selección de variables (punto 4) es libre de elegir la versión cruda o la
normalizada según cuál sea más estable en cada horizonte — no hay una regla
fija que descarte una u otra por nombre.

**Targets**: `target_h1` a `target_h7` se generan en una sola pasada
(`total_atenciones.shift(-(h+1))` — el "+1" refleja que el pipeline corre
la mañana siguiente a que un día quede completo). El `dropna()` final se
aplica **solo sobre las features**, nunca sobre las columnas target: cada
horizonte conserva su propia cola de NaN al final de la serie, y cada
script downstream descarta esa cola por su cuenta antes de entrenar o
evaluar ese horizonte específico.
"""
        )

    with st.expander("4. Selección de variables (`feature_selection.py`)"):
        st.markdown(
            """
Corre **una vez por horizonte** (7 corridas independientes — las variables
ganadoras de un horizonte no se asumen válidas para otro). Split temporal
por **ventana fija de 365 días** para validación y para test (no por
porcentaje del dataset): así val y test cubren cada uno un ciclo estacional
completo, evitando que el PSI train→val quede inflado por comparar una
mezcla de estaciones (train) contra una ventana parcial.

1. **Eliminación de constantes**: variables con varianza ≈ 0 sobre train se
   descartan de entrada.
2. **Deduplicación intra-familia** (`FAMILY_CORR_THRESHOLD = 0.90`): dentro
   de cada familia (mismo nombre base tras quitar sufijos `_lag*/_ma*/
   _std*/_max*/_min*/_delta*`), las variables muy correlacionadas entre sí
   se reducen a una sola — la de mayor **importancia SHAP** de un modelo
   entrenado *solo* con esa familia. Se prefirió SHAP sobre correlación de
   Pearson simple con el target porque, en la práctica, Pearson elegía mal
   al "campeón" de la familia (por ejemplo, descartaba `atenciones_ma7`
   frente a variantes menos útiles).
3. **Selección por estabilidad**: se entrenan `N_MODELS = 30` modelos con
   `TimeSeriesSplit` (`CV_FOLDS = 5`, con `gap = horizonte` para no filtrar
   información del futuro hacia el pasado) y se mide la importancia SHAP de
   cada variable en cada fold. `stability_score = (1 / (1 + CV)) ×
   presencia`, donde CV es el coeficiente de variación de la importancia
   entre folds y "presencia" es el % de folds donde la variable tuvo
   importancia > 0. Solo pasan las variables con `presence_pct ≥
   PRESENCE_THRESHOLD = 0.6`.
4. **PSI train→val** (`HARD_PSI_LIMIT = 0.5`): el Índice de Estabilidad
   Poblacional mide cuánto cambió la *distribución* de una variable entre
   train y val (drift de los datos de entrada, no del error del modelo).
   Variables con PSI ≥ 0.5 se excluyen por drift extremo; el resto recibe
   un `adjusted_score = stability_score / (1 + PSI)` que penaliza
   (sin excluir) el drift moderado.
5. **Diversificación por dominio**: las candidatas se reordenan para
   garantizar representación mínima de cada uno de los 4 dominios
   (`atenciones`, `clima_meteo`, `clima_contaminacion`, `calendario`) al
   frente de la lista — pero solo si su `adjusted_score` supera un piso
   (`MIN_SCORE_RATIO_FOR_DIVERSITY = 0.5` del mejor score elegible), para
   que la cuota no fuerce candidatas débiles solo por llenar un dominio poco
   poblado.
6. **k óptimo por regla de 1 error estándar (1-SE)**: se evalúa la curva de
   MAE de validación vs. número de variables (k), promediada sobre
   `N_SEEDS_K_CURVE = 5` semillas distintas. El error estándar se calcula
   **en el punto del mínimo** de la curva (no sobre la dispersión de toda
   la curva — un error metodológico corregido en una iteración anterior del
   proyecto), y se elige el k más pequeño cuyo MAE promedio quede dentro de
   ese margen del mínimo — parsimonia: preferir menos variables si el costo
   en error es marginal.
7. **Red de seguridad final**: una última pasada elimina correlaciones
   residuales (> 0.95) entre las variables ya seleccionadas, por si alguna
   redundancia cruzó los límites de familia/dominio.

Cada horizonte exporta su propio `selected_features_h{horizonte}.csv` junto
con un reporte detallado y la curva de k, todos en
`ml/features/feature_selection_outputs/`.
"""
        )

    with st.expander("5. Entrenamiento del modelo (`model_xgboost.py`)"):
        st.markdown(
            """
Corre **una vez por horizonte** (1 a 7 días), usando únicamente las
variables que ese horizonte seleccionó en el paso anterior. Split temporal
por ventana fija: train = resto del historial, val = 365 días, test =
últimos 365 días.

- **Optimización de hiperparámetros (Optuna, 50 trials)**: para cada
  intento, Optuna elige entre dos funciones objetivo de conteo —
  `count:poisson` o `reg:tweedie` (más adecuadas que una pérdida gaussiana
  para una variable que cuenta personas) — y un conjunto de
  hiperparámetros (learning rate, profundidad, hojas máximas, subsample,
  colsample, regularización L1/L2, gamma). El MAE se mide con
  `TimeSeriesSplit` (`gap = horizonte`) calculado **exclusivamente sobre
  train**. Esto es deliberado: en una versión anterior del proyecto, val se
  usaba tanto para elegir hiperparámetros como para early stopping/
  evaluación final, lo que generaba una fuga de información (val dejaba de
  ser una medición independiente). Ahora val se reserva sin tocar hasta
  después de fijar los hiperparámetros.
- **Entrenamiento final — bagging de 5 semillas** (`N_SEEDS_BAGGING = 5`):
  con los hiperparámetros ya fijos, se entrenan 5 modelos XGBoost
  idénticos salvo por la semilla (42, 43, 44, 45, 46), cada uno con su
  propio early stopping sobre val. La predicción final del ensamble es el
  **promedio** de las 5 predicciones individuales (cada una ya recortada a
  ≥ 0 antes de promediar) — esto reduce la varianza asociada a la
  inicialización aleatoria de un solo modelo.
- **Intervalos de predicción (split conformal)**: los residuos absolutos de
  val (nunca usados para entrenar ni para elegir hiperparámetros) calibran
  los cuantiles que definen los intervalos IC80 e IC90, con garantía de
  cobertura marginal sin asumir una distribución particular del error.

El ensamble completo (lista de 5 `xgb.Booster`) se guarda en
`save/model/xgboost_regression_h{horizonte}.pkl`, junto con un JSON de
métricas detallado en el mismo directorio.
"""
        )

    with st.expander("6. Métricas de evaluación"):
        st.markdown(
            """
El pipeline reporta seis métricas complementarias — cada una expone un
ángulo distinto del error, porque ninguna por sí sola cuenta toda la
historia:

- **MAE** (Error Absoluto Medio): el error "típico" en las mismas unidades
  que la variable (consultas/día). Fácil de comunicar a un no-técnico.
- **RMSE** (Raíz del Error Cuadrático Medio): como el MAE, pero penaliza
  más fuerte los errores grandes — útil para detectar si el modelo falla
  feo en algunos días puntuales aunque en promedio (MAE) se vea bien.
- **MAPE** (Error Porcentual Absoluto Medio): el error en términos
  relativos (%), pero se distorsiona — puede irse a valores absurdamente
  altos — cuando el valor real está cerca de 0 (día con muy pocas
  consultas), porque se divide por ese valor real.
- **sMAPE** (MAPE simétrico): versión acotada entre 0% y 200% del error
  porcentual, con un denominador que no colapsa cuando el valor real es
  bajo — más confiable que el MAPE para comparar horizontes entre sí.
- **R²**: qué proporción de la variabilidad de las consultas reales explica
  el modelo, respecto a simplemente predecir el promedio histórico. Un R²
  negativo significa que el modelo (o el baseline, en la pestaña
  "Resultados") es peor que ese promedio.
- **Bias**: el error medio **con signo** (no absoluto) — indica si el
  modelo tiende a sub-predecir (bias positivo, real > predicción en
  promedio) o sobre-predecir (bias negativo) de forma sistemática, algo que
  el MAE por sí solo no distingue.

Todas se calculan filtrando pares con NaN antes de comparar (ej. el naive
anual no tiene dato disponible 365 días atrás al inicio de la serie
histórica, lo cual es esperable y no un error).
"""
        )

# ──────────────────────────────────────────────────────────────────────────
# 3. RESULTADOS
# ──────────────────────────────────────────────────────────────────────────

elif seccion == "📊 Resultados":
    st.subheader("📊 Resultados: Machine Learning vs. baselines naive")

    st.markdown(
        """
Para saber si el modelo realmente aporta valor, no basta con mirar su error
de forma aislada — hay que compararlo contra métodos simples que no usan
machine learning, llamados baselines "naive" (ingenuos).

El **naive semanal** predice que un día futuro tendrá el mismo número de
consultas que el mismo día de la semana anterior (por ejemplo, para el
próximo martes usa el valor del martes pasado), asumiendo que el patrón se
repite semana a semana. El **naive anual** predice el valor observado el
mismo día un año atrás, capturando la estacionalidad básica (que junio se
parezca a junio del año pasado, por ejemplo).

La métrica central para comparar es el **MAE** (Error Absoluto Medio): el
promedio de cuánto se equivoca cada predicción, medido en las mismas
unidades que el dato original (consultas respiratorias por día) — mientras
más bajo, mejor.

Pero un MAE bajo por sí solo no dice mucho. Podría deberse simplemente a que
la serie es fácil de predecir — al punto que hasta un método ingenuo como el
naive semanal lograría un error parecido sin aprender nada. Lo que realmente
demuestra que el modelo aporta valor es que su MAE sea consistentemente más
bajo que el de ambos naive, y por un margen relevante: esa diferencia — la
mejora porcentual que se muestra a continuación — es la evidencia de que el
modelo aprende patrones útiles, más allá de simplemente repetir el pasado.
"""
    )

    metrics_df = load_backtesting_metrics()

    if metrics_df.empty:
        st.error(
            "No se encontró `ml/production/results/backtesting_metrics.csv`. "
            "Corre `ml/production/run_pipeline.py` para generarlo."
        )
    else:
        metric_labels = {
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE (%)",
            "smape": "sMAPE (%)",
            "r2": "R²",
            "bias": "Bias",
        }
        metric_sel = st.selectbox(
            "Métrica a comparar por horizonte:",
            list(metric_labels.keys()),
            format_func=lambda m: metric_labels[m],
            index=0,
        )

        fig_cmp = build_comparison_figure(metrics_df, metric_sel, metric_labels[metric_sel], get_plotly_template())
        st.plotly_chart(fig_cmp, width='stretch')

        if metric_sel == "mape":
            st.caption(
                "⚠️ El MAPE puede distorsionarse cuando `valor_real` está cerca de 0 "
                "(ver horizonte t+7, donde unos pocos días con consultas muy bajas disparan "
                "el promedio). El sMAPE, acotado entre 0 y 200%, es más confiable para "
                "comparar entre horizontes."
            )

        st.markdown("#### Tabla completa de métricas")
        st.caption("🏆 marca, para cada horizonte, cuál de los tres modelos gana en esa métrica.")

        MODELO_LABELS = {"ML": "Machine Learning", "Naive_semanal": "Naive_semanal", "Naive_anual": "Naive_anual"}
        # Regla de comparación por métrica: no todas compiten igual —
        # MAE/RMSE/MAPE/sMAPE ganan con el valor más bajo, R² con el más
        # alto, y Bias con el más cercano a 0 en magnitud absoluta.
        REGLA_GANADOR = {
            "mae": "min", "rmse": "min", "mape": "min", "smape": "min",
            "r2": "max", "bias": "abs_min",
        }

        tabla = metrics_df.sort_values(["horizonte", "modelo"]).reset_index(drop=True)

        ganador_por_columna = {}
        for columna, regla in REGLA_GANADOR.items():
            indices_ganadores = set()
            for _, grupo in tabla.groupby("horizonte"):
                valores = grupo[columna]
                if regla == "min":
                    indices_ganadores.add(valores.idxmin())
                elif regla == "max":
                    indices_ganadores.add(valores.idxmax())
                else:  # abs_min
                    indices_ganadores.add(valores.abs().idxmin())
            ganador_por_columna[columna] = indices_ganadores

        tabla_display = tabla.copy()
        tabla_display["modelo"] = tabla_display["modelo"].map(MODELO_LABELS)
        for columna in REGLA_GANADOR:
            tabla_display[columna] = [
                f"{tabla.loc[i, columna]:.2f} 🏆" if i in ganador_por_columna[columna]
                else f"{tabla.loc[i, columna]:.2f}"
                for i in tabla.index
            ]

        def resaltar_ganadores(df):
            estilos = pd.DataFrame("", index=df.index, columns=df.columns)
            for columna, indices in ganador_por_columna.items():
                for i in indices:
                    estilos.loc[i, columna] = "background-color: rgba(46, 204, 113, 0.35); font-weight: 700;"
            return estilos

        st.dataframe(
            tabla_display.style.apply(resaltar_ganadores, axis=None),
            width='stretch',
        )

        st.write("---")
        st.markdown("#### ¿Por qué el modelo Machine Learning es mejor que los naive?")

        pivot_mae = metrics_df.pivot(index="horizonte", columns="modelo", values="mae")
        pivot_mae["mejora_vs_semanal_%"] = (
            (pivot_mae["Naive_semanal"] - pivot_mae["ML"]) / pivot_mae["Naive_semanal"] * 100
        )
        pivot_mae["mejora_vs_anual_%"] = (
            (pivot_mae["Naive_anual"] - pivot_mae["ML"]) / pivot_mae["Naive_anual"] * 100
        )

        r2_pivot = metrics_df.pivot(index="horizonte", columns="modelo", values="r2")
        horizontes_negativos = r2_pivot.index[r2_pivot["Naive_anual"] < 0].tolist()
        horizontes_txt = ", ".join(f"t+{h}" for h in horizontes_negativos) if horizontes_negativos else "ninguno"

        col1, col2, col3 = st.columns(3)
        metric_card(
            col1, "Mejora MAE vs. Naive semanal",
            f"{pivot_mae['mejora_vs_semanal_%'].min():.1f}% – {pivot_mae['mejora_vs_semanal_%'].max():.1f}%",
        )
        metric_card(
            col2, "Mejora MAE vs. Naive anual",
            f"{pivot_mae['mejora_vs_anual_%'].min():.1f}% – {pivot_mae['mejora_vs_anual_%'].max():.1f}%",
        )
        metric_card(
            col3, "Horizontes con R² negativo (Naive anual)",
            f"{len(horizontes_negativos)} de {r2_pivot.shape[0]}",
            help=f"Horizontes afectados: {horizontes_txt}",
        )

        st.dataframe(
            pivot_mae.round(2).rename_axis("horizonte").reset_index().rename(columns={"ML": "Machine Learning"}),
            width='stretch',
        )

        st.markdown(
            f"""
- El modelo **Machine Learning** mejora el MAE frente al **naive semanal**
  entre **{pivot_mae['mejora_vs_semanal_%'].min():.1f}%** y
  **{pivot_mae['mejora_vs_semanal_%'].max():.1f}%** según el horizonte, y
  frente al **naive anual** entre
  **{pivot_mae['mejora_vs_anual_%'].min():.1f}%** y
  **{pivot_mae['mejora_vs_anual_%'].max():.1f}%**.
- El **naive anual** (repetir el valor de hace 365 días, ajustado por
  calendario) tiene **R² negativo** en los horizontes: **{horizontes_txt}**.
  Un R² negativo significa que, en esos horizontes, predecir simplemente el
  promedio histórico habría sido mejor que ese baseline — lo que lo
  convierte en un punto de comparación particularmente débil frente al cual
  la ventaja del modelo Machine Learning es aún más significativa.
- El **naive semanal** (repetir el valor de hace 7 días) es un baseline más
  competitivo (R² positivo en todos los horizontes), pero el modelo Machine
  Learning lo supera en MAE de forma consistente en los 7 horizontes.
"""
        )
