import requests
import pandas as pd

# =====================================================================
# CONFIGURACIÓN: Cabrero, Chile
# =====================================================================
LATITUD = -37.03394
LONGITUD = -72.40468

# Período de datos reales a extraer (Año 2025 completo)
FECHA_INICIO = "2022-08-04"
FECHA_FIN = pd.Timestamp.now().strftime("%Y-%m-%d")

def obtener_clima_real(fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Consume la API de archivo histórico de Open-Meteo para obtener el
    clima real horario que hubo en Cabrero durante el período seleccionado.
    """
    print(f"Consumiendo datos reales de clima para Cabrero ({fecha_inicio} a {fecha_fin})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "America/Santiago"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        hourly_data = response.json()["hourly"]
        
        df_clima = pd.DataFrame({
            "fecha_hora": pd.to_datetime(hourly_data["time"]),
            "temperatura": hourly_data["temperature_2m"],
            "humedad": hourly_data["relative_humidity_2m"],
            "precipitacion": hourly_data["precipitation"]
        })
        print(f"-> ¡Éxito! Datos de clima descargados. Total de registros horarios: {len(df_clima)}")
        return df_clima
    except Exception as e:
        print(f"Error al conectar con la API de clima de Open-Meteo: {e}")
        return pd.DataFrame()

def obtener_contaminacion_meteo(fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Consume la API de Calidad del Aire de Open-Meteo para Cabrero.
    Trae las variables clave para modelos de salud y series de tiempo.
    """
    print(f"Conectando a Open-Meteo Air Quality para Cabrero...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Parámetros solicitando contaminantes críticos medidos por hora
    params = {
        "latitude": LATITUD,
        "longitude": LONGITUD,
        "start_date": fecha_inicio,
        "end_date": fecha_fin,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "America/Santiago"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        hourly_data = response.json()["hourly"]
        
        # Construimos el DataFrame estructurado
        df_aire = pd.DataFrame({
            "fecha_hora": pd.to_datetime(hourly_data["time"]),
            "pm2_5": hourly_data["pm2_5"],             # Material Particulado Fino (Calefacción/Combustión)
            "pm10": hourly_data["pm10"],               # Material Particulado Grueso (Polvo/Ceniza)
            "no2": hourly_data["nitrogen_dioxide"],    # Dióxido de Nitrógeno (Tránsito vehicular)
            "ozono": hourly_data["ozone"]              # Ozono Troposférico (Contaminante secundario)
        })
        
        print(f"-> ¡Éxito! Datos de contaminación obtenidos correctamente.")
        print(f"Total de registros horarios extraídos: {len(df_aire)}")
        return df_aire
        
    except Exception as e:
        print(f"Error al conectar con la API de contaminación de Open-Meteo: {e}")
        return pd.DataFrame()

def consolidar_datos_horarios():
    """
    Función principal que obtiene los datos climáticos y de contaminación,
    y los une en un único DataFrame consolidado por hora.
    """
    print("Iniciando proceso de consolidación de datos...")
    
    # Obtener datos climáticos
    df_clima = obtener_clima_real(FECHA_INICIO, FECHA_FIN)
    
    # Obtener datos de contaminación
    df_contaminacion = obtener_contaminacion_meteo(FECHA_INICIO, FECHA_FIN)
    
    # Verificar si ambos DataFrames tienen datos
    if df_clima.empty or df_contaminacion.empty:
        print("Error: Uno o ambos DataFrames están vacíos. No se puede consolidar.")
        return pd.DataFrame()
    
    # Unir los DataFrames en uno solo basado en la columna 'fecha_hora'
    # Usamos un join tipo 'outer' para mantener todas las horas disponibles
    df_consolidado = pd.merge(df_clima, df_contaminacion, on='fecha_hora', how='outer')
    
    # Ordenar por fecha y hora
    df_consolidado.sort_values(by='fecha_hora', inplace=True)
    df_consolidado.index.name = None  # Quita el nombre del índice
    
    print(f"-> Consolidación completada. Total de registros consolidados: {len(df_consolidado)}")
    print(f"Columnas en el dataset consolidado: {list(df_consolidado.columns)}")
    
    return df_consolidado

# =====================================================================
# EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    df_consolidado = consolidar_datos_horarios()
    
    if not df_consolidado.empty:
        print("\n=== METADATOS DEL DATASET CONSOLIDADO ===")
        print(df_consolidado.info())
        print("\nMuestra de las primeras horas del dataset consolidado:")
        print(df_consolidado.head())
        
        # Opcional: Guardamos el archivo consolidado para dejarlo disponible en el proyecto
        df_consolidado.to_csv("data/raw/datos_clima.csv", index=False)
        print("\nArchivo 'datos_clima.csv' guardado localmente.")
    else:
        print("\nNo se pudo generar el dataset consolidado.")