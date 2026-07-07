import pandas as pd
import os

# Directorio donde se encuentran los archivos CSV
directorio = "data/raw/datos_deis"

# Obtener lista de archivos CSV
archivos_csv = [f for f in os.listdir(directorio) if f.endswith(".csv")]

# Lista para almacenar los DataFrames procesados
dfs_procesados = []

# Establecimientos de interés
establecimientos = [
    #'SAR Tucapel',
    #'Hospital Clínico Regional Dr. Guillermo Grant Benavente (Concepción)',
    #'SAPU Lorenzo Arenas',
    #'SAPU Juan Soto Fernández',
    #'SAPU Santa Sabina',
    #'SAR Víctor Manuel Fernández',
    #"SAPU Cesfam O'Higgins"
    "SAR Cabrero"
]

# Causas respiratorias
causas = [
    "TOTAL CAUSA SISTEMA  RESPIRATORIO (J00-J98)",
    "TOTAL CAUSAS SISTEMA RESPIRATORIO"
]


def corregir_outliers(
    df,
    columna_fecha="fecha",
    columna_valor="total_atenciones",
    umbral_alto=100,
    umbral_anterior=40
):
    """
    Corrige outliers en una serie de tiempo.

    Reglas:
    1. Si el valor es 0, se reemplaza por el promedio entre el valor anterior y el siguiente.
    2. Si el valor >= umbral_alto y el valor anterior < umbral_anterior,
       se reemplaza por el promedio entre el valor anterior y el siguiente.

    Parámetros
    ----------
    df : pandas.DataFrame
    columna_fecha : str
    columna_valor : str
    umbral_alto : int
    umbral_anterior : int

    Retorna
    -------
    DataFrame con los valores corregidos.
    """

    df_corregido = df.copy().reset_index(drop=True)

    modificaciones = []

    for i in range(1, len(df_corregido) - 1):

        valor = df_corregido.loc[i, columna_valor]
        anterior = df_corregido.loc[i - 1, columna_valor]
        siguiente = df_corregido.loc[i + 1, columna_valor]

        corregir = False

        # Caso 1: valor igual a 0
        if valor == 0:
            corregir = True

        # Caso 2: valor extremadamente alto
        elif valor >= umbral_alto and anterior < umbral_anterior:
            corregir = True

        if corregir:
            nuevo_valor = round((anterior + siguiente) / 2)

            modificaciones.append({
                "fecha": df_corregido.loc[i, columna_fecha],
                "valor_original": valor,
                "valor_nuevo": nuevo_valor
            })

            df_corregido.loc[i, columna_valor] = nuevo_valor

    # Mostrar resumen
    if modificaciones:
        print("\nValores corregidos:")
        print("-" * 60)

        for m in modificaciones:
            print(
                f"{m['fecha']} | "
                f"Original: {m['valor_original']:>3} "
                f"--> Nuevo: {m['valor_nuevo']:>3}"
            )

        print("-" * 60)
        print(f"Total de correcciones: {len(modificaciones)}")
    else:
        print("No se encontraron valores para corregir.")

    return df_corregido


for archivo in archivos_csv:

    ruta_archivo = os.path.join(directorio, archivo)

    print(f"Procesando: {archivo}")

    # Leer CSV
    try:
        df_original = pd.read_csv(
            ruta_archivo,
            encoding="latin-1",
            sep=";",
            low_memory=False
        )
    except UnicodeDecodeError:
        try:
            df_original = pd.read_csv(
                ruta_archivo,
                encoding="utf-8",
                sep=";",
                low_memory=False
            )
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
            continue
    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")
        continue

    # Verificar columnas necesarias
    columnas_requeridas = [
        "NEstablecimiento",
        "GlosaCausa",
        "Total",
        "fecha"
    ]

    faltantes = [c for c in columnas_requeridas if c not in df_original.columns]

    if faltantes:
        print(f"Columnas faltantes en {archivo}: {faltantes}")
        continue

    # Filtrar establecimientos
    df = df_original[
        df_original["NEstablecimiento"].isin(establecimientos)
    ].copy()

    if df.empty:
        continue

    # Filtrar causas respiratorias
    df = df[
        df["GlosaCausa"].isin(causas)
    ].copy()

    if df.empty:
        continue

    # Mantener solo variables necesarias
    df = df[["fecha", "Total"]].copy()

    # Renombrar
    df.rename(
        columns={"Total": "total_atenciones"},
        inplace=True
    )

    # Convertir fecha
    df["fecha"] = pd.to_datetime(
        df["fecha"],
        dayfirst=True,
        errors="coerce"
    )

    # Eliminar fechas inválidas
    df = df.dropna(subset=["fecha"])

    # Convertir total a numérico
    df["total_atenciones"] = pd.to_numeric(
        df["total_atenciones"],
        errors="coerce"
    )

    df = df.dropna(subset=["total_atenciones"])

    # SUMAR TODOS LOS CENTROS DEL MISMO DÍA
    df = (
        df.groupby("fecha", as_index=False)["total_atenciones"]
          .sum()
    )

    dfs_procesados.append(df)

# Consolidación final
if not dfs_procesados:
    raise ValueError(
        "No se encontraron datos válidos para procesar."
    )

df_final = pd.concat(
    dfs_procesados,
    ignore_index=True
)

# Por si la misma fecha aparece en más de un archivo
df_final = (
    df_final.groupby("fecha", as_index=False)["total_atenciones"]
            .sum()
)

df_final = df_final.sort_values("fecha").reset_index(drop=True)

print("\nResumen final (antes de corregir outliers)")
print("=" * 50)
print(f"Fechas únicas: {len(df_final):,}")
print(
    f"Rango: {df_final['fecha'].min().date()} -> "
    f"{df_final['fecha'].max().date()}"
)
print(f"Suma total consultas: {df_final['total_atenciones'].sum():,.0f}")
print("=" * 50)

# Corregir outliers
df_final = corregir_outliers(df_final)

# Crear directorio de salida si no existe
os.makedirs("data/processed", exist_ok=True)

# Guardar
ruta_salida = "data/processed/datos_consultas_corregidos.csv"
df_final.to_csv(ruta_salida, index=False)

print("\nArchivo guardado:")
print(ruta_salida)