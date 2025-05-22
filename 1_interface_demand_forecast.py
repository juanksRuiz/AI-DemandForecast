import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

from model.utility_funcs import *

# ----- Streamlit App -----
st.set_page_config(page_title="Predicción de Demanda", layout="centered")

# Título y descripción
st.title("📦 Predicción de Demanda")
st.write("Predice la demanda futura de tus productos rápidamente.")

st.markdown("""
    <h1 style='font-size:32px;'>Sube tu CSV de ventas</h1>
""", unsafe_allow_html=True)


# Para que no salga error
max_date = datetime.today().date()

# Carga de archivo CSV
uploaded_file = st.file_uploader("", type=["csv"] )


# Definición de columnas
date_col = st.text_input("Columna de FECHA (nombre EXACTO):")
target_col = st.text_input("Columna de UNIDADES vendidas (nombre EXACTO):")

# Verificar si el archivo fue cargado
if uploaded_file is not None:
    # Leer el archivo como DataFrame de Pandas
    df = pd.read_csv(uploaded_file)

    if date_col not in df.columns or target_col not in df.columns:
        st.write("Revisa que hayas escrito el nombre exacto de cada columna")
    else:
        # Mostrar el DataFrame en la aplicación
        st.write("Contenido del archivo CSV:")
        st.dataframe(df[[date_col, target_col]])

# Mostrar mínimo y máximo de fechas una vez definidos
# df = None

if uploaded_file and date_col and target_col:
    df[str(date_col)] = pd.to_datetime(df[str(date_col)])
    min_date, max_date = df[date_col].min(), df[date_col].max()
    st.write(f"**Rango de fechas ingresadas:** {min_date.date()} al {max_date.date()}")

# Input fecha final
iforecast = st.date_input(
    "Fecha hasta la cual predecir:",
    value=datetime.today().date()  + pd.Timedelta(days=1)
)

#if pd.Timestamp(iforecast) > pd.Timestamp(max_date):
#    st.write("Fecha ingresada correctamente")
#else:
#    st.write("ERROR: Asegurate de ingresar una fecha maxima de prediccion mayor a la ultima fecha ingresada")
# -----------------------------------------------------------------------------


# Boton para prediccion:
if st.button("Estimar demanda"):
    # Generacion de columnas exogenas

    forecast_days = (iforecast - max_date.date()).days
    total_days = len(df) + forecast_days
    exog_vars_df = pd.DataFrame(generate_exogenous(min_date.strftime('%Y-%m-%d')
                                                   ,total_days
                                ))
    

    exog_vars_train_df = exog_vars_df.iloc[:len(df)]  # mismas filas que tu df original
    exog_vars_test_df = exog_vars_df.iloc[len(df):]   # filas para predicción futura


    # Llamada a la funcion qu hace prediccion cuando se hace clic
    predicciones = forecast_xgboost(df[target_col],
                                     (iforecast - max_date.date()).days
                                     , N_train=365
                                     ,exog_vars = (exog_vars_train_df, exog_vars_test_df)
                                     )
                                     
    st.write("Entrenaminto de modelo exitoso !")

    # Graficacion de datos
    # Ejemplo: Predicciones del modelo
    date_x_test = pd.date_range(max_date + pd.Timedelta(days=1), pd.Timestamp(iforecast), freq='D')
    date_x_test = pd.to_datetime(date_x_test)  # Convertir a formato de fecha

    # Título de la aplicación
    st.title('Demanda actual y Predicción de demanda')

    # Crear la gráfica

    ULTIMOS_DIAS_DATOS = 60


    grafica_df = pd.DataFrame({
        'fecha': list(df[date_col][-ULTIMOS_DIAS_DATOS:]) + list(date_x_test),
        'valor': list(df[target_col][-ULTIMOS_DIAS_DATOS:]) + list(predicciones),
        'tipo': (['Real'] * (ULTIMOS_DIAS_DATOS-1)) + (['Predicción'] * (len(predicciones)+1))
    })


    grafica_df['fecha'] = pd.to_datetime(grafica_df['fecha'])
    grafica_df['valor'] = grafica_df['valor'].astype(int)
    grafica_df = grafica_df.sort_values('fecha')

    st.write("Datos de Predicción:")
    st.write(grafica_df[grafica_df['tipo']=='Predicción'])

    # Crear el gráfico con Altair
    chart = alt.Chart(grafica_df).mark_line().encode(
        x='fecha',
        y='valor',
        color=alt.Color('tipo', scale=alt.Scale(domain=['Real', 'Predicción'], range=['lightblue', 'red'])),
        tooltip=['fecha', 'valor', 'tipo']
    ).properties(
        title='Pronóstico de Demanda'
    )

    # Mostrar el gráfico en Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Calcular la suma total de las predicciones
    total_predicciones = int(sum(predicciones))

    # Mostrar la cifra total debajo de la gráfica
    st.subheader(f"Cantidad de productos necesarios para los {len(predicciones)} Días: {total_predicciones}")