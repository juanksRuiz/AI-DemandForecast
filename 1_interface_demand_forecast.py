# ---- Pruebas a validar ----
# 1. Reconocimiento de archivo subido - check
# 2. Visualizacion de tabla - check
# 3. Fechas hasta la cual predecir estrictamente mayor a la fecha maxima pasada
# 4. Boton de predecir
# 5. Si prediccion satisfactoria graficar datos de entrenamiento y de test en colores diferentes


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

import pickle
import os
from model.utility_funcs import *

from sklearn.linear_model import LinearRegression


# ----- Streamlit App -----
st.set_page_config(page_title="Predicción de Demanda", layout="wide")

# Título y descripción
st.title("📦 Predicción de Demanda")
st.write("Predice la demanda futura de tus productos rápidamente.")

# Input de frecuencia
dfreq = st.selectbox(
    "Selecciona frecuencia de tus datos:",
    options=["Diaria", "Semanal", "Mensual"]
)

# Para que no salga error
max_date = datetime.today().date()

# Carga de archivo CSV
uploaded_file = st.file_uploader("Sube tu CSV de ventas", type=["csv"] )
# Verificar si el archivo fue cargado
if uploaded_file is not None:
    # Leer el archivo como DataFrame de Pandas
    df = pd.read_csv(uploaded_file)

    # Mostrar el DataFrame en la aplicación
    st.write("Contenido del archivo CSV:")
    st.dataframe(df)

# Definición de columnas
date_col = st.text_input("Columna de FECHA (nombre EXACTO):")
target_col = st.text_input("Columna de UNIDADES vendidas (nombre EXACTO):")

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

if pd.Timestamp(iforecast) > pd.Timestamp(max_date):
    st.write("Fecha ingresada correctamente")
else:
    st.write("ERROR: Asegurate de ingresar una fecha maxima de prediccion mayor a la ultima fecha ingresada")
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
    grafica_df = pd.DataFrame({
        'fecha': list(df[date_col]) + list(date_x_test),
        'valor': list(df[target_col]) + list(predicciones),
        'tipo': (['Real'] * len(df)) + (['Predicción'] * len(predicciones))
    })
    
    grafica_df['valor'] = grafica_df['valor'].astype(int)

    # Configurarlo para que Streamlit lo entienda
    grafica_df = grafica_df.sort_values('fecha')
    st.write(grafica_df[grafica_df['tipo']=='Predicción'])

    # Graficar usando st.line_chart agrupando por tipo
    st.line_chart(
        data=grafica_df,
        x='fecha',
        y='valor',
        color='tipo'
    )


