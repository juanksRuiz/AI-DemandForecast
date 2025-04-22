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


# ----- Helper Functions -----
def load_model(path):
    """Carga el modelo de ML serializado."""
    return joblib.load(path)


def make_forecast(model, df, date_col, target_col, freq, forecast_end):
    """
    Genera predicciones desde el último valor hasta forecast_end con frecuencia dada.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    last_date = df[date_col].max()
    freq_map = {'Diaria': 'D', 'Semanal': 'W', 'Mensual': 'M'}
    pd_freq = freq_map.get(freq, 'D')
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit=pd_freq),
                                 end=forecast_end,
                                 freq=pd_freq)
    future_df = pd.DataFrame({date_col: future_dates})
    # Aquí adapta según tu pipeline real
    X_future = future_df[date_col].map(datetime.toordinal).values.reshape(-1, 1)
    preds = model.predict(X_future)
    future_df['prediction'] = preds
    return future_df

def rolling_slope(series, window=14):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series[i-window:i].values.reshape(-1, 1)
            x = np.arange(window).reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            slopes.append(reg.coef_[0][0])
    return slopes

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
    # df = pd.read_csv(uploaded_file)
    print(f"date_col: {date_col}")
    print(f"target_col: {target_col}")
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
# Funcion que sera llamad al hacer clic en un boton
def forecast_demanda_xgboost(test_df):
    #st.write("Funcion ejecutada exitosamente !")
    # Llamar modelo XGBoost entrenado
    ruta_modelo = os.path.join('.', 'model', 'xgboost_demand_forecast_with_exogenous-0.1.0.pkl')
    with open(ruta_modelo, 'rb') as file:
        trained_xgboost = pickle.load(file)

    # Indicarle columnas usadas en entrenamiento
    train_cols = ['demanda_lag_1', 'demanda_lag_2', 'rolling_trend', 'rolling_slope_14',
       'es_fin_de_semana', 'is_feb_29', 'mes_sin', 'mes_cos', 'dia_sin',
       'dia_cos', 'dia_de_semana_sin', 'dia_de_semana_cos',
       'flg_despues_llegada_competencia']
    
    # Generacion de columnas
    # Preprocesamiento de la serie de tiempo
    test_df['demanda'].ffill(inplace=True)
    test_df['demanda'].bfill(inplace=True)
    test_df = test_df.sort_values(by='date', ascending=True)
    
    exog_vars_dict = generate_exogenous(test_df['date'].min(), test_df.shape[0])
    exog_vars_df = pd.DataFrame(exog_vars_dict)
    exog_vars_df['date'] = test_df['date']

    if 'demanda_lag_1' not in test_df.columns or 'demanda_lag_2' not in test_df.columns:
        test_df['demanda_lag_1'] = test_df['demanda'].shift(1)
        test_df['demanda_lag_2'] = test_df['demanda'].shift(2)

    # inclusion de media movil:
    test_df['rolling_mean_7'] = test_df['demanda'].shift(1).rolling(window=7).mean()
    test_df['rolling_trend'] = test_df['demanda'].shift(1) - test_df['rolling_mean_7']

    test_df['rolling_slope_14'] = rolling_slope(test_df['demanda'], window=14)
    
    # Seleccionar esas columnas del dataset cargado (si en el datasset cargado no estan, hay crearlas)
    X_test = test_df[train_cols]

    # Retornar las predicciones en una lista
    return list(trained_xgboost.predict(X_test))
# -----------------------------------------------------------------------------

# Boton para prediccion:
if st.button("Estimar demanda"):
    # Llamada a la funcion qu hace prediccion cuando se hace clic
    predicciones = forecast_demanda_xgboost(df) # CORREGIRIIIR
    st.write("Entrenaminto de modelo exitoso !")

    # Graficacion de datos
    # Ejemplo: Predicciones del modelo
    date_x_test = df[date_col]
    date_x_test = pd.to_datetime(date_x_test)  # Convertir a formato de fecha

    # Título de la aplicación
    st.title('Demanda actual y Predicción de demanda')

    # Crear la gráfica
    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[target_col], label='Ultima demanda ingresada', marker='o')
    ax.plot(date_x_test, predicciones, label='Predicciones', marker='x', linestyle='--')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Demanda')
    ax.set_title('Comparación entre Demanda Real y Predicción')
    ax.legend()
    plt.xticks(rotation=45)

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)