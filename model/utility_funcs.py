import numpy as np
import pandas as pd

# Evaluacion de la descomposicion de la serie con STL
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene

import matplotlib.pyplot as plt

import joblib

def create_exog_cols_from_date_series(y, event_date=None, event_name=None):
    # Crea variables exogenas a partir de la columna de las fechas
    # Retorna: copia de dataframe original con las variables exogenas
    y = pd.to_datetime(y)
    y_anio = y.dt.year
    y_mes = y.dt.month
    y_dia = y.dt.day
    y_dia_de_semana = y.dt.dayofweek

    y_es_fin_de_semana = y.dt.weekday.isin([5,6]).astype(int)
    

    # Crear una columna indicadora para el 29 de febrero
    y_is_feb_29 = (y.dt.month == 2) & (y.dt.day == 29)

    # Encoding ciclico para variables ciclicas
    y_mes_sin = np.sin(2 * np.pi *y_mes / 12)
    y_mes_cos = np.cos(2 * np.pi *y_mes / 12)

    y_dia_sin = np.sin(2 * np.pi *y_dia / 31)
    y_dia_cos = np.cos(2 * np.pi *y_dia / 31)

    y_dia_de_semana_sin = np.sin(2 * np.pi *y_dia_de_semana / 7)
    y_dia_de_semana_cos = np.cos(2 * np.pi *y_dia_de_semana / 7)

    # Datos booleanos transformados a enteros 1 o 0
    final_array = [y_es_fin_de_semana, y_is_feb_29
                     , y_mes_sin, y_mes_cos, y_dia_sin, y_dia_cos
                     , y_dia_de_semana_sin, y_dia_de_semana_cos]
    # Flag de datos despues de aparicion de tienda de la competencia
    if event_date is not None:
        fecha_cambio = pd.Timestamp(event_date)
        y_flg_event_name = (y >= fecha_cambio).astype(int)
    else:
        y_flg_event_name = np.array(len(y)*[False])
    final_array.append(y_flg_event_name)
    
    return np.array(final_array)


#------------------------------------------------------------------------------
def get_diff_order(df, col, make_plot=False):
    # Determina el orden de diferenciacion de una serie de tiempo de un dataframe
    # Retorna: nada

    copy_df = df.copy()
    # Aplicar la primera diferenciación
    copy_df['demanda_diff_1'] = copy_df[col].diff()

    # Eliminar valores NaN generados por la diferenciación
    copy_df.dropna(inplace=True)
    copy_df.reset_index(inplace=True, drop=True)

    if make_plot:
        # Graficar la serie diferenciada
        plt.figure(figsize=(10,5))
        plt.plot(copy_df[col +'_diff_1'], label='Serie diferenciada (1ra orden)')
        plt.legend()
        plt.title('Serie Diferenciada - Primera Diferenciación')
        plt.show()

    # Prueba ADF en la serie diferenciada
    adf_result = adfuller(copy_df[col+'_diff_1'])
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print(f'Critical Values:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value}')

    # Verificar si aún no es estacionaria (p-valor > 0.05), aplicar segunda diferenciación
    if adf_result[1] > 0.05:
        print("\nLa serie sigue sin ser estacionaria. Aplicando segunda diferenciación...\n")

        # Segunda diferenciación
        copy_df[col+'_diff_2'] = copy_df[col+'_diff_1'].diff()
        copy_df.dropna(inplace=True)
        copy_df.reset_index(inplace=True, drop=True)
        # Graficar la serie diferenciada (2da orden)
        if make_plot:
            plt.figure(figsize=(10,5))
            plt.plot(copy_df[col+'_diff_2'], label='Serie diferenciada (2da orden)')
            plt.legend()
            plt.title('Serie Diferenciada - Segunda Diferenciación')
            plt.show()

        # Prueba ADF en la serie diferenciada (2da orden)
        adf_result_2 = adfuller(copy_df[col+'_diff_2'])
        print(f'ADF Statistic (2da orden): {adf_result_2[0]}')
        print(f'p-value: {adf_result_2[1]}')
        print(f'Critical Values:')
        for key, value in adf_result_2[4].items():
            print(f'   {key}: {value}')

    # Verifica si ahora la serie es estacionaria
    if adf_result[1] < 0.05:
        print("\n✅ La serie se volvió estacionaria con una sola diferenciación.")
    elif 'adf_result_2' in locals() and adf_result_2[1] < 0.05:
        print("\n✅ La serie se volvió estacionaria con dos diferenciaciones.")
    else:
        print("\n⚠️ La serie aún no es estacionaria. Considera otras transformaciones (log, sqrt).")
#------------------------------------------------------------------------------

def get_STL_series_decomposition(df, y_col, date_col, seasonal, flg_plot=True):
    # Grafica Descomposicion de serie de tiempo de un dataframe con metodo STL
    # Retorna: Objeto STL con la descomposicion de lo observadom tendencia, estacionalidad y residuo


    stl = STL(df[y_col], period=7)
    descomposicion = stl.fit()
    if flg_plot:
        # Graficar los componentes de la descomposición
        plt.figure(figsize=(12, 6))

        # Observación original
        plt.subplot(411)
        plt.plot(df[date_col], descomposicion.observed)
        # Formatear los ticks para que muestren solo el mes y el anio
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.title('Observación')

        # Tendencia
        plt.subplot(412)
        plt.plot(df[date_col], descomposicion.trend)
        # Formatear los ticks para que muestren solo el mes y el anio
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.title('Tendencia')

        # Estacionalidad
        plt.subplot(413)
        plt.plot(df[date_col], descomposicion.seasonal)
        # Formatear los ticks para que muestren solo el mes y el anio
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.title('Estacionalidad')

        # Residual
        plt.subplot(414)
        plt.plot(df[date_col], descomposicion.resid)
        # Formatear los ticks para que muestren solo el mes y el anio
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.title('Residual')

        plt.tight_layout()
        plt.show()
    return descomposicion

#------------------------------------------------------------------------------
# CLASE
class ForecastModelWithExogenous:
    def __init__(self, model):
        self.model = model
        self.min_train_date = None
        self.max_train_date = None

    def generate_exogenous(self, dates):
        """Genera las variables exógenas necesarias a partir de las fechas"""
        dates = pd.to_datetime(dates)
        exogenous_vars = create_exog_cols_from_date_series(dates, event_date='2021-07-02', event_name='llegada_competencia')
        return exogenous_vars
    
    def forecast(self, steps, exogenous_dates):
        """Genera las predicciones utilizando variables exógenas"""
        exogenous_vars = self.generate_exogenous(exogenous_dates)
        return self.model.forecast(steps=steps, exog=exogenous_vars)