import numpy as np
import pandas as pd

# Evaluacion de la descomposicion de la serie con STL
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import levene
import xgboost as xgb

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import os
import pickle

import copy


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
    
    exog_var_names = ['es_fin_de_semana', 'is_feb_29'
                      , 'mes_sin', 'mes_cos', 'dia_sin', 'dia_cos'
                      , 'dia_de_semana_sin', 'dia_de_semana_cos'
                      , 'flg_despues_' + event_name
                      ]
    dict_exog_vars = {exog_var_names[i]: final_array[i] for i in range(len(final_array))}
    
    return dict_exog_vars


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
def generate_exogenous(initial_date, steps):
        """Genera las variables exógenas necesarias a partir de las fechas
        Args:
            - initial_date (str): string de fecha desde la cual se haran las predicciones
            - steps (int): cantidad de dias posteriores a la ultima fecha de entrenamioento usada en el train
        Returns:
            dataframe con variables exogenas para test
        """
        fecha_inicial = pd.to_datetime(initial_date)

        # Generar la serie de fechas
        serie_fechas = pd.date_range(start=fecha_inicial, 
                                    end=fecha_inicial + pd.Timedelta(days=steps), 
                                    freq='D')

        # Convertir a pandas Series
        test_dates = pd.Series(serie_fechas)

        test_exog_df = create_exog_cols_from_date_series(test_dates
                                                           , event_date='2021-07-02'
                                                           , event_name='llegada_competencia')
        return test_exog_df

#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
def get_XGBoost_features(y_var):
    """
    Calcula las variables necesarias (lags y tendencia) a partir de la variable objetivo.
    Pensado para predicción recursiva con XGBoost.
    """
    # Copiar la variable para evitar modificar el original
    y_var_copy = copy.copy(y_var)
    df = pd.DataFrame({'demanda': y_var_copy})

    # Validar tipo de datos inicial
    #print("Tipo inicial de 'demanda':", df['demanda'].dtype)

    # Convertir 'demanda' a tipo numérico
    df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')

    # Validar después de la conversión
    #print("Tipo después de convertir 'demanda':", df['demanda'].dtype)
    #print("Valores únicos de 'demanda':", df['demanda'].unique())

    # Manejo de valores faltantes
    df['demanda'].ffill(inplace=True)
    df['demanda'].bfill(inplace=True)

    # Validar después de manejar valores faltantes
    #print("Número de valores NaN después de ffill y bfill:", df['demanda'].isna().sum())

    # Lags
    df['demanda_lag_1'] = df['demanda'].shift(1)
    df['demanda_lag_2'] = df['demanda'].shift(2)

    # Validar lags
    #print("Primeras filas con lags:\n", df[['demanda', 'demanda_lag_1', 'demanda_lag_2']].head())

    # Tendencia y pendiente
    df['rolling_mean_7'] = df['demanda'].shift(1).rolling(window=7).mean()

    # Validar media móvil
    #print("Primeras filas con rolling_mean_7:\n", df[['demanda', 'rolling_mean_7']].head(10))

    df['rolling_trend'] = df['demanda'].shift(1) - df['rolling_mean_7']
    df.drop(columns=['rolling_mean_7'], inplace=True)

    # Validar rolling_trend
    #print("Primeras filas con rolling_trend:\n", df[['demanda', 'rolling_trend']].head(10))

    # Pendiente (rolling slope)
    df['rolling_slope_14'] = rolling_slope(df['demanda'], window=14)

    # Validar rolling_slope_14
    #print("Primeras filas con rolling_slope_14:\n", df[['demanda', 'rolling_slope_14']].head(10))

    # Eliminar las filas con NaN para evitar errores
    df.dropna(inplace=True)

    # Validar datos finales
    #print("Datos finales después de eliminar NaN:\n", df.head())

    # Devuelve el DataFrame procesado y el índice sincronizado
    return df, df.index

#------------------------------------------------------------------------------
def forecast_xgboost(target_var, steps, exog_vars, N_train=None):
    """
    Realiza forecast (predicción recursiva) con un modelo XGBoost.

    Args:
        target_var (list or array): Serie histórica de la variable objetivo (ej. ventas).
        steps (int): Número de pasos (días) a predecir.
        N_train (int): Tamaño de la ventana de datos históricos.
        exog_vars (tuple): Tupla de datos de entrenamiento y prueba de variables exogenas
    Returns:
        list: Predicciones para los próximos 'steps' días.
    """
    
    if N_train is not None:
        assert N_train <= len(target_var), "ERROR: la ventana de entrenamiento excede la longitud de la serie"
    else:
        N_train = len(target_var)

    # 1. Crear modelo xgboost
    reg = reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1
    )
    
    # Opcional:
    #integrar modulo de cross validation para que determine mejores parametros
    # y regularizacion

    # 2. Entrenar modelo generando training features
    # Generar features
    X_train, indices = get_XGBoost_features(target_var)

    # Sincronizar target_var con X_train y crear variable temporal de la serie
    temp_target_var = list(target_var.loc[indices].copy())  # Evita modificar el original
    
    exog_vars_train_df, exog_vars_test_df = exog_vars

    # Creacion de variables de entrenamiento finales
    final_X_train = pd.concat([X_train, exog_vars_train_df], axis=1)
    final_X_train = final_X_train.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)

    # Entrenamiento usando la ventana final de N_train
    reg.fit(final_X_train.iloc[-N_train:, :], temp_target_var[-N_train:], verbose=False)


    # 3. Inicializar lista de predicciones
    predictions = []
    temp_window = temp_target_var[-N_train:]  # Inicia con la ventana de tamaño N_train

    # 4. Predicción paso a paso
    for step in range(steps):
        # 3.1 Features de la ventana actual
        X_test_step, _ = get_XGBoost_features(temp_window)

        # 3.2 Unir con variables exógenas del step actual
        X_exog_step = exog_vars_test_df.iloc[[step]]  # DataFrame de 1 fila
        X_input = pd.concat([X_test_step.iloc[[-1]], X_exog_step], axis=1)

        X_input = X_input.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)

        # 3.3 Predecir
        y_pred = reg.predict(X_input)[0]
        predictions.append(y_pred)

        # 3.4 Actualizar ventana
        temp_window.append(y_pred)
        temp_window = temp_window[-N_train:]  # Mantener siempre tamaño N_train


    return predictions
