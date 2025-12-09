# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:41:07 2025

@author: cami_
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


# Importante: Esta función asume que la columna de tiempo en df_data se llama 'datetime'.
def _filter_df_by_date(df, fecha_inicio, fecha_fin):
    """Aplica el filtro de fechas al DataFrame si se proporcionan las fechas."""
    if fecha_inicio is None and fecha_fin is None:
        return df.copy()

    df_filtered = df.copy()
    
    if 'datetime' not in df_filtered.columns:
        print("Advertencia: Columna 'datetime' no encontrada para filtrar.")
        return df_filtered
        
    if fecha_inicio:
        inicio = pd.to_datetime(fecha_inicio).date()
        df_filtered = df_filtered[df_filtered['datetime'].dt.date >= inicio]
    if fecha_fin:
        # Aseguramos que incluya todo el día final
        fin = pd.to_datetime(fecha_fin).date()
        df_filtered = df_filtered[df_filtered['datetime'].dt.date <= fin]
        
    return df_filtered


def calculate_trm(df_data, col_temp_ext, data_frequency_min=60):
    """
    Calcula la Runing Mean Temperature (Trm) como el promedio móvil 
    de 7 días anteriores de la temperatura exterior (Tamb).

    Args:
        df_data (pd.DataFrame): DataFrame con la columna de temperatura exterior.
        col_temp_ext (str): Nombre de la columna de temperatura exterior (Text).
        data_frequency_min (int, opcional): Frecuencia de los datos en minutos.

    Returns:
        pd.Series: Una serie de Pandas con el valor de Trm.
    """
    if data_frequency_min <= 0 or 60 % data_frequency_min != 0:
        raise ValueError("data_frequency_min debe ser un divisor de 60 (ej: 60, 30, 15).")
        
    intervalos_por_hora = 60 / data_frequency_min
    # Ventana de 7 días para Trm
    intervalo_dias = int(intervalos_por_hora * 24 * 7)
    
    # Cálculo de la Trm
    # Se utiliza .bfill() para rellenar los NaNs iniciales generados por la ventana de rolling.
    # Al usar .bfill() le estoy asignando a los primeros 6 dias la misma Trm que la del dia 7

    trm_series = df_data[col_temp_ext].shift(1).rolling(window=intervalo_dias).mean().bfill()
    
    return trm_series


def calculate_tconf(trm_series, building_category='III', model_type='freeruning'):
    """
    Calcula la Temperatura de Confort (Tconf) o el umbral superior de temperatura 
    (T_upper) según la Trm y la Categoría ISO.

    Args:
        trm_series (pd.Series): Serie de Pandas con los valores de Trm.
        building_category (str): Categoría ISO ('I', 'II', 'III', 'IV').
        model_type (str): 'freeruning' o 'airconditioned'. (Aplica la lógica Tconf = T_upper).

    Returns:
        pd.Series: Una serie de Pandas con el valor de Tconf o T_upper.
    """
    model_type = model_type.lower()
    
    if model_type == 'freeruning':
        # Definir la constante A
        if building_category == 'I':
            A = 20.8
        elif building_category == 'II':
            A = 21.8
        elif building_category == 'III':
            A = 22.8 
        elif building_category == 'IV':
            A = 23.8
        else:
            raise ValueError("Categoría de edificio no válida. Use 'I', 'II', 'III', o 'IV'.")
            
        tconf_series = 0.33 * trm_series + A
        
        # Limite superior por si da muy grande la Tconf        
        tconf_series = np.where(tconf_series > 27, 27, tconf_series)
        
    elif model_type == 'airconditioned':
        # Definir T_upper estático
        if building_category == 'I':
            T_upper_airconditioned = 25.5
        elif building_category == 'II':
            T_upper_airconditioned = 26.0
        elif building_category == 'III':
            T_upper_airconditioned = 27.0
        elif building_category == 'IV':
            T_upper_airconditioned = 28.0
        else:
            raise ValueError("Categoría de edificio no válida. Use 'I', 'II', 'III', o 'IV'.")
            
        tconf_series = pd.Series(T_upper_airconditioned, index=trm_series.index)
        
    else:
        raise ValueError("Tipo de modelo no válido. Use 'freeruning' o 'airconditioned'.")
        
    return tconf_series


def calculate_heat_index(df_data, col_temp, col_humidity, fecha_inicio=None, fecha_fin=None):
    """
    Calcula el Índice de Calor (Heat Index - HI) para el periodo especificado
    (si se provee) o para todo el DataFrame (por defecto).

    Args:
        df_data (pd.DataFrame): DataFrame con todas las columnas necesarias.
        col_temp (str): Nombre de la columna de temperatura del aire int (Ta).
        col_humidity (str): Nombre de la columna de humedad relativa int (RH).
        fecha_inicio (str, optional): Fecha de inicio del análisis ('YYYY-MM-DD'). Por defecto None.
        fecha_fin (str, optional): Fecha de fin del análisis ('YYYY-MM-DD'). Por defecto None.

    Returns:
        pd.Series: Una serie de Pandas con el valor del Heat Index (HI) calculado.
    """
    
    # 0. APLICAR FILTRO DE FECHAS
    df_filtered = _filter_df_by_date(df_data, fecha_inicio, fecha_fin)

    if df_filtered.empty:
        print("ADVERTENCIA: El DataFrame filtrado por fechas está vacío. Retornando serie vacía.")
        return pd.Series([], dtype=float, name='HeatIndex')
        
    Ta = df_filtered[col_temp]
    RH = df_filtered[col_humidity]
    
    # --- 1. Calcular la fórmula compleja (Rothfusz) PRIMERO ---
    
    # Términos de la ecuación
    T2 = Ta * Ta
    R2 = RH * RH
    TR = Ta * RH
    T2R = T2 * RH
    TR2 = Ta * R2
    T2R2 = T2 * R2
    
    hi_complejo = (
        -8.785
        + 1.61139411 * Ta
        + 2.338549 * RH
        - 0.14611605 * TR
        - 0.012308094 * T2
        - 0.016424828 * R2
        + 0.002211732 * T2R
        + 0.00072546 * TR2
        - 0.000003582 * T2R2
    )
    
    # --- 2. Calcular la fórmula lineal simple ---
    hi_simple = (1.1 * Ta) + (0.0261 * RH) - 3.94
    
    # --- 3. Aplicar la condición ---    
    heat_index_final = np.where(hi_complejo < 26.7, hi_simple, hi_complejo)
    
    # Devolver como una Serie de Pandas. Hay que usar el índice del df_filtered
    # para que coincida con el resto de los datos en el bloque principal de ejecución.
    return pd.Series(heat_index_final, index=df_filtered.index, name='HeatIndex')


def calculate_iod(df_data, col_temp_int, col_temp_ext, inicio_horas=7, fin_horas=23, 
                  group_by_day=True, data_frequency_min=60,
                  fecha_inicio=None, fecha_fin=None):
    """
    Calcula el Grado de Sobrecalentamiento Interior (IOD) basado en un DataFrame 
    único, reutilizando la función de filtrado de fechas.
    """
    
    # Aseguramos que 'df_data' tenga una columna 'datetime'
    if 'datetime' not in df_data.columns:
        raise ValueError("El DataFrame de entrada debe tener una columna llamada 'datetime'.")
            
    # --- 1. FILTRADO POR RANGO DE FECHAS (Reutilizando la función auxiliar) ---
    # Asumimos que la columna 'datetime' ya está en formato datetime 
    df = _filter_df_by_date(df_data, fecha_inicio, fecha_fin)


    # Chequeo de datos después del filtro
    if df.empty:
        if group_by_day:
             return pd.DataFrame({'Fecha': [], 'IOD_Diario': []})
        else:
             return 0.0
    
    # --- 2. CONFIGURACIÓN DE FRECUENCIA ---
    if data_frequency_min <= 0 or data_frequency_min > 60 or 60 % data_frequency_min != 0:
          raise ValueError("data_frequency_min debe ser un divisor de 60 (ej: 60, 30, 15).")
          
    intervalos_por_hora = 60 / data_frequency_min
    
    df['Tint'] = df[col_temp_int]
    df['Text'] = df[col_temp_ext]

    # Cálculo de la Trm (usa el df ya filtrado)
    df['Trm'] = calculate_trm(df, col_temp_ext, data_frequency_min)
    
    # Cálculo de la temperatura de confort (Tconf) - Usamos Cat III por defecto en IOD
    df['Tconf'] = calculate_tconf(df['Trm'], building_category='III', model_type='freeruning')
    
    # Cálculo del numerador del IOD (sólo si Tint > Tconf)
    df['DeltaT'] = (df.Tint - df.Tconf)
    df['Num'] = np.where(df.DeltaT > 0, df.DeltaT, 0) # Si DeltaT o Tconf es NaN, Num será 0
    
    # Filtrar solo el período de ocupación
    df_ocupacion = df[(df.datetime.dt.hour >= inicio_horas) & (df.datetime.dt.hour < fin_horas)].copy()
    
    
    # --- 3. AGREGACIÓN (DIARIO vs. TOTAL) ---
    
    # El divisor es el número de puntos de datos esperados en el período de ocupación de un día
    divisor_diario = (fin_horas - inicio_horas) * intervalos_por_hora
    
    if group_by_day:
        # CÁLCULO DIARIO
        df_ocupacion['Fecha'] = df_ocupacion['datetime'].dt.date
        
        df_agregado = df_ocupacion.groupby('Fecha').agg(
            Suma_Num=('Num', 'sum'),
            Conteo_Intervalos=('Num', 'size')
        ).reset_index()
        
        # El IOD Diario es la suma de los excesos de temperatura, dividido por los intervalos esperados
        df_agregado['IOD_Diario'] = df_agregado['Suma_Num'] / divisor_diario
        
        return df_agregado[['Fecha', 'IOD_Diario']]
        
    else:
        # CÁLCULO TOTAL
        
        suma_num_total = df_ocupacion['Num'].sum()
        conteo_dias = df_ocupacion['datetime'].dt.date.nunique()
        
        # El divisor total es el número de días * divisor diario
        divisor_total = conteo_dias * divisor_diario
        
        if divisor_total > 0:
            iod_total = suma_num_total / divisor_total
            return iod_total
        else:
            return 0.0    


def calculate_awd(df_data, col_temp_ext, temp_base=18, group_by_day=True, time_interval_h=1,
                  fecha_inicio=None, fecha_fin=None):
    """
    Calcula el Ambient Warmness Degree (AWD), reutilizando el filtro de fechas.
    """
    
    # Aseguramos que 'df_data' tenga una columna 'datetime'
    if 'datetime' not in df_data.columns:
        raise ValueError("El DataFrame de entrada debe tener una columna llamada 'datetime'.")
    
    # --- 1. FILTRADO POR RANGO DE FECHAS (Reutilizando la función auxiliar) ---
    df = _filter_df_by_date(df_data, fecha_inicio, fecha_fin)
    # --- FIN FILTRADO ---

    if df.empty:
        if group_by_day:
             return pd.DataFrame({'Fecha': [], 'AWD_Diario': []})
        else:
             return 0.0
    
    # --- 2. CÁLCULO DEL NUMERADOR ---
    
    # Diferencia de temperatura (solo si es positiva)
    df['DeltaT'] = df[col_temp_ext] - temp_base
    df['Numerador'] = np.where(df.DeltaT > 0, df.DeltaT, 0)
    
    # Ponderar por el intervalo de tiempo (Delta_t)
    df['Numerador_Ponderado'] = df['Numerador'] * time_interval_h
    
    # --- 3. AGREGACIÓN (DIARIO vs. TOTAL) ---
    
    if group_by_day:
        # A. Cálculo Diario
        
        df['Fecha'] = df['datetime'].dt.date
        
        # Agregación por día
        df_agregado = df.groupby('Fecha').agg(
            Suma_Numerador=('Numerador_Ponderado', 'sum'),
            Conteo_Intervalos=('Numerador_Ponderado', 'size') # Usamos size para contar filas por día
        ).reset_index()
        
        # Denominador diario: Conteo_Intervalos * time_interval_h
        df_agregado['Denominador'] = df_agregado['Conteo_Intervalos'] * time_interval_h
        
        # Cálculo final: Suma(Numerador_Ponderado) / Denominador
        df_agregado['AWD_Diario'] = np.where(
            df_agregado['Denominador'] > 0,
            df_agregado['Suma_Numerador'] / df_agregado['Denominador'],
            0
        )
        
        return df_agregado[['Fecha', 'AWD_Diario']]
        
    else:
        # B. Cálculo Total (para todo el período filtrado)
        
        suma_numerador_total = df['Numerador_Ponderado'].sum()
        conteo_intervalos_total = len(df)
        
        denominador_total = conteo_intervalos_total * time_interval_h
        
        if denominador_total > 0:
            awd_total = suma_numerador_total / denominador_total
            return awd_total
        else:
            return 0.0
        
        
def calculate_alpha_iod(
    df_data, 
    col_temp_int, 
    col_temp_ext, 
    inicio_horas=7, 
    fin_horas=23, 
    temp_base_awd=18, 
    group_by_day=True, 
    data_frequency_min=60,
    fecha_inicio=None, fecha_fin=None
):
    """
    Calculate Overheating escalation factor (alpha_IOD).
    
    alpha_IOD = IOD / AWD (base temp 18°C)
    
    Esta función es un envoltorio que llama a calculate_iod y calculate_awdtb.

    Args:
        df_data (pd.DataFrame): DataFrame con todas las columnas necesarias.
        col_temp_int (str): Nombre de la columna de temperatura interior.
        col_temp_ext (str): Nombre de la columna de temperatura exterior.
        inicio_horas (int): Hora de inicio para IOD.
        fin_horas (int): Hora de fin para IOD.
        temp_base_awd (float, opcional): Temperatura base para AWD. Por defecto es 18.
        group_by_day (bool, opcional): Si es True (defecto), calcula el alpha diario. 
                                     Si es False, calcula el alpha total para el período.
        data_frequency_min (int, opcional): Frecuencia de los datos en minutos (ej: 60 para horario).
        fecha_inicio (str, opcional): Fecha de inicio del cálculo (ej: '2023-01-01').
        fecha_fin (str, opcional): Fecha de fin del cálculo (ej: '2023-12-31').

    Returns:
        pd.DataFrame or float: DataFrame con 'Fecha' y 'alpha_IOD', o el alpha total como float.
    """
    
    # Calcular el intervalo de tiempo en horas para AWD
    time_interval_h = data_frequency_min / 60.0

    if group_by_day:
        # --- 1. CÁLCULO DIARIO ---
        
        # 1.1 Calcular IOD Diario
        df_iod = calculate_iod(
            df_data=df_data.copy(), 
            col_temp_int=col_temp_int, 
            col_temp_ext=col_temp_ext,
            inicio_horas=inicio_horas, 
            fin_horas=fin_horas,
            group_by_day=True,
            data_frequency_min=data_frequency_min,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin    
        )
        
        # 1.2 Calcular AWD Diario
        df_awd = calculate_awd(
            df_data=df_data.copy(), 
            col_temp_ext=col_temp_ext, 
            temp_base=temp_base_awd,
            group_by_day=True,
            time_interval_h=time_interval_h,
            fecha_inicio=fecha_inicio, 
            fecha_fin=fecha_fin      
        )
        
        # Si alguno de los DataFrames está vacío (porque no había datos en el rango de fechas)
        if df_iod.empty or df_awd.empty:
            return pd.DataFrame({'Fecha': [], 'IOD_Diario': [], 'AWD_Diario': [], 'alpha_IOD': []})
        
        # 1.3 Mergear y calcular Alpha
        df_merged = pd.merge(df_iod, df_awd, on='Fecha', how='left')
        
        # 1.4 Calcular Alpha (con protección de división por cero)
        # Si AWD es 0 (no hubo estrés externo), alpha es 0.
        df_merged['alpha_IOD'] = np.where(
            df_merged['AWD_Diario'] > 0,
            df_merged['IOD_Diario'] / df_merged['AWD_Diario'],
            0 
        )
        
        return df_merged[['Fecha', 'IOD_Diario', 'AWD_Diario', 'alpha_IOD']]

    else:
        # --- 2. CÁLCULO TOTAL ---
        
        # 2.1 Calcular IOD Total
        iod_total = calculate_iod(
            df_data=df_data.copy(), 
            col_temp_int=col_temp_int, 
            col_temp_ext=col_temp_ext,
            inicio_horas=inicio_horas, 
            fin_horas=fin_horas,
            group_by_day=False,
            data_frequency_min=data_frequency_min,
            fecha_inicio=fecha_inicio, # PASAR ARGUMENTO
            fecha_fin=fecha_fin      # PASAR ARGUMENTO
        )

        # 2.2 Calcular AWD Total
        awd_total = calculate_awd(
            df_data=df_data.copy(), 
            col_temp_ext=col_temp_ext, 
            temp_base=temp_base_awd,
            group_by_day=False,
            time_interval_h=time_interval_h,
            fecha_inicio=fecha_inicio, # PASAR ARGUMENTO
            fecha_fin=fecha_fin      # PASAR ARGUMENTO
        )
        
        # 2.3 Calcular Alpha Total (con protección de división por cero)
        if awd_total > 0:
            alpha_total = iod_total / awd_total
            return alpha_total
        else:
            return 0.0
        

def calculate_wet_bulb_temp_stull(Ta, RH):
    """
    Estima la temperatura de bulbo húmedo (Tw) utilizando la aproximación de Stull.
    Válido para condiciones cercanas al nivel del mar.
    """
    # Casteo por si alguien llama solo a esta funcion pasandole una lista de valores y no una pd.Series
    Ta = pd.Series(Ta)
    RH = pd.Series(RH, index=Ta.index)

    # Fórmula de Stull
    Tw = (
        Ta * np.arctan(0.151977 * (RH + 8.313659)**0.5)
        + np.arctan(Ta + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )
    return Tw


def calculate_wbgt(df_data, col_temp, col_humidity, fecha_inicio=None, fecha_fin=None):
    """
    Calcula la Wet-Bulb Globe Temperature (WBGT) para cada fila de un DataFrame.
    
    WBGT = 0.67 * Tw + 0.33 * Ta
    
    Args:
        df_data (pd.DataFrame): DataFrame con todas las columnas necesarias.
        col_temp (str): Nombre de la columna de temperatura del aire int (Ta).
        col_humidity (str): Nombre de la columna de humedad relativa int (RH).

    Returns:
        pd.Series: Una serie de Pandas con el valor del WBGT calculado.
    """
    # Filtrar fechas
    df_filtered = _filter_df_by_date(df_data, fecha_inicio, fecha_fin)
    
    if df_filtered.empty:
        print("ADVERTENCIA: El DataFrame filtrado por fechas está vacío. Retornando serie vacía.")
        return pd.Series([], dtype=float, name='WGBT')
    
    # Asignar variables locales para legibilidad (Ta y RH)
    Ta = df_filtered[col_temp]
    RH = df_filtered[col_humidity]

    # Calcular Tw
    Tw = calculate_wet_bulb_temp_stull(Ta, RH)

    # Calcular la WBGT
    wbgt_final = 0.67 * Tw + 0.33 * Ta

    # Devolver como una Serie de Pandas con el filtro de fechas
    return pd.Series(wbgt_final, index=df_filtered.index, name='WBGT')


def calculate_he(df_data, col_temp_int, col_temp_ext,
                 building_category='III', model_type='freeruning',
                 inicio_horas=7, fin_horas=23, data_frequency_min=60,
                 fecha_inicio=None, fecha_fin=None):
    """
    Calcula el Heat Exceedance (HE), el número de horas en que la temperatura
    interior excede un umbral de confort durante el período de ocupación.
    """

    df = df_data.copy()
    # Aseguramos que 'df_data' tenga una columna 'datetime'
    if 'datetime' not in df.columns:
        raise ValueError("El DataFrame de entrada debe tener una columna llamada 'datetime'.")
    df['datetime'] = pd.to_datetime(df['datetime']) # Asegurar tipo datetime

    # --- NUEVO: FILTRADO POR RANGO DE FECHAS ---
    if fecha_inicio:
        df = df[df['datetime'].dt.date >= pd.to_datetime(fecha_inicio).date()]
    if fecha_fin:
        df = df[df['datetime'].dt.date <= pd.to_datetime(fecha_fin).date()]
    # --- FIN FILTRADO ---
    
    if df.empty:
        return pd.DataFrame({'Fecha': [], 'HE_Diario': [], 'Horas_Ocupadas': []})

    # --- 1. CONFIGURACIÓN DE PARÁMETROS ---
    model_type = model_type.lower()
    time_interval_h = data_frequency_min / 60.0 # Factor de ponderación por tiempo

    # Definir X (umbral de superación)
    X = 1.0 if model_type == 'freeruning' else 0.0

    # --- 2. CÁLCULO DEL UMBRAL DE CONFORT (T_op,i,upper) ---

    df['Tint'] = df[col_temp_int]
    df['Text'] = df[col_temp_ext]

    if model_type == 'freeruning':
        # Calcular Trm para el modelo adaptativo (misma lógica que IOD)
        df['Trm'] = calculate_trm(df, col_temp_ext, data_frequency_min)

        # Usar la función centralizada para calcular el T_upper (Tconf)
        df['T_upper'] = calculate_tconf(df['Trm'], building_category, model_type)


    elif model_type == 'airconditioned':
        # Creamos una serie temporal para que calculate_tconf pueda calcular el valor estático
        # Pasamos un Trm_fake, ya que Tconf estático no lo necesita, pero la función lo requiere como argumento
        trm_fake = pd.Series(np.nan, index=df.index) 
        df['T_upper'] = calculate_tconf(trm_fake, building_category, model_type)

        
    else:
        raise ValueError("Tipo de modelo no válido. Use 'freeruning' o 'airconditioned'.")


    # --- 3. CÁLCULO DE LA FUNCIÓN DE PESO (Wf,i) ---

    # Condición de superación: T_op,i - T_op,i,upper >= X
    df['Supera'] = (df.Tint - df.T_upper) >= X

    # Wf,i: 1 si supera, 0 si no. Multiplicado por el factor de tiempo (horas)
    df['Wf'] = np.where(df.Supera, 1 * time_interval_h, 0)


    # --- 4. FILTRADO Y AGREGACIÓN ---

    # Filtrar solo el período de ocupación
    df_ocupacion = df[
        (df.datetime.dt.hour >= inicio_horas) &
        (df.datetime.dt.hour < fin_horas)
    ].copy()

    # Asegurar que no haya NaNs en el cálculo principal (p. ej. en Tint o T_upper)
    df_ocupacion.dropna(subset=['Wf'], inplace=True)

    # Agrupar por día
    df_ocupacion['Fecha'] = df_ocupacion['datetime'].dt.date

    df_agregado = df_ocupacion.groupby('Fecha').agg(
        HE_Diario=('Wf', 'sum'),
        Conteo_Intervalos=('Wf', 'size')
    ).reset_index()

    # Calcular el número total de horas ocupadas monitoreadas ese día
    df_agregado['Horas_Ocupadas'] = df_agregado['Conteo_Intervalos'] * time_interval_h

    # Devolver el DataFrame diario
    return df_agregado[['Fecha', 'HE_Diario', 'Horas_Ocupadas']]

def calculate_iod_reg(df_daily_data, col_iod='IOD_Diario', col_awd='AWD_Diario', 
                      col_solar_h='H_DIARIO', lag_days=2):
    """
    Calcula la regresión lineal múltiple para IOD_i = f(AWD_i, H_i, IOD_i-1, IOD_i-2...).
    
    El DataFrame de entrada (df_daily_data) DEBE estar a nivel diario y contener
    IOD, AWD y la Irradiancia Solar Horizontal (H).
    
    Args:
        df_daily_data (pd.DataFrame): DataFrame con IOD, AWD y H. 
        col_iod (str): Nombre de la columna de IOD Diario.
        col_awd (str): Nombre de la columna de AWD Diario.
        col_solar_h (str): Nombre de la columna de Irradiancia Solar Horizontal Diaria.
        lag_days (int): Número de días anteriores de IOD a incluir (e.g., 2 para i-1, i-2).

    Returns:
        tuple: (df_results, dict_coefficients)
            - df_results (pd.DataFrame): DataFrame con Fecha, IODi, IOD Predicho.
            - dict_coefficients (dict): Diccionario con los coeficientes significativos (alpha, beta, gamma, etc.)
    """
    print("\n" + "="*50)
    print(f"Regresión IOD: Modelando IOD con {lag_days} días de historia.")

    df_reg = df_daily_data.copy()
    
    # 1. Creación de Variables de Lag
    X_cols = [col_awd, col_solar_h]
    for lag in range(1, lag_days + 1):
        lag_col_name = f'IOD_i-{lag}'
        df_reg[lag_col_name] = df_reg[col_iod].shift(lag)
        X_cols.append(lag_col_name)

    # 2. Limpieza (Eliminar NaNs de las filas generadas por el shift)
    df_reg.dropna(subset=X_cols + [col_iod], inplace=True)
    
    if df_reg.empty:
        print("ERROR: DataFrame vacío después de crear variables de lag. Asegura que tienes suficientes días de datos.")
        return pd.DataFrame(), {}
    
    Y = df_reg[col_iod] # IOD i
    X = df_reg[X_cols] # Predictoras

    # 3. Estandarización de las Predictoras - hacemos esto para que los coeficientes de regresión se vuelven directamente comparables
    scaler = StandardScaler()  # herramienta que calcula la media y la desviación estándar
    X_scaled = scaler.fit_transform(X)   #fit --> calcula la media y la desviación estándar. transform --> aplica la fórmula de estandarización a cada valor x de la matriz X
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_cols, index=X.index) #x_scaled es un objeto de numpy entonces o tengo que hacer de pandas de vuelta
    X_scaled_df = sm.add_constant(X_scaled_df) # agrega cte a0 como una columna de 1 que representa al termino independiente en la regresion

    # 4. Ejecución de la Regresión OLS (Ordinary Least Squares)
    try:
        modelo = sm.OLS(Y, X_scaled_df).fit()
        print("Regresión OLS ajustada con éxito.")
    except Exception as e:
        print(f"ERROR al ajustar el modelo OLS: {e}")
        return pd.DataFrame(), {}

    # 5. Extracción de Resultados y P-Values
    
    # 5.1. Coeficientes Estandarizados y P-values
    coef_df = modelo.summary2().tables[1] # Obtener la tabla de coeficientes
    
    # 5.2. Filtrar variables significativas (P>|t| < 0.05)
    significant_coefs = coef_df[coef_df['P>|t|'] < 0.05]
    
    # 5.3. Imprimir resultados significativos
    if not significant_coefs.empty:
        print("\n Coeficientes Estadísticamente Significativos (P < 0.05):")
        # Imprimimos solo las columnas más relevantes de la tabla
        print(significant_coefs[['Coef.', 'P>|t|']])
    else:
        print("\n ADVERTENCIA: Ningún predictor fue estadísticamente significativo (P > 0.05).")
    
    # 5.4. Mapear y guardar coeficientes (estandarizados)
    dict_coefficients = {}

    # El coeficiente para 'Solar_Irradiance_H' es beta
    beta_val = modelo.params.get(col_solar_h, np.nan) 
    p_beta = modelo.pvalues.get(col_solar_h, 1.0)
    
    # El coeficiente para IOD_i-1 es gamma (si existe)
    gamma_val = modelo.params.get('IOD_i-1', np.nan)
    p_gamma = modelo.pvalues.get('IOD_i-1', 1.0)
    
    dict_coefficients = {
        'a0': modelo.params['const'],
        'alpha_IOD': modelo.params.get(col_awd, np.nan),
        'beta_IOD': beta_val,
        'gamma_IOD': gamma_val,
        'R_Squared': modelo.rsquared,
        'P_value_beta': p_beta,
        'P_value_gamma': p_gamma
    }
    
    print(f"R-Squared del modelo: {modelo.rsquared:.4f}")
    
    # 6. Cálculo de IOD Predicho
    IOD_predicho = modelo.predict(X_scaled_df)
    
    # 7. Construir DataFrame de Resultados
    df_results = pd.DataFrame({
        'Fecha': df_reg['Fecha'],
        'IODi_Diario': Y,   
        'IOD_Diario_Predicho': IOD_predicho
    }).reset_index(drop=True)
    
    return df_results, dict_coefficients


def calculate_score_amaripadath(iod_average_period, df_hi_aligned, df_wbgt_aligned, df_he_aligned):
    """
    Calcula el Score de Resiliencia a Olas de Calor.
    
    ASUME que los DataFrames de entrada ya están:
    1. FILTRADOS por horas de ocupación.
    2. ALINEADOS (mismo índice, sin datos faltantes).
    3. Tienen las columnas 'HI', 'WBGT', y 'HE'.
    
    Args:
        iod_average_period (float): Valor promedio del IOD para todo el periodo.
        df_hi_aligned (pd.DataFrame): DataFrame ALINEADO Y FILTRADO con la columna 'HI'.
        df_wbgt_aligned (pd.DataFrame): DataFrame ALINEADO Y FILTRADO con la columna 'WBGT'.
        df_he_aligned (pd.DataFrame): DataFrame ALINEADO Y FILTRADO con la columna 'HE'.
        
    Returns:
        float: El Score de Resiliencia a Olas de Calor.
    """
    
    print("\n" + "="*50)
    print("Calculando Score de Resiliencia a Olas de Calor de Amaripadath...")

    # --- 1. Definición de Umbrales ---
    IOD_THRESHOLD_MODERATE = 0.5
    IOD_THRESHOLD_STRONG = 2.0
    THRESHOLD_HE = 27.0   
    THRESHOLD_WBGT = 28.0 
    THRESHOLD_HI = 26.7   
    
    # --- 2. Cálculo del Factor de Ponderación IOD_W ---
    iod_w = 0.0
    if iod_average_period <= IOD_THRESHOLD_MODERATE:
        iod_w = 1.0
        category = "Moderate"
    elif iod_average_period < IOD_THRESHOLD_STRONG:
        iod_w = 0.5
        category = "Strong"
    else:
        iod_w = 0.0
        category = "Extreme"
        
    print(f"  - IOD Promedio General: {iod_average_period:.3f}. Categoría IOD_W: {category} ({iod_w})")

    # --- 3. Determinación del Total de Horas Válidas ---
    # La función confía en que los DFs ya están alineados.
    total_hours = len(df_hi_aligned) 
    
    if total_hours == 0:
        print("ERROR: Los DataFrames alineados están vacíos.")
        return 0.0

    print(f"Total de horas de ocupación alineadas para análisis: {total_hours}")
    
    # --- 4. Cálculo de Cumplimiento (%) para HE, WBGT, HI ---
    
    # 1. Hours of Exceedence (HE)
    he_compliant_hours = len(df_he_aligned[df_he_aligned['HE'] < THRESHOLD_HE])
    HE_hours_percent = (he_compliant_hours / total_hours) * 100

    # 2. WBGT
    wbgt_compliant_hours = len(df_wbgt_aligned[df_wbgt_aligned['WBGT'] < THRESHOLD_WBGT])
    WBGT_hours_percent = (wbgt_compliant_hours / total_hours) * 100

    # 3. Heat Index (HI)
    hi_compliant_hours = len(df_hi_aligned[df_hi_aligned['HI'] < THRESHOLD_HI])
    HI_hours_percent = (hi_compliant_hours / total_hours) * 100
    
    # --- 5. Cálculo del Score Final ---
    
    compliance_sum_percent = HE_hours_percent + WBGT_hours_percent + HI_hours_percent
    final_score = iod_w + (compliance_sum_percent / 100.0)

    print(f"\n Score de Resiliencia Final: {final_score:.4f}")
    print("="*50)
    
    return final_score