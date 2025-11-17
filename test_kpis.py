# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:57:11 2025

@author: cami_
"""

# -*- coding: utf-8 -*-
"""
Script de prueba para el cálculo de KPIs de confort térmico.
"""
import pandas as pd
import numpy as np
import os
import KPIs

# ----------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------

# Ruta a tu archivo de datos unificado
ARCHIVO_DATOS_UNIFICADO = 'datos_kpis_unificado.xlsx' 

# Nombres de las columnas exteriores (AJUSTAR SI ES NECESARIO)
COL_TEMP_EXT = 'TEMP OUT (°C)' 
COL_HUM_EXT = 'OUT HUM (%)' 
COL_SOLAR_RAD = 'Solar Rad (W/m2) CALIBRADA'

# Carpeta donde se guardarán los resultados
CARPETA_SALIDA = 'Resultados_KPIs' 

# --- PARÁMETROS DE CÁLCULO ---
# Frecuencia de los datos (60 para horario, 15 para 15 minutos)
DATA_FREQUENCY_MIN = 60
TIME_INTERVAL_H = DATA_FREQUENCY_MIN / 60.0 # 1.0 para horario, 0.25 para 15 min

# --- PARÁMETROS DE FILTRADO POR FECHA ---
# Define el rango de fechas para el análisis (Formato 'YYYY-MM-DD'). 
# Deja en None para usar todos los datos.
FECHA_INICIO_ANALISIS = None # Ej: '2024-10-01'
FECHA_FIN_ANALISIS = None    # Ej: '2024-12-31'

# ----------------------------------------------------------------
# FUNCIÓN DE CARGA Y PREPARACIÓN
# ----------------------------------------------------------------

def load_and_prepare_data():
    """Carga el DataFrame unificado y realiza la limpieza inicial."""
    try:
        df_full = pd.read_excel(ARCHIVO_DATOS_UNIFICADO) 
        print(f"✅ DataFrame unificado cargado con {len(df_full)} registros.")

    except FileNotFoundError:
        print(f"❌ ERROR: Archivo '{ARCHIVO_DATOS_UNIFICADO}' no encontrado. Asegúrate de que exista.")
        return None
    except Exception as e:
        print(f"❌ ERROR al leer el archivo unificado: {e}")
        return None

    # Renombrar la columna de tiempo a 'datetime' si fuera necesario
    if df_full.columns[0] != 'datetime':
        df_full.rename(columns={df_full.columns[0]: 'datetime'}, inplace=True)
        
    # Validar columnas críticas
    required_cols = ['datetime', COL_TEMP_EXT, COL_HUM_EXT]
    for col in required_cols:
        if col not in df_full.columns:
            print(f"❌ ERROR: Columna requerida '{col}' no encontrada en el DataFrame.")
            return None
        
    # Asegurar el formato datetime y limpiar nulos en columnas críticas
    df_full['datetime'] = pd.to_datetime(df_full['datetime'], errors='coerce')
    # Solo limpiamos NaNs de las columnas exteriores críticas
    df_full.dropna(subset=['datetime', COL_TEMP_EXT, COL_HUM_EXT], inplace=True)

    return df_full

# ----------------------------------------------------------------
# FUNCIÓN AUXILIAR DE FILTRADO DE DATOS POR FECHA
# ----------------------------------------------------------------

def filter_df_by_date(df, fecha_inicio, fecha_fin):
    """Aplica el filtro de fechas al DataFrame."""
    df_filtered = df.copy()
    
    if fecha_inicio is None and fecha_fin is None:
        print("    (No se aplicó filtro de fechas, usando el DataFrame completo.)")
        return df_filtered
        
    print(f"    Filtrando datos entre {fecha_inicio or 'Inicio'} y {fecha_fin or 'Fin'}...")
    
    if fecha_inicio:
        inicio = pd.to_datetime(fecha_inicio).date()
        df_filtered = df_filtered[df_filtered['datetime'].dt.date >= inicio]
    if fecha_fin:
        fin = pd.to_datetime(fecha_fin).date()
        df_filtered = df_filtered[df_filtered['datetime'].dt.date <= fin]
    
    if not df_filtered.empty:
        print(f"    Filtrado: {len(df_filtered)} registros restantes.")
    else:
        print("    ❌ ADVERTENCIA: El DataFrame filtrado por fechas está vacío.")
        
    return df_filtered

# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - HEAT INDEX (HI) INTERIOR
# ----------------------------------------------------------------

def run_hi_interior_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """
    Identifica pares (Temp, HR) interiores, calcula el HI para cada par 
    y consolida los resultados en un único archivo.
    """
    print("\n" + "="*50)
    print("🔍 Calculando Heat Index (HI) para Zonas Interiores...")
    
    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return

    df_hi_consolidado = df_filtered[['datetime']].copy()
    columnas_disponibles = df_filtered.columns.tolist()
    zonas_procesadas = 0

    prefixes = sorted(list(set([
        col.replace('Temp', '').replace('HR', '').strip()
        for col in columnas_disponibles if 'Temp' in col or 'HR' in col
    ])))

    for prefix in prefixes:
        col_temp = f'{prefix}Temp'
        col_hr = f'{prefix}HR'
        nombre_hi = f'{prefix}_HI'
        
        if col_temp in columnas_disponibles and col_hr in columnas_disponibles:
            df_subset = df_filtered[['datetime', col_temp, col_hr]].copy()
            df_subset.dropna(subset=[col_temp, col_hr], inplace=True) 

            if not df_subset.empty:
                try:
                    # --- MODIFICACIÓN: Pasando los argumentos de fecha ---
                    hi_series = KPIs.calculate_heat_index(
                        df_data=df_subset, 
                        col_temp=col_temp, 
                        col_humidity=col_hr,
                        fecha_inicio=fecha_inicio, # La librería (metrics.py) también filtra
                        fecha_fin=fecha_fin
                    )
                    
                    df_hi_consolidado[nombre_hi] = hi_series.reindex(df_filtered.index)
                    zonas_procesadas += 1
                    # print(f"    ✅ HI calculado para la zona: {prefix}") # Descomentar para debug
                
                except Exception as e:
                    print(f"    ❌ ERROR al calcular HI para {prefix}: {e}")
        
    if zonas_procesadas > 0:
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'HeatIndex_Interiores_Consolidado.xlsx')
        df_hi_consolidado.to_excel(nombre_archivo_salida, index=False)
        print(f"\n✅ HI de {zonas_procesadas} zonas guardado en '{nombre_archivo_salida}'.")
    else:
        print("\n❌ ADVERTENCIA: No se encontraron pares válidos (Temp/HR) para calcular el HI interior.")


# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - IOD (Índice de Disconfort por Ocupación)
# ----------------------------------------------------------------

def run_iod_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """Calcula el IOD para todas las zonas y consolida en UN archivo."""

    print(f"\n" + "="*50)
    print(f"🔍 IOD: Iniciando cálculo para datos con frecuencia de {DATA_FREQUENCY_MIN} min.")
    
    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return

    temp_interior_cols = [
        col for col in df_filtered.columns 
        if col.endswith('Temp') and col != COL_TEMP_EXT
    ]
    
    if not temp_interior_cols:
        print("❌ ADVERTENCIA: No se encontraron columnas de T Interior ('*Temp') para IOD.")
        return
        
    print(f"    Encontradas {len(temp_interior_cols)} zonas interiores para IOD.")
    
    iod_results_list = [] 

    for col_int in temp_interior_cols:
        df_subset = df_filtered[['datetime', col_int, COL_TEMP_EXT]].copy()
        
        try:
            # Llamada a la librería (que también filtra por fecha)
            df_resultado_iod = KPIs.calculate_iod(
                df_data=df_subset, 
                col_temp_int=col_int, 
                col_temp_ext=COL_TEMP_EXT,
                inicio_horas=7, fin_horas=23,
                group_by_day=True, 
                data_frequency_min=DATA_FREQUENCY_MIN,
                fecha_inicio=fecha_inicio, # Se pasa el argumento
                fecha_fin=fecha_fin      # Se pasa el argumento
            )

            nombre_zona = col_int.replace(' ', '_').replace('(', '').replace(')', '').replace('Temp', '')
            df_resultado_iod['Zona'] = nombre_zona
            
            iod_results_list.append(df_resultado_iod)
            # print(f"    ✅ IOD Diario (en lista) para: {col_int}") # Descomentar para debug

        except Exception as e:
            print(f"    ❌ ERROR al calcular IOD para {col_int}: {e}")

    if iod_results_list:
        df_iod_consolidado = pd.concat(iod_results_list, ignore_index=True)
        
        df_iod_pivot = df_iod_consolidado.pivot(
            index='Fecha', 
            columns='Zona', 
            values='IOD_Diario'
        ).reset_index()
        
        df_iod_pivot.columns.name = None
        
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'IOD_Diario_Consolidado.xlsx')
        df_iod_pivot.to_excel(nombre_archivo_salida, index=False)
        
        print(f"\n✅ IOD de {len(iod_results_list)} zonas consolidado en '{nombre_archivo_salida}'.")
    else:
        print("\n❌ ADVERTENCIA: No se calcularon resultados de IOD.")


# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - AWD
# ----------------------------------------------------------------

def run_awd_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """Calcula y exporta el AWD diario y total."""
    
    print("\n" + "="*50)
    print(f"🔍 Calculando AWD (Grados-Día Acumulados) para datos con intervalo de {TIME_INTERVAL_H}h...")
    
    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return
    
    # 1. CÁLCULO DIARIO (AWD por día)
    try:
        # --- MODIFICACIÓN: Pasando los argumentos de fecha ---
        df_awd_diario = KPIs.calculate_awd(
            df_data=df_filtered.copy(), # Usamos el DF filtrado 
            col_temp_ext=COL_TEMP_EXT, 
            temp_base=18,
            group_by_day=True,
            time_interval_h=TIME_INTERVAL_H,
            fecha_inicio=fecha_inicio, # Y también pasamos las fechas a la librería
            fecha_fin=fecha_fin
        )
        
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'AWD_Diario.xlsx')
        df_awd_diario.to_excel(nombre_archivo_salida, index=False)
        
        print(f"    ✅ AWD Diario guardado en '{nombre_archivo_salida}'.")
        
    except Exception as e:
        print(f"    ❌ ERROR al calcular AWD Diario: {e}")
        
    # 2. CÁLCULO TOTAL (AWD para todo el período)
    try:
        # --- MODIFICACIÓN: Pasando los argumentos de fecha ---
        awd_total = KPIs.calculate_awd(
            df_data=df_filtered.copy(), 
            col_temp_ext=COL_TEMP_EXT, 
            temp_base=18, 
            group_by_day=False,
            time_interval_h=TIME_INTERVAL_H,
            fecha_inicio=fecha_inicio, # Y también pasamos las fechas a la librería
            fecha_fin=fecha_fin
        )
        
        print(f"    ✅ AWD Total (Base 18°C) para el período completo: {awd_total:.4f}")
        
    except Exception as e:
        print(f"    ❌ ERROR al calcular AWD Total: {e}")

# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - ALPHA IOD
# ----------------------------------------------------------------      
        
def run_alpha_iod_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """
    Calcula el Factor de Escalada (alpha_IOD) para cada zona interior
    y consolida los resultados en un único archivo.
    """
    print("\n" + "="*50)
    print(f"🔍 Alpha_IOD: Iniciando cálculo para datos con frecuencia de {DATA_FREQUENCY_MIN} min.")
    
    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return
    
    temp_interior_cols = [
        col for col in df_filtered.columns 
        if col.endswith('Temp') and col != COL_TEMP_EXT
    ]
    
    if not temp_interior_cols:
        print("❌ ADVERTENCIA: No se encontraron columnas 'Temp' interiores para Alpha_IOD.")
        return
        
    print(f"    Encontradas {len(temp_interior_cols)} zonas interiores para Alpha_IOD.")
    
    df_alpha_consolidado = None
    
    for col_int in temp_interior_cols:
        try:
            # Llamada a la librería (que también filtra por fecha)
            df_alpha_diario_zona = KPIs.calculate_alpha_iod(
                df_data=df_filtered.copy(), # Usar el DF filtrado
                col_temp_int=col_int,
                col_temp_ext=COL_TEMP_EXT,
                inicio_horas=7, 
                fin_horas=23,
                temp_base_awd=18,
                group_by_day=True, 
                data_frequency_min=DATA_FREQUENCY_MIN,
                fecha_inicio=fecha_inicio, # Se pasa el argumento
                fecha_fin=fecha_fin      # Se pasa el argumento
            )
            
            nombre_zona = col_int.replace(' ', '_').replace('(', '').replace(')', '').replace('Temp', '')
            df_alpha_diario_zona.rename(columns={
                'IOD_Diario': f'{nombre_zona}_IOD',
                'alpha_IOD': f'{nombre_zona}_alpha'
            }, inplace=True)
            
            if df_alpha_consolidado is None:
                df_alpha_consolidado = df_alpha_diario_zona
            else:
                df_alpha_consolidado = pd.merge(
                    df_alpha_consolidado, 
                    df_alpha_diario_zona[['Fecha', f'{nombre_zona}_IOD', f'{nombre_zona}_alpha']],
                    on='Fecha',
                    how='outer'
                )
            
            # print(f"    ✅ Alpha_IOD Diario (en lista) para: {col_int}") # Descomentar para debug

        except Exception as e:
            print(f"    ❌ ERROR al calcular Alpha_IOD para {col_int}: {e}")
            
    if df_alpha_consolidado is not None:
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'Alpha_IOD_Diario_Consolidado.xlsx')
        df_alpha_consolidado.to_excel(nombre_archivo_salida, index=False)
        print(f"\n✅ Alpha_IOD de {len(temp_interior_cols)} zonas consolidado en '{nombre_archivo_salida}'.")
    else:
        print("\n❌ ADVERTENCIA: No se calcularon resultados de Alpha_IOD.")      

# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - WBGT
# ----------------------------------------------------------------

def run_wbgt_interior_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """
    Identifica pares (Temp, HR) interiores, calcula el WBGT para cada par
    y consolida los resultados en un único archivo.
    """
    print("\n" + "="*50)
    print("🔍 Calculando Wet-Bulb Globe Temperature (WBGT) para Zonas Interiores...")
    
    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return

    df_wbgt_consolidado = df_filtered[['datetime']].copy()
    columnas_disponibles = df_filtered.columns.tolist()
    zonas_procesadas = 0

    prefixes = sorted(list(set([
        col.replace('Temp', '').replace('HR', '').strip()
        for col in columnas_disponibles if 'Temp' in col or 'HR' in col
    ])))

    for prefix in prefixes:
        col_temp = f'{prefix}Temp'
        col_hr = f'{prefix}HR'
        nombre_wbgt = f'{prefix}_WBGT'
        
        if col_temp in columnas_disponibles and col_hr in columnas_disponibles:
            df_subset = df_filtered[['datetime', col_temp, col_hr]].copy()
            df_subset.dropna(subset=[col_temp, col_hr], inplace=True) 

            if not df_subset.empty:
                try:
                    # --- MODIFICACIÓN: Pasando los argumentos de fecha ---
                    # (Asegúrate de que calculate_wbgt en metrics.py acepte estas fechas)
                    wbgt_series = KPIs.calculate_wbgt(
                        df_data=df_subset, 
                        col_temp=col_temp, 
                        col_humidity=col_hr,
                        fecha_inicio=fecha_inicio, # La librería (metrics.py) también filtra
                        fecha_fin=fecha_fin
                    )
                    
                    df_wbgt_consolidado[nombre_wbgt] = wbgt_series.reindex(df_filtered.index)
                    zonas_procesadas += 1
                    # print(f"    ✅ WBGT calculado para la zona: {prefix}") # Descomentar para debug
                
                except Exception as e:
                    print(f"    ❌ ERROR al calcular WBGT para {prefix}: {e}")
        
    if zonas_procesadas > 0:
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'WBGT_Interiores_Consolidado.xlsx')
        df_wbgt_consolidado.to_excel(nombre_archivo_salida, index=False)
        print(f"\n✅ WBGT de {zonas_procesadas} zonas guardado en '{nombre_archivo_salida}'.")
    else:
        print("\n❌ ADVERTENCIA: No se encontraron pares válidos (Temp/HR) para calcular el WBGT interior.")


# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - HEAT EXCEEDANCE (HE)
# ----------------------------------------------------------------

def run_he_analysis(df_full, category='III', model='freeruning', inicio=7, fin=23, fecha_inicio=None, fecha_fin=None):
    """Calcula el Heat Exceedance (HE) para todas las zonas y consolida."""

    print("\n" + "="*50)
    print(f"🔍 HE: Calculando Excedencia de Calor (Cat: {category}, Modelo: {model.capitalize()})...")

    # Filtrar el DataFrame al inicio de la función
    df_filtered = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered.empty: return

    temp_interior_cols = [
        col for col in df_filtered.columns
        if col.endswith('Temp') and col != COL_TEMP_EXT
    ]

    if not temp_interior_cols:
        print("❌ ADVERTENCIA: No se encontraron columnas de T Interior ('*Temp') para HE.")
        return

    print(f"    Encontradas {len(temp_interior_cols)} zonas interiores para HE.")

    he_results_list = []
    zonas_procesadas = 0

    for col_int in temp_interior_cols:
        df_subset = df_filtered[['datetime', col_int, COL_TEMP_EXT]].copy()

        try:
            # Llamada a la librería (que también filtra por fecha)
            df_resultado_he = KPIs.calculate_he(
                df_data=df_subset,
                col_temp_int=col_int,
                col_temp_ext=COL_TEMP_EXT,
                building_category=category,
                model_type=model,
                inicio_horas=inicio, fin_horas=fin,
                data_frequency_min=DATA_FREQUENCY_MIN,
                fecha_inicio=fecha_inicio, # Se pasa el argumento
                fecha_fin=fecha_fin      # Se pasa el argumento
            )

            nombre_zona = col_int.replace(' ', '_').replace('(', '').replace(')', '').replace('Temp', '')
            df_resultado_he['Zona'] = nombre_zona
            df_resultado_he.rename(columns={'HE_Diario': f'HE_{nombre_zona}_h'}, inplace=True)
            
            he_results_list.append(df_resultado_he[['Fecha', f'HE_{nombre_zona}_h']])
            zonas_procesadas += 1
            # print(f"    ✅ HE Diario (en lista) para: {col_int}") # Descomentar para debug

        except Exception as e:
            print(f"    ❌ ERROR al calcular HE para {col_int}: {e}")

    if he_results_list:
        df_he_consolidado = he_results_list[0]
        
        for i in range(1, len(he_results_list)):
            df_he_consolidado = pd.merge(df_he_consolidado, he_results_list[i], on='Fecha', how='outer')

        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, f'HE_Diario_Consolidado_{model.capitalize()}_Cat{category}.xlsx')
        df_he_consolidado.to_excel(nombre_archivo_salida, index=False)

        print(f"\n✅ HE de {zonas_procesadas} zonas consolidado en '{nombre_archivo_salida}'.")
    else:
        print("\n❌ ADVERTENCIA: No se calcularon resultados de HE.")
        
        
# ----------------------------------------------------------------
# FUNCIÓN PRINCIPAL - REGRESIÓN IOD (IOD_REG)
# ----------------------------------------------------------------

def run_iod_regression_analysis(df_full, fecha_inicio=None, fecha_fin=None):
    """
    Prepara los datos, ejecuta la Regresión Múltiple para IOD iterando sobre 
    todas las zonas interiores y exporta los coeficientes clave.
    """
    
    print("\n" + "="*50)
    print("🔬 Regresión IOD: Ejecutando modelo predictivo por zonas...")
    
    # 1. Filtrar y preparar el DataFrame de datos horarios
    df_filtered_hourly = filter_df_by_date(df_full, fecha_inicio, fecha_fin)
    if df_filtered_hourly.empty: return

    # Obtener todas las columnas de temperatura interior
    temp_interior_cols = [
        col for col in df_filtered_hourly.columns 
        if col.endswith('Temp') and col != COL_TEMP_EXT
    ]
    
    if not temp_interior_cols:
        print("❌ ADVERTENCIA: No se encontraron columnas 'Temp' interiores. Finalizando análisis de regresión.")
        return
    
    # --- CÁLCULO DE AWD Y SOLAR DIARIO (Solo se calcula una vez) ---
    # B. AWD Diario (CORRECCIÓN APLICADA: argumentos completos)
    df_awd_diario = KPIs.calculate_awd(
        df_data=df_filtered_hourly[['datetime', COL_TEMP_EXT]].copy(),
        col_temp_ext=COL_TEMP_EXT,
        temp_base=18,
        group_by_day=True,
        time_interval_h=TIME_INTERVAL_H,
    ).rename(columns={'AWD_Diario': 'AWD_Diario'})
    
    # C. Irradiancia Solar Diaria
    df_solar_diario = df_filtered_hourly.groupby(
        df_filtered_hourly['datetime'].dt.date
    )[COL_SOLAR_RAD].sum().reset_index()
    df_solar_diario.rename(columns={'datetime': 'Fecha', COL_SOLAR_RAD: 'H_DIARIO'}, inplace=True)
    
    # Lista para almacenar los resultados de coeficientes de todas las zonas
    all_coefficients = []
    
    print(f"\nSe encontraron {len(temp_interior_cols)} zonas para analizar.")
    
    # --- ITERACIÓN SOBRE CADA ZONA ---
    for col_int_reg in temp_interior_cols:
        
        zone_name = col_int_reg.replace('Temp', '')
        print(f"\n--- 🌡️ Analizando Regresión IOD para la zona: {zone_name} ---")
        
        try:
            # A. IOD Diario de la zona actual
            df_iod_diario = KPIs.calculate_iod(
                df_data=df_filtered_hourly[['datetime', col_int_reg, COL_TEMP_EXT]].copy(), 
                col_temp_int=col_int_reg, 
                col_temp_ext=COL_TEMP_EXT,
                inicio_horas=7, fin_horas=23, group_by_day=True, 
                data_frequency_min=DATA_FREQUENCY_MIN
            ).rename(columns={'IOD_Diario': 'IOD_Diario'})
            
            # 1. Consolidar el DataFrame Diario (IOD de la zona actual + AWD + Solar)
            df_daily_reg = pd.merge(df_iod_diario, df_awd_diario, on='Fecha', how='inner')
            df_daily_reg = pd.merge(df_daily_reg, df_solar_diario, on='Fecha', how='inner')
            
            # 2. LLAMADA A LA FUNCIÓN DE REGRESIÓN
            df_prediccion, coeficientes = KPIs.calculate_iod_reg(
                df_daily_data=df_daily_reg,
                col_iod='IOD_Diario',
                col_awd='AWD_Diario',
                col_solar_h='H_DIARIO',
                lag_days=2
            )
            
            # 3. Guardar y reportar los resultados
            if not df_prediccion.empty:
                
                # Añadir el nombre de la zona a los coeficientes
                coeficientes['Zona'] = zone_name
                all_coefficients.append(coeficientes)
                
                # Exportar la predicción (opcionalmente)
                nombre_archivo_salida = os.path.join(CARPETA_SALIDA, f'IOD_Regresion_Prediccion_{zone_name}.xlsx')
                df_prediccion.to_excel(nombre_archivo_salida, index=False)
                
                R2 = coeficientes.get('R_Squared', np.nan)
                beta = coeficientes.get('beta_SolarH', np.nan)
                p_beta = coeficientes.get('P_value_beta', 1.0)
                gamma = coeficientes.get('gamma_IOD_i-1', np.nan)
                
                print(f"    R-Squared: {R2:.4f}")
                print(f"    Alpha (AWD): {coeficientes.get('alpha_AWD', np.nan):.4f}")
                print(f"    Beta (Solar H): {beta:.4f} (P-value: {p_beta:.2f})")
                print(f"    Gamma (IOD i-1): {gamma:.4f}")
        
        except Exception as e:
            # El bloque except debe estar al mismo nivel que el try
            print(f"    ❌ ERROR al procesar la zona {zone_name}: {e}")
            
    # --- EXPORTAR RESULTADOS FINALES DE COEFICIENTES ---
    # Este bloque debe estar FUERA del bucle 'for'
    if all_coefficients:
        df_coef_final = pd.DataFrame(all_coefficients)
        
        # Reordenar las columnas clave para mejor visualización
        col_order = ['Zona', 'R_Squared', 'alpha_AWD', 'beta_SolarH', 'P_value_beta', 'gamma_IOD_i-1', 'P_value_gamma', 'a0_Intercepto']
        df_coef_final = df_coef_final.reindex(columns=col_order, fill_value=np.nan)
        
        nombre_coef_salida = os.path.join(CARPETA_SALIDA, 'IOD_Regresion_Coeficientes_RESUMEN.xlsx')
        df_coef_final.to_excel(nombre_coef_salida, index=False)
        print(f"\n✅ Resumen de coeficientes para todas las zonas guardado en '{nombre_coef_salida}'.")

# ----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------

if __name__ == '__main__':
    df_data = load_and_prepare_data()
    
    if df_data is not None:
        if not os.path.exists(CARPETA_SALIDA):
            os.makedirs(CARPETA_SALIDA)
            
        print(f"\n--- INICIANDO CÁLCULO DE KPIs ---")
        if FECHA_INICIO_ANALISIS or FECHA_FIN_ANALISIS:
            print(f"Rango de análisis: {FECHA_INICIO_ANALISIS or 'Inicio'} hasta {FECHA_FIN_ANALISIS or 'Fin'}")
        else:
            print("Rango de análisis: DataFrame completo.")
            
        # 1. Ejecutar análisis de HI Interior
        #run_hi_interior_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 2. Ejecutar análisis de IOD Diario
        #run_iod_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 3. Ejecutar análisis de AWD
        #run_awd_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 4. Ejecutar análisis de Alpha_IOD
        #run_alpha_iod_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 5. Ejecutar análisis de WBGT Interior
        #run_wbgt_interior_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 6. Ejecutar análisis de Heat Exceedance (HE)
        #run_he_analysis(df_data.copy(), category='III', model='freeruning', inicio=7, fin=23, fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        # 7. Ejecutar análisis de Regresión IOD (¡NUEVO!)
        run_iod_regression_analysis(df_data.copy(), fecha_inicio=FECHA_INICIO_ANALISIS, fecha_fin=FECHA_FIN_ANALISIS)
        
        print("\n--- CÁLCULO DE KPIs FINALIZADO ---")