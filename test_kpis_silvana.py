import pandas as pd
import numpy as np
import os
# Importamos el módulo completo para acceder a todas las funciones
import KPIs 

# ----------------------------------------------------------------
# CONFIGURACIÓN DEL PROYECTO
# ----------------------------------------------------------------

ARCHIVO_DATOS = 'TodasOct-Feb_ParaAnalisisPython.csv'

# Nombres de las columnas en tu CSV
COL_FECHA = 'Fecha grafico'
COL_TEMP_EXT = 'Temp ext Gbourg'

# --- RUTA DE SALIDA CORREGIDA ---
CARPETA_SALIDA = 'Resultados_KPIs_Silvana'

# --- PARÁMETROS DE CÁLCULO ---
DATA_FREQUENCY_MIN = 15
TIME_INTERVAL_H = DATA_FREQUENCY_MIN / 60.0 # 0.25 para 15 min

# Horario de ocupación (0 a 24 para análisis completo)
HORA_INICIO_OCUPACION = 0
HORA_FIN_OCUPACION = 24

# Parámetros para HE 
HE_CATEGORY = 'III'
HE_MODEL = 'freeruning' 

# ----------------------------------------------------------------
# FUNCIÓN DE CARGA Y PREPARACIÓN
# ----------------------------------------------------------------

def load_and_prepare_data():
    """Carga el DataFrame unificado y realiza la limpieza inicial."""
    try:
        df_full = pd.read_csv(
            ARCHIVO_DATOS,
            sep=',' 
        )
        print(f"DataFrame unificado CSV cargado con {len(df_full)} registros.")

    except FileNotFoundError:
        print(f"❌ ERROR: Archivo '{ARCHIVO_DATOS}' no encontrado.")
        return None
    except Exception as e:
        print(f"❌ ERROR al leer el archivo: {e}: {type(e).__name__}")
        return None

    # --- 1. SANEAMIENTO DE LA COLUMNA DE TIEMPO ---
    if COL_FECHA != 'datetime' and COL_FECHA in df_full.columns:
         df_full.rename(columns={COL_FECHA: 'datetime'}, inplace=True)
    elif COL_FECHA not in df_full.columns:
         print(f"❌ ERROR: La columna de fecha '{COL_FECHA}' no se encontró.")
         return None

    df_full['datetime'] = pd.to_datetime(df_full['datetime'], errors='coerce', format='%d/%m/%Y %H:%M')

    # 2. Validar COL_TEMP_EXT (nombre)
    if COL_TEMP_EXT not in df_full.columns:
        print(f"❌ ERROR: Columna requerida '{COL_TEMP_EXT}' no encontrada.")
        return None
        
    return df_full

# ----------------------------------------------------------------
# FUNCIONES DE CÁLCULO (HI y WBGT)
# ----------------------------------------------------------------

def run_hi_wbgt_analysis(df_full):
    """Calcula HI y WBGT para pares Temp/HR y consolida."""
    print("\n" + "="*60)
    print("🔍 Calculando Heat Index (HI) y WBGT para Zonas Interiores...")
    
    df_hi_consolidado = df_full[['datetime']].copy()
    df_wbgt_consolidado = df_full[['datetime']].copy()
    columnas_disponibles = df_full.columns.tolist()
    zonas_procesadas = 0

    prefixes = sorted(list(set([
        col.replace('Temp_', '').replace('HR_', '').strip() 
        for col in columnas_disponibles if 'Temp_' in col or 'HR_' in col
    ])))

    for prefix in prefixes:
        col_temp = f'Temp_{prefix}'
        col_hr = f'HR_{prefix}'
        
        if col_temp in columnas_disponibles and col_hr in columnas_disponibles:
            df_subset = df_full[['datetime', col_temp, col_hr]].copy()
            df_subset.dropna(subset=[col_temp, col_hr], inplace=True) 

            if not df_subset.empty:
                try:
                    hi_series = KPIs.calculate_heat_index(df_data=df_subset, col_temp=col_temp, col_humidity=col_hr)
                    df_hi_consolidado[f'{prefix}_HI'] = hi_series.reindex(df_full.index)
                    
                    wbgt_series = KPIs.calculate_wbgt(df_data=df_subset, col_temp=col_temp, col_humidity=col_hr)
                    df_wbgt_consolidado[f'{prefix}_WBGT'] = wbgt_series.reindex(df_full.index)
                    
                    zonas_procesadas += 1
                
                except Exception as e:
                    print(f"    ❌ ERROR al calcular HI/WBGT para {prefix}: {e}")
        
    if zonas_procesadas > 0:
        nombre_archivo_hi = os.path.join(CARPETA_SALIDA, 'HeatIndex_Interiores_Consolidado.xlsx')
        df_hi_consolidado.to_excel(nombre_archivo_hi, index=False)
        print(f"\n✅ HI de {zonas_procesadas} zonas guardado en '{nombre_archivo_hi}'.")
        
        nombre_archivo_wbgt = os.path.join(CARPETA_SALIDA, 'WBGT_Interiores_Consolidado.xlsx')
        df_wbgt_consolidado.to_excel(nombre_archivo_wbgt, index=False)
        print(f"✅ WBGT de {zonas_procesadas} zonas guardado en '{nombre_archivo_wbgt}'.")
    else:
        print("\n❌ ADVERTENCIA: No se encontraron pares válidos (Temp/HR) para calcular HI/WBGT.")
    
    return df_hi_consolidado, df_wbgt_consolidado

# ----------------------------------------------------------------
# FUNCIÓN CONSOLIDADA (IOD, AWD, ALPHA_IOD, HE)
# ----------------------------------------------------------------

def run_daily_metrics_and_consolidate(df_full):
    """Calcula IOD, AWD, Alpha_IOD y HE diarios y consolida."""
    print("\n" + "="*60)
    print("🔍 Calculando IOD, AWD, Alpha_IOD y HE Diarios...")
    
    temp_interior_cols = [
        col for col in df_full.columns 
        if col.startswith('Temp_') and col != COL_TEMP_EXT
    ]
    
    if not temp_interior_cols:
        print("❌ ADVERTENCIA: No se encontraron columnas 'Temp_*' para métricas diarias.")
        return None

    df_consolidado = None
    
    # --- 1. Calcular AWD Diario (Base 18°C) ---
    try:
        df_awd = KPIs.calculate_awd(
            df_data=df_full.copy(), 
            col_temp_ext=COL_TEMP_EXT, 
            temp_base=18, 
            group_by_day=True,
            time_interval_h=TIME_INTERVAL_H
        ).rename(columns={'AWD_Diario': 'AWD_Diario (Base 18)'})
        
        awd_total = KPIs.calculate_awd(
            df_data=df_full.copy(), 
            col_temp_ext=COL_TEMP_EXT, 
            group_by_day=False,
            time_interval_h=TIME_INTERVAL_H
        )
        print(f"    ✅ AWD Diario calculado. Total acumulado: {awd_total:.4f}")
        df_consolidado = df_awd
    except Exception as e:
        print(f"    ❌ ERROR al calcular AWD Diario: {e}")
        return None

    # --- 2. Iterar por Zona para IOD, Alpha_IOD y HE ---
    for col_int in temp_interior_cols:
        nombre_zona = col_int.replace('Temp_', '')
        
        try:
            # --- CÁLCULO IOD y ALPHA_IOD ---
            df_alpha = KPIs.calculate_alpha_iod(
                df_data=df_full.copy(), 
                col_temp_int=col_int, 
                col_temp_ext=COL_TEMP_EXT,
                inicio_horas=HORA_INICIO_OCUPACION, 
                fin_horas=HORA_FIN_OCUPACION,
                group_by_day=True, 
                data_frequency_min=DATA_FREQUENCY_MIN
            )
            df_metrics = df_alpha[['Fecha', 'IOD_Diario', 'alpha_IOD']].rename(columns={
                'IOD_Diario': f'{nombre_zona}_IOD',
                'alpha_IOD': f'{nombre_zona}_Alpha'
            })
            
            # --- CÁLCULO HE (Heat Exceedance) ---
            df_he = KPIs.calculate_he(
                df_data=df_full.copy(),
                col_temp_int=col_int,
                col_temp_ext=COL_TEMP_EXT,
                building_category=HE_CATEGORY,
                model_type=HE_MODEL,
                inicio_horas=HORA_INICIO_OCUPACION,
                fin_horas=HORA_FIN_OCUPACION,
                data_frequency_min=DATA_FREQUENCY_MIN
            )
            df_metrics = pd.merge(df_metrics, df_he[['Fecha', 'HE_Diario']], on='Fecha', how='outer')
            df_metrics.rename(columns={'HE_Diario': f'{nombre_zona}_HE_h'}, inplace=True)

            # --- MERGE CON EL CONSOLIDADO PRINCIPAL ---
            df_consolidado = pd.merge(
                df_consolidado, 
                df_metrics, 
                on='Fecha', 
                how='outer'
            )
            print(f"    ✅ IOD/Alpha/HE para: {nombre_zona}")

        except Exception as e:
            print(f"    ❌ ERROR al calcular IOD/Alpha/HE para {nombre_zona}: {e}")

    # --- 3. Exportar y Reportar Totales ---
    if df_consolidado is not None:
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, 'KPIs_Diarios_Consolidado.xlsx')
        df_consolidado.to_excel(nombre_archivo_salida, index=False)
        print(f"\n✅ Todos los KPIs Diarios consolidados en '{nombre_archivo_salida}'.")
        
        # Calcular y mostrar el Alpha Total promedio para todas las zonas
        try:
            alpha_total_list = []
            for col_int in temp_interior_cols:
                alpha_total = KPIs.calculate_alpha_iod(
                    df_data=df_full.copy(), col_temp_int=col_int, col_temp_ext=COL_TEMP_EXT,
                    inicio_horas=HORA_INICIO_OCUPACION, fin_horas=HORA_FIN_OCUPACION,
                    group_by_day=False, data_frequency_min=DATA_FREQUENCY_MIN
                )
                alpha_total_list.append(alpha_total)
            print(f"✅ Alpha_IOD Total (Promedio de todas las zonas): {np.mean(alpha_total_list):.4f}")
        except:
            pass
    else:
        print("\n❌ ADVERTENCIA: No se pudo generar el archivo consolidado diario.")

# ----------------------------------------------------------------
# FUNCIÓN DE CÁLCULO DEL SCORE DE RESILIENCIA
# ----------------------------------------------------------------

def run_resilience_score(df_full, df_hi_full, df_wbgt_full):
    """
    Calcula el Score de Resiliencia para cada zona.
    Esta función PREPARA los datos (filtra, alinea) y luego llama
    a la función de cálculo 'calculate_score_amaripadath'.
    """
    temp_interior_cols = [col for col in df_full.columns if col.startswith('Temp_')]
    scores = {}

    print("\n" + "="*60)
    print("Iniciando Cálculo del Score de Resiliencia por Zona...")

    for col_int in temp_interior_cols:
        nombre_zona = col_int.replace('Temp_', '')
        
        # --- 1. CÁLCULO DE IOD PROMEDIO DEL PERIODO ---
        try:
            iod_average_period = KPIs.calculate_iod(
                df_data=df_full.copy(), 
                col_temp_int=col_int,
                col_temp_ext=COL_TEMP_EXT,
                inicio_horas=HORA_INICIO_OCUPACION, 
                fin_horas=HORA_FIN_OCUPACION,
                group_by_day=False, # <-- Correcto, calcula el promedio total
                data_frequency_min=DATA_FREQUENCY_MIN
            )
            
        except Exception as e:
            print(f"❌ ERROR (IOD Promedio) para {nombre_zona}: {e}")
            continue

        # --- 2. PREPARACIÓN Y FILTRADO DE KPIs HORARIOS (HE, HI, WBGT) ---
        
        # 2.1 Preparar HE (T_op / Temp Interior)
        # Filtramos por ocupación y preparamos el DF
        df_he_temp = df_full[
            (df_full['datetime'].dt.hour >= HORA_INICIO_OCUPACION) &
            (df_full['datetime'].dt.hour < HORA_FIN_OCUPACION)
        ].copy()
        df_he_clean = df_he_temp[['datetime', col_int]].rename(
            columns={'datetime': 'datetime', col_int: 'HE'}
        ).set_index('datetime').dropna() # dropna aquí es correcto para esta serie

        # 2.2 Preparar HI y WBGT
        col_hi = f'{nombre_zona}_HI'
        col_wbgt = f'{nombre_zona}_WBGT'
        
        if col_hi not in df_hi_full.columns or col_wbgt not in df_wbgt_full.columns:
            print(f"❌ ADVERTENCIA: No se encontraron columnas HI/WBGT para {nombre_zona}. Saltando score.")
            continue
            
        # Filtrar HI y WBGT por horas de ocupación
        df_hi_occupied = df_hi_full[
            (df_hi_full['datetime'].dt.hour >= HORA_INICIO_OCUPACION) &
            (df_hi_full['datetime'].dt.hour < HORA_FIN_OCUPACION)
        ]
        df_wbgt_occupied = df_wbgt_full[
            (df_wbgt_full['datetime'].dt.hour >= HORA_INICIO_OCUPACION) &
            (df_wbgt_full['datetime'].dt.hour < HORA_FIN_OCUPACION)
        ]

        df_hi_clean = df_hi_occupied[['datetime', col_hi]].rename(
            columns={'datetime': 'datetime', col_hi: 'HI'}
        ).set_index('datetime').dropna()
        
        df_wbgt_clean = df_wbgt_occupied[['datetime', col_wbgt]].rename(
            columns={'datetime': 'datetime', col_wbgt: 'WBGT'}
        ).set_index('datetime').dropna()
        
        # --- 3. ALINEACIÓN ESTRICTA (CRÍTICO) ---
        # El script de prueba se asegura de que los datos estén perfectos.
        
        df_aligned = pd.merge(df_hi_clean, df_wbgt_clean, left_index=True, right_index=True, how='inner')
        df_aligned = pd.merge(df_aligned, df_he_clean, left_index=True, right_index=True, how='inner')
        
        if df_aligned.empty:
            print(f"❌ ADVERTENCIA: Datos insuficientes (HI/WBGT/HE horario) tras alinear para {nombre_zona}.")
            continue
             
        # Separar los DFs finales alineados
        df_hi_final = df_aligned[['HI']]
        df_wbgt_final = df_aligned[['WBGT']]
        df_he_final = df_aligned[['HE']]

        # --- 4. CÁLCULO DEL SCORE ---
        print(f"\nCalculando Score para Zona: {nombre_zona}...")
        try:
            # Llamada a la función SIMPLIFICADA de la librería
            score = KPIs.calculate_score_amaripadath(
                iod_average_period=iod_average_period,
                df_hi_aligned=df_hi_final,
                df_wbgt_aligned=df_wbgt_final,
                df_he_aligned=df_he_final
                # Ya no se pasan horas de inicio/fin
            )
            scores[nombre_zona] = score
        except Exception as e:
            print(f"❌ ERROR al calcular score para {nombre_zona}: {e}")

    # --- 5. Reporte Final ---
    if scores:
        df_scores = pd.DataFrame(list(scores.items()), columns=['Zona', 'Resilience_Score'])
        nombre_archivo_score = os.path.join(CARPETA_SALIDA, 'Resilience_Scores_Consolidado.xlsx')
        df_scores.to_excel(nombre_archivo_score, index=False)
        print("\n" + "="*60)
        print("🎉 Scores de Resiliencia Finales:")
        print(df_scores.to_string(index=False))
        print(f"✅ Scores guardados en '{nombre_archivo_score}'.")
        print("="*60)
        
    return scores


# ----------------------------------------------------------------
# EJECUCIÓN PRINCIPAL (CORREGIDA)
# ----------------------------------------------------------------

if __name__ == '__main__':
    df_data = load_and_prepare_data()
    
    if df_data is not None:
        # 1. Crear la carpeta de salida
        if not os.path.exists(CARPETA_SALIDA):
            os.makedirs(CARPETA_SALIDA)
            
        # 2. Ejecutar análisis de HI y WBGT
        df_hi_c, df_wbgt_c = run_hi_wbgt_analysis(df_data.copy())
        
        # 3. Ejecutar análisis diario (IOD, AWD, Alpha_IOD, HE)
        run_daily_metrics_and_consolidate(df_data.copy())
        
        # 4. Ejecutar el Score de Resiliencia (NUEVO PASO)
        if df_hi_c is not None and df_wbgt_c is not None:
             run_resilience_score(df_data.copy(), df_hi_c, df_wbgt_c)