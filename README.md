# KPIs: Índices de Resiliencia Térmica

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/estado-En%20Desarrollo-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)

Librería de Python para el cálculo de índices de resiliencia térmica para zonas interiores. Incluye implementaciones de:

* **HI (Heat Index):** Índice de calor.
* **IOD (Indoor Overheating Degree):** Grado de sobrecalentamiento interior.
* **AWD (Ambient Warmness Degree):** Grado de calidez ambiental.
* **ALPHA_IOD (Overheating escalation factor):** Factor de aumento de sobrecalentamiento.
* **WBGT (Wet Bulb Globe Temperature):** Estimación simplificada para interiores.
* **HE (Heat Exceedance):** Horas de superación de umbrales de confort.
* **Regresión lineal de IOD:** Análisis de inercia térmica y radiación solar mediante modelos OLS.
* **Score de Resiliencia (Amaripadath):** Evaluación integral basada en ponderación de IOD y horas de cumplimiento de HI, WBGT y HE ($T_{op}$).

También permite calcular:
* **Trm (Runing Mean Temperature):** Promedio de la temperatura exterior de los 7 días anteriores.
* **Tconf (Indoor Overheating Degree):** Temperatura de confort de acuerdo a la norma ISO 17772-1

## Requisitos Previos

Para utilizar esta librería es necesario contar con **Python 3.8 o superior**.

Las dependencias principales son:
* `pandas` (Manipulación de series temporales)
* `numpy` (Cálculos numéricos)
* `statsmodels` (Para modelos de regresión lineal)
* `scikit-learn` (Para pre-procesamiento de datos en regresiones)

## Uso Básico
A continuación se muestra un ejemplo de cómo calcular el Score de Resiliencia.

import pandas as pd
import KPIs

# --- PASO 1: Calcular el IOD promedio del periodo ---
# Se asume que df_datos tiene columnas de temperatura interior y exterior
iod_promedio = KPIs.calculate_iod(
    df_data=df_datos, 
    col_temp_int='Temp_Dormitorio',
    col_temp_ext='Temp_Exterior',
    inicio_horas=0, 
    fin_horas=24,
    group_by_day=False, # False para obtener el promedio total
    data_frequency_min=15
)

# --- PASO 2: Preparar DataFrames Horarios (HI, WBGT, HE) ---
# Nota: Estos deben estar filtrados por las horas de ocupación (ej. noche)
# y alineados temporalmente.

# ... (código de usuario para filtrar df_hi, df_wbgt, df_he) ...

# --- PASO 3: Calcular Score de Resiliencia ---
score_final = KPIs.calculate_score_amaripadath(
    iod_average_period=iod_promedio,
    df_hi=df_hi_filtrado,
    df_wbgt=df_wbgt_filtrado,
    df_he=df_he_filtrado
)

print(f"Score de Resiliencia Térmica: {score_final:.4f}")


## Instalación

Actualmente, la librería se encuentra alojada en GitHub. Puedes instalarla directamente usando `pip`:

```bash
pip install git+[https://github.com/camiescuderofiqueni/KPIs.git](https://github.com/camiescuderofiqueni/KPIs.git)
```

## Contribución

¡Las contribuciones son bienvenidas! Si encuentras un error o quieres proponer una mejora:

1. Abre un Issue para discutir el cambio.
2. O envía un Pull Request con tus mejoras.

## Licencia
Este proyecto está bajo la Licencia MIT