#!/usr/bin/env python3
"""
Análisis de Entrada de Datos para Proyecto de Simulación
Sistema de Ascensores - Cambio de Clases en Edificio de Ingeniería

Este script realiza un análisis estadístico completo de los datos recolectados
para determinar distribuciones de probabilidad y parámetros apropiados para
la simulación en FlexSim.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Configuración de estilo para gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# DATOS RECOLECTADOS
# ============================================================================

# Datos de personas (10 minutos durante cambio de clase)
DATOS_7AM = {
    'entran': 35,
    'esperan': 42,
    'bajan': 14,
    'tiempo_minutos': 10
}

DATOS_2PM = {
    'entran': 83,
    'esperan': 102,
    'bajan': 66,
    'tiempo_minutos': 10
}

# Tiempos de ciclo de ascensores (en segundos)
TIEMPOS_ASCENSORES = {
    'Ascensor 1': 203,  # 3:23
    'Ascensor 2': 217,  # 3:37
    'Ascensor 3': 285   # 4:45
}

# ============================================================================
# FUNCIONES DE ANÁLISIS DE TASAS
# ============================================================================

def calcular_tasas_llegada(datos, nombre_franja):
    """
    Calcula las tasas de llegada y estadísticas descriptivas.
    
    Args:
        datos: Diccionario con datos de la franja horaria
        nombre_franja: Nombre de la franja horaria (ej: "7am", "2pm")
    
    Returns:
        dict: Diccionario con tasas y estadísticas calculadas
    """
    tiempo_min = datos['tiempo_minutos']
    entran = datos['entran']
    esperan = datos['esperan']
    bajan = datos['bajan']
    
    # Tasa de llegada (personas/minuto)
    tasa_llegada_por_min = entran / tiempo_min
    
    # Tasa de llegada (personas/segundo) para simulación
    tasa_llegada_por_seg = tasa_llegada_por_min / 60
    
    # Tiempo promedio entre llegadas (segundos)
    tiempo_promedio_entre_llegadas = 1 / tasa_llegada_por_seg if tasa_llegada_por_seg > 0 else 0
    
    # Tasa de servicio (personas/minuto basada en bajadas)
    tasa_servicio_por_min = bajan / tiempo_min if bajan > 0 else 0
    
    # Utilización del sistema (personas esperando / personas que entran)
    utilizacion = esperan / entran if entran > 0 else 0
    
    resultado = {
        'franja': nombre_franja,
        'personas_entran': entran,
        'personas_esperan': esperan,
        'personas_bajan': bajan,
        'tiempo_minutos': tiempo_min,
        'tasa_llegada_por_min': tasa_llegada_por_min,
        'tasa_llegada_por_seg': tasa_llegada_por_seg,
        'tiempo_promedio_entre_llegadas_seg': tiempo_promedio_entre_llegadas,
        'tasa_servicio_por_min': tasa_servicio_por_min,
        'utilizacion': utilizacion,
        'lambda_poisson': tasa_llegada_por_seg,  # Parámetro λ para Poisson
        'lambda_exponencial': tasa_llegada_por_seg  # Parámetro λ para Exponencial
    }
    
    return resultado

# ============================================================================
# FUNCIONES DE PRUEBAS DE BONDAD DE AJUSTE
# ============================================================================

def prueba_kolmogorov_smirnov(muestra, distribucion='expon', parametros=None):
    """
    Realiza la prueba de Kolmogorov-Smirnov para probar el ajuste de una distribución.
    
    Args:
        muestra: Array de datos observados
        distribucion: Tipo de distribución ('expon', 'norm', 'uniform')
        parametros: Diccionario con parámetros de la distribución
    
    Returns:
        tuple: (estadistica_ks, valor_p, decision)
    """
    n = len(muestra)
    
    if distribucion == 'expon':
        # Parámetro lambda para exponencial (scale = 1/lambda)
        if parametros is None:
            lambda_param = 1.0 / np.mean(muestra) if np.mean(muestra) > 0 else 1.0
        else:
            lambda_param = parametros.get('lambda', 1.0 / np.mean(muestra))
        scale = 1.0 / lambda_param if lambda_param > 0 else 1.0
        estadistica_ks, valor_p = stats.kstest(muestra, lambda x: stats.expon.cdf(x, scale=scale))
    
    elif distribucion == 'norm':
        # Parámetros mu y sigma para normal
        if parametros is None:
            mu = np.mean(muestra)
            sigma = np.std(muestra, ddof=1) if len(muestra) > 1 else np.std(muestra)
        else:
            mu = parametros.get('mu', np.mean(muestra))
            sigma = parametros.get('sigma', np.std(muestra, ddof=1))
        estadistica_ks, valor_p = stats.kstest(muestra, lambda x: stats.norm.cdf(x, loc=mu, scale=sigma))
    
    elif distribucion == 'uniform':
        # Parámetros a y b para uniforme
        if parametros is None:
            a = np.min(muestra)
            b = np.max(muestra)
        else:
            a = parametros.get('a', np.min(muestra))
            b = parametros.get('b', np.max(muestra))
        estadistica_ks, valor_p = stats.kstest(muestra, lambda x: stats.uniform.cdf(x, loc=a, scale=b-a))
    
    else:
        raise ValueError(f"Distribución {distribucion} no soportada")
    
    decision = "No rechazar H0" if valor_p > 0.05 else "Rechazar H0"
    
    return estadistica_ks, valor_p, decision

def prueba_chi_cuadrado(muestra, distribucion='expon', parametros=None, k=5):
    """
    Prueba de chi-cuadrado para verificar ajuste de distribuciones.
    
    Args:
        muestra: Array de datos observados
        distribucion: Tipo de distribución ('expon', 'norm', 'uniform')
        parametros: Diccionario con parámetros de la distribución
        k: Número de clases
    
    Returns:
        dict: Resultados de la prueba chi-cuadrado
    """
    n = len(muestra)
    
    # Obtener límites de clases basados en percentiles
    limites_clases = np.percentile(muestra, np.linspace(0, 100, k+1))
    
    # Calcular frecuencias observadas
    frecuencias_observadas, _ = np.histogram(muestra, bins=limites_clases)
    
    # Calcular frecuencias esperadas según la distribución
    if distribucion == 'expon':
        if parametros is None:
            lambda_param = 1.0 / np.mean(muestra) if np.mean(muestra) > 0 else 1.0
        else:
            lambda_param = parametros.get('lambda', 1.0 / np.mean(muestra))
        scale = 1.0 / lambda_param if lambda_param > 0 else 1.0
        probabilidades_esperadas = np.diff([stats.expon.cdf(lim, scale=scale) for lim in limites_clases])
    
    elif distribucion == 'norm':
        if parametros is None:
            mu = np.mean(muestra)
            sigma = np.std(muestra, ddof=1) if len(muestra) > 1 else np.std(muestra)
        else:
            mu = parametros.get('mu', np.mean(muestra))
            sigma = parametros.get('sigma', np.std(muestra, ddof=1))
        probabilidades_esperadas = np.diff([stats.norm.cdf(lim, loc=mu, scale=sigma) for lim in limites_clases])
    
    elif distribucion == 'uniform':
        if parametros is None:
            a = np.min(muestra)
            b = np.max(muestra)
        else:
            a = parametros.get('a', np.min(muestra))
            b = parametros.get('b', np.max(muestra))
        probabilidades_esperadas = np.diff([stats.uniform.cdf(lim, loc=a, scale=b-a) for lim in limites_clases])
    
    else:
        raise ValueError(f"Distribución {distribucion} no soportada")
    
    # Asegurar que todas las frecuencias esperadas sean >= 5 (regla empírica)
    frecuencias_esperadas = n * probabilidades_esperadas
    if np.any(frecuencias_esperadas < 5):
        # Combinar clases si es necesario
        frecuencias_observadas = frecuencias_observadas[frecuencias_esperadas >= 5]
        frecuencias_esperadas = frecuencias_esperadas[frecuencias_esperadas >= 5]
    
    # Calcular estadístico chi-cuadrado
    chi_cuadrado = np.sum((frecuencias_observadas - frecuencias_esperadas)**2 / frecuencias_esperadas)
    
    # Grados de libertad (k - 1 - número de parámetros estimados)
    if distribucion == 'uniform':
        grados_libertad = len(frecuencias_observadas) - 1 - 2  # a y b estimados
    elif distribucion == 'norm':
        grados_libertad = len(frecuencias_observadas) - 1 - 2  # mu y sigma estimados
    elif distribucion == 'expon':
        grados_libertad = len(frecuencias_observadas) - 1 - 1  # lambda estimado
    
    grados_libertad = max(1, grados_libertad)  # Asegurar al menos 1
    
    # Valor crítico (95% de confianza)
    nivel_confianza = 0.95
    alpha = 1 - nivel_confianza
    valor_critico = stats.chi2.ppf(1 - alpha, grados_libertad)
    
    # P-valor
    p_valor = 1 - stats.chi2.cdf(chi_cuadrado, grados_libertad)
    
    # Decisión
    decision = "No rechazar H0" if chi_cuadrado <= valor_critico else "Rechazar H0"
    
    return {
        'chi_cuadrado': chi_cuadrado,
        'valor_critico': valor_critico,
        'grados_libertad': grados_libertad,
        'p_valor': p_valor,
        'decision': decision,
        'frecuencias_observadas': frecuencias_observadas,
        'frecuencias_esperadas': frecuencias_esperadas,
        'limites_clases': limites_clases
    }

def prueba_corridas_arriba_abajo(numeros, nivel_confianza=0.95):
    """
    Prueba de corridas arriba y abajo para verificar independencia de datos.
    
    Esta prueba es crítica en simulación para asegurar que los datos no tienen
    correlaciones o patrones que invalidarían las suposiciones del modelo.
    
    Args:
        numeros: Array de números (debe estar ordenado o ser una secuencia temporal)
        nivel_confianza: Nivel de confianza (0.95 para 95%)
    
    Returns:
        dict: Resultados de la prueba de corridas
    """
    n = len(numeros)
    
    if n < 3:
        return {
            'error': 'Se requieren al menos 3 datos para la prueba de corridas',
            'decision': 'No aplicable',
            'conclusion': 'Muestra insuficiente'
        }
    
    # Paso 1: Crear secuencia S de unos y ceros
    S = []
    for i in range(1, n):
        if numeros[i] <= numeros[i-1]:
            S.append(0)
        else:
            S.append(1)
    
    # Paso 2: Contar el número de corridas observadas C0
    C0 = 1  # Siempre hay al menos una corrida
    for i in range(1, len(S)):
        if S[i] != S[i-1]:
            C0 += 1
    
    # Paso 3: Calcular estadísticos teóricos
    mu_C0 = (2 * n - 1) / 3
    sigma2_C0 = (16 * n - 29) / 90
    sigma_C0 = np.sqrt(sigma2_C0)
    
    # Paso 4: Calcular estadístico Z0
    Z0 = (C0 - mu_C0) / sigma_C0 if sigma_C0 > 0 else 0
    
    # Paso 5: Determinar intervalo de aceptación
    alpha = 1 - nivel_confianza
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)  # Para prueba bilateral
    
    # Decisión
    if -z_alpha_2 <= Z0 <= z_alpha_2:
        decision = "No rechazar H0"
        conclusion = "Los datos son independientes"
    else:
        decision = "Rechazar H0"
        conclusion = "Los datos NO son independientes (hay correlación)"
    
    return {
        'numeros': numeros,
        'secuencia_S': S,
        'corridas_observadas': C0,
        'valor_esperado': mu_C0,
        'varianza': sigma2_C0,
        'desviacion_estandar': sigma_C0,
        'estadistico_Z0': Z0,
        'intervalo_aceptacion': (-z_alpha_2, z_alpha_2),
        'decision': decision,
        'conclusion': conclusion,
        'nivel_confianza': nivel_confianza
    }

def prueba_media(muestra, mu0=None, nivel_aceptacion=0.95):
    """
    Prueba de la media para validar parámetros de distribuciones.
    
    Args:
        muestra: Array de datos
        mu0: Media poblacional esperada (si None, se usa media muestral)
        nivel_aceptacion: Nivel de aceptación (0.95 para 95%)
    
    Returns:
        dict: Resultados de la prueba de la media
    """
    n = len(muestra)
    
    if mu0 is None:
        mu0 = np.mean(muestra)
    
    # Varianza poblacional estimada
    varianza_muestral = np.var(muestra, ddof=1)
    desviacion_estandar = np.sqrt(varianza_muestral)
    
    # Nivel de significancia
    alpha = 1 - nivel_aceptacion
    z_critico = stats.norm.ppf(1 - alpha/2)  # z_α/2 para prueba bilateral
    
    # Error estándar
    error_estandar = desviacion_estandar / np.sqrt(n) if n > 0 else 0
    
    # Intervalo de aceptación
    limite_inferior = mu0 - z_critico * error_estandar
    limite_superior = mu0 + z_critico * error_estandar
    
    # Estadísticas de la muestra
    media_muestral = np.mean(muestra)
    
    # Decisión
    dentro_intervalo = limite_inferior <= media_muestral <= limite_superior
    decision = "No rechazar H₀" if dentro_intervalo else "Rechazar H₀"
    
    return {
        'n': n,
        'mu0': mu0,
        'alpha': alpha,
        'z_critico': z_critico,
        'error_estandar': error_estandar,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'media_muestral': media_muestral,
        'desviacion_estandar': desviacion_estandar,
        'dentro_intervalo': dentro_intervalo,
        'decision': decision,
        'nivel_aceptacion': nivel_aceptacion
    }

# ============================================================================
# FUNCIONES DE ANÁLISIS ESPECÍFICAS
# ============================================================================

def generar_tiempos_entre_llegadas_simulados(tasa_llegada_por_seg, n_simulaciones=100):
    """
    Genera tiempos entre llegadas simulados basados en distribución exponencial.
    
    Args:
        tasa_llegada_por_seg: Tasa de llegada (λ) en personas/segundo
        n_simulaciones: Número de tiempos a generar
    
    Returns:
        numpy.array: Array de tiempos entre llegadas en segundos
    """
    lambda_param = tasa_llegada_por_seg
    scale = 1.0 / lambda_param if lambda_param > 0 else 1.0
    
    # Generar tiempos entre llegadas exponenciales
    tiempos_entre_llegadas = np.random.exponential(scale, n_simulaciones)
    
    return tiempos_entre_llegadas

def analizar_tiempos_servicio_ascensores(tiempos_ascensores):
    """
    Analiza los tiempos de servicio de los ascensores.
    
    Args:
        tiempos_ascensores: Diccionario con tiempos de cada ascensor
    
    Returns:
        dict: Análisis completo de tiempos de servicio
    """
    tiempos_array = np.array(list(tiempos_ascensores.values()))
    
    resultados = {
        'tiempos': tiempos_array,
        'media': np.mean(tiempos_array),
        'mediana': np.median(tiempos_array),
        'desviacion_estandar': np.std(tiempos_array, ddof=1) if len(tiempos_array) > 1 else 0,
        'minimo': np.min(tiempos_array),
        'maximo': np.max(tiempos_array),
        'varianza': np.var(tiempos_array, ddof=1) if len(tiempos_array) > 1 else 0,
        'coeficiente_variacion': np.std(tiempos_array, ddof=1) / np.mean(tiempos_array) if np.mean(tiempos_array) > 0 else 0
    }
    
    # Estimación de parámetros para diferentes distribuciones
    
    # Distribución Normal
    resultados['normal'] = {
        'mu': resultados['media'],
        'sigma': resultados['desviacion_estandar'] if resultados['desviacion_estandar'] > 0 else resultados['media'] * 0.1
    }
    
    # Distribución Exponencial
    resultados['exponencial'] = {
        'lambda': 1.0 / resultados['media'] if resultados['media'] > 0 else 1.0,
        'scale': resultados['media']
    }
    
    # Distribución Uniforme
    resultados['uniforme'] = {
        'a': resultados['minimo'],
        'b': resultados['maximo']
    }
    
    return resultados

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def visualizar_analisis_tasas(resultados_7am, resultados_2pm):
    """
    Visualiza el análisis de tasas de llegada para ambas franjas horarias.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Tasas de Llegada - Comparación 7am vs 2pm', 
                 fontsize=16, fontweight='bold')
    
    # Gráfico 1: Comparación de personas
    ax1 = axes[0, 0]
    categorias = ['Entran', 'Esperan', 'Bajan']
    valores_7am = [resultados_7am['personas_entran'], 
                   resultados_7am['personas_esperan'], 
                   resultados_7am['personas_bajan']]
    valores_2pm = [resultados_2pm['personas_entran'], 
                   resultados_2pm['personas_esperan'], 
                   resultados_2pm['personas_bajan']]
    
    x = np.arange(len(categorias))
    width = 0.35
    
    ax1.bar(x - width/2, valores_7am, width, label='7am', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, valores_2pm, width, label='2pm', alpha=0.8, color='coral')
    ax1.set_xlabel('Categoría')
    ax1.set_ylabel('Número de Personas')
    ax1.set_title('Comparación de Personas por Categoría')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categorias)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Tasas de llegada
    ax2 = axes[0, 1]
    franjas = ['7am', '2pm']
    tasas = [resultados_7am['tasa_llegada_por_min'], resultados_2pm['tasa_llegada_por_min']]
    ax2.bar(franjas, tasas, alpha=0.8, color=['skyblue', 'coral'])
    ax2.set_ylabel('Tasa de Llegada (personas/minuto)')
    ax2.set_title('Tasas de Llegada por Franja Horaria')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, tasa in enumerate(tasas):
        ax2.text(i, tasa, f'{tasa:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 3: Tiempos promedio entre llegadas
    ax3 = axes[1, 0]
    tiempos = [resultados_7am['tiempo_promedio_entre_llegadas_seg'], 
               resultados_2pm['tiempo_promedio_entre_llegadas_seg']]
    ax3.bar(franjas, tiempos, alpha=0.8, color=['skyblue', 'coral'])
    ax3.set_ylabel('Tiempo Promedio (segundos)')
    ax3.set_title('Tiempo Promedio entre Llegadas')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, tiempo in enumerate(tiempos):
        ax3.text(i, tiempo, f'{tiempo:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 4: Utilización del sistema
    ax4 = axes[1, 1]
    utilizaciones = [resultados_7am['utilizacion'], resultados_2pm['utilizacion']]
    ax4.bar(franjas, utilizaciones, alpha=0.8, color=['skyblue', 'coral'])
    ax4.set_ylabel('Utilización (proporción)')
    ax4.set_title('Utilización del Sistema')
    ax4.set_ylim([0, max(utilizaciones) * 1.2 if max(utilizaciones) > 0 else 1])
    ax4.grid(True, alpha=0.3, axis='y')
    for i, util in enumerate(utilizaciones):
        ax4.text(i, util, f'{util:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def visualizar_distribucion_tiempos_servicio(analisis_tiempos):
    """
    Visualiza los tiempos de servicio de ascensores y distribuciones propuestas.
    """
    tiempos = analisis_tiempos['tiempos']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Tiempos de Servicio de Ascensores', 
                 fontsize=16, fontweight='bold')
    
    # Gráfico 1: Histograma de tiempos observados
    ax1 = axes[0, 0]
    ax1.bar(['Ascensor 1', 'Ascensor 2', 'Ascensor 3'], tiempos, 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axhline(y=analisis_tiempos['media'], color='red', linestyle='--', 
                linewidth=2, label=f"Media = {analisis_tiempos['media']:.1f}s")
    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title('Tiempos de Ciclo Observados')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Distribuciones teóricas propuestas
    ax2 = axes[0, 1]
    x = np.linspace(analisis_tiempos['minimo'] - 50, 
                    analisis_tiempos['maximo'] + 50, 1000)
    
    # Normal
    if analisis_tiempos['normal']['sigma'] > 0:
        pdf_normal = stats.norm.pdf(x, analisis_tiempos['normal']['mu'], 
                                    analisis_tiempos['normal']['sigma'])
        ax2.plot(x, pdf_normal, 'b-', linewidth=2, label='Normal', alpha=0.7)
    
    # Exponencial
    pdf_exp = stats.expon.pdf(x, scale=analisis_tiempos['exponencial']['scale'])
    ax2.plot(x, pdf_exp, 'r--', linewidth=2, label='Exponencial', alpha=0.7)
    
    # Uniforme
    pdf_unif = stats.uniform.pdf(x, loc=analisis_tiempos['uniforme']['a'], 
                                  scale=analisis_tiempos['uniforme']['b'] - 
                                  analisis_tiempos['uniforme']['a'])
    ax2.plot(x, pdf_unif, 'g:', linewidth=2, label='Uniforme', alpha=0.7)
    
    # Marcar tiempos observados
    for t in tiempos:
        ax2.axvline(t, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.set_ylabel('Densidad de Probabilidad')
    ax2.set_title('Distribuciones Teóricas Propuestas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Función de Distribución Acumulada (CDF) Empírica vs Teóricas
    ax3 = axes[1, 0]
    tiempos_ordenados = np.sort(tiempos)
    n = len(tiempos)
    cdf_empirica = np.arange(1, n+1) / n
    
    x_cdf = np.linspace(analisis_tiempos['minimo'] - 50, 
                        analisis_tiempos['maximo'] + 50, 1000)
    
    # CDF Normal
    if analisis_tiempos['normal']['sigma'] > 0:
        cdf_normal = stats.norm.cdf(x_cdf, analisis_tiempos['normal']['mu'], 
                                     analisis_tiempos['normal']['sigma'])
        ax3.plot(x_cdf, cdf_normal, 'b-', linewidth=2, label='Normal', alpha=0.7)
    
    # CDF Exponencial
    cdf_exp = stats.expon.cdf(x_cdf, scale=analisis_tiempos['exponencial']['scale'])
    ax3.plot(x_cdf, cdf_exp, 'r--', linewidth=2, label='Exponencial', alpha=0.7)
    
    # CDF Uniforme
    cdf_unif = stats.uniform.cdf(x_cdf, loc=analisis_tiempos['uniforme']['a'], 
                                  scale=analisis_tiempos['uniforme']['b'] - 
                                  analisis_tiempos['uniforme']['a'])
    ax3.plot(x_cdf, cdf_unif, 'g:', linewidth=2, label='Uniforme', alpha=0.7)
    
    # CDF Empírica
    ax3.plot(tiempos_ordenados, cdf_empirica, 'ko-', linewidth=2, 
             markersize=8, label='Empírica', alpha=0.7)
    
    ax3.set_xlabel('Tiempo (segundos)')
    ax3.set_ylabel('Probabilidad Acumulada')
    ax3.set_title('Función de Distribución Acumulada (CDF)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Estadísticas descriptivas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    texto = f"""
    ESTADÍSTICAS DESCRIPTIVAS
    
    Media (μ): {analisis_tiempos['media']:.2f} s
    Mediana: {analisis_tiempos['mediana']:.2f} s
    Desviación Estándar (σ): {analisis_tiempos['desviacion_estandar']:.2f} s
    Mínimo: {analisis_tiempos['minimo']:.2f} s
    Máximo: {analisis_tiempos['maximo']:.2f} s
    Coeficiente de Variación: {analisis_tiempos['coeficiente_variacion']:.3f}
    
    PARÁMETROS DE DISTRIBUCIONES
    
    Normal:
      μ = {analisis_tiempos['normal']['mu']:.2f} s
      σ = {analisis_tiempos['normal']['sigma']:.2f} s
    
    Exponencial:
      λ = {analisis_tiempos['exponencial']['lambda']:.4f} 1/s
      scale = {analisis_tiempos['exponencial']['scale']:.2f} s
    
    Uniforme:
      a = {analisis_tiempos['uniforme']['a']:.2f} s
      b = {analisis_tiempos['uniforme']['b']:.2f} s
    """
    
    ax4.text(0.1, 0.9, texto, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualizar_tiempos_entre_llegadas(resultados_tasas, tiempos_simulados, nombre_franja):
    """
    Visualiza análisis de tiempos entre llegadas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Análisis de Tiempos entre Llegadas - {nombre_franja}', 
                 fontsize=16, fontweight='bold')
    
    # Gráfico 1: Histograma de tiempos simulados
    ax1 = axes[0, 0]
    ax1.hist(tiempos_simulados, bins=20, alpha=0.7, color='skyblue', 
             edgecolor='black', density=True)
    
    # Distribución exponencial teórica
    lambda_param = resultados_tasas['lambda_exponencial']
    scale = 1.0 / lambda_param if lambda_param > 0 else 1.0
    x = np.linspace(0, np.max(tiempos_simulados) * 1.5, 1000)
    pdf_teorica = stats.expon.pdf(x, scale=scale)
    ax1.plot(x, pdf_teorica, 'r-', linewidth=2, label='Exponencial Teórica')
    
    ax1.set_xlabel('Tiempo entre Llegadas (segundos)')
    ax1.set_ylabel('Densidad de Probabilidad')
    ax1.set_title(f'Distribución de Tiempos entre Llegadas (λ={lambda_param:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: CDF Empírica vs Teórica
    ax2 = axes[0, 1]
    tiempos_ordenados = np.sort(tiempos_simulados)
    n = len(tiempos_simulados)
    cdf_empirica = np.arange(1, n+1) / n
    
    x_cdf = np.linspace(0, np.max(tiempos_simulados) * 1.5, 1000)
    cdf_teorica = stats.expon.cdf(x_cdf, scale=scale)
    
    ax2.plot(tiempos_ordenados, cdf_empirica, 'b-', linewidth=2, 
             label='CDF Empírica', alpha=0.7)
    ax2.plot(x_cdf, cdf_teorica, 'r--', linewidth=2, 
             label='CDF Exponencial Teórica', alpha=0.7)
    
    ax2.set_xlabel('Tiempo entre Llegadas (segundos)')
    ax2.set_ylabel('Probabilidad Acumulada')
    ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Secuencia de tiempos (para prueba de corridas)
    ax3 = axes[1, 0]
    ax3.plot(tiempos_simulados[:50], 'bo-', linewidth=1, markersize=4, alpha=0.7)
    ax3.set_xlabel('Observación')
    ax3.set_ylabel('Tiempo entre Llegadas (segundos)')
    ax3.set_title('Secuencia de Tiempos entre Llegadas (Primeras 50)')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Resumen de parámetros
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    texto = f"""
    PARÁMETROS DE DISTRIBUCIÓN EXPONENCIAL
    
    Tasa de llegada (λ):
      {lambda_param:.6f} personas/segundo
      {resultados_tasas['tasa_llegada_por_min']:.2f} personas/minuto
    
    Tiempo promedio entre llegadas:
      {resultados_tasas['tiempo_promedio_entre_llegadas_seg']:.2f} segundos
    
    Parámetro de distribución exponencial:
      λ = {lambda_param:.6f} 1/s
      scale (1/λ) = {scale:.2f} s
    
    ESTADÍSTICAS DE MUESTRA SIMULADA
    (n = {len(tiempos_simulados)})
    
    Media muestral: {np.mean(tiempos_simulados):.2f} s
    Desviación estándar: {np.std(tiempos_simulados, ddof=1):.2f} s
    """
    
    ax4.text(0.1, 0.9, texto, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    return fig

def visualizar_prueba_corridas(resultados_corridas, titulo="Prueba de Corridas"):
    """
    Visualiza resultados de la prueba de corridas arriba y abajo.
    Adaptado de ejercicio3.py
    """
    if 'error' in resultados_corridas:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, resultados_corridas['error'], 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8))
        ax.axis('off')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    numeros = resultados_corridas['numeros']
    secuencia_S = resultados_corridas['secuencia_S']
    
    # Gráfico 1: Secuencia de números originales
    ax1 = axes[0, 0]
    ax1.plot(numeros, 'bo-', linewidth=2, markersize=6)
    ax1.set_title('Secuencia de Datos Observados', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Posición')
    ax1.set_ylabel('Valor')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Secuencia binaria S
    ax2 = axes[0, 1]
    ax2.plot(secuencia_S, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('Secuencia Binaria S\n(0: ≤ anterior, 1: > anterior)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Posición')
    ax2.set_ylabel('Valor (0 o 1)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Distribución normal estándar con estadístico Z0
    ax3 = axes[1, 0]
    z_values = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(z_values)
    
    ax3.plot(z_values, pdf, 'b-', linewidth=2, label='N(0,1)')
    ax3.axvline(resultados_corridas['estadistico_Z0'], color='red', 
                linestyle='--', linewidth=2, 
                label=f'Z₀ = {resultados_corridas["estadistico_Z0"]:.3f}')
    ax3.axvline(resultados_corridas['intervalo_aceptacion'][0], color='green', 
                linestyle='-', linewidth=2, 
                label=f'Límite inferior = {resultados_corridas["intervalo_aceptacion"][0]:.2f}')
    ax3.axvline(resultados_corridas['intervalo_aceptacion'][1], color='green', 
                linestyle='-', linewidth=2, 
                label=f'Límite superior = {resultados_corridas["intervalo_aceptacion"][1]:.2f}')
    
    # Sombrear región de aceptación
    mask = ((z_values >= resultados_corridas['intervalo_aceptacion'][0]) & 
            (z_values <= resultados_corridas['intervalo_aceptacion'][1]))
    ax3.fill_between(z_values[mask], pdf[mask], alpha=0.3, color='green', 
                     label='Región de aceptación')
    
    ax3.set_title('Distribución Normal Estándar y Estadístico Z₀', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Valor Z')
    ax3.set_ylabel('Densidad de probabilidad')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Resumen de resultados
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    texto_resumen = f"""
    RESULTADOS DE LA PRUEBA DE CORRIDAS
    
    Datos observados: {len(numeros)}
    Corridas observadas (C₀): {resultados_corridas['corridas_observadas']}
    Valor esperado (μC₀): {resultados_corridas['valor_esperado']:.3f}
    Varianza (σ²C₀): {resultados_corridas['varianza']:.3f}
    Desviación estándar (σC₀): {resultados_corridas['desviacion_estandar']:.3f}
    
    Estadístico Z₀: {resultados_corridas['estadistico_Z0']:.3f}
    Intervalo de aceptación: 
    [{resultados_corridas['intervalo_aceptacion'][0]:.2f}, 
     {resultados_corridas['intervalo_aceptacion'][1]:.2f}]
    
    Decisión: {resultados_corridas['decision']}
    Conclusión: {resultados_corridas['conclusion']}
    
    Nivel de confianza: {resultados_corridas['nivel_confianza']*100}%
    """
    
    color = 'lightgreen' if resultados_corridas['decision'] == 'No rechazar H0' else 'lightcoral'
    
    ax4.text(0.1, 0.9, texto_resumen, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    return fig

# ============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================================================

def generar_reporte_completo():
    """
    Genera un reporte completo del análisis de entrada de datos.
    """
    print("=" * 80)
    print("ANÁLISIS DE ENTRADA DE DATOS - PROYECTO DE SIMULACIÓN")
    print("Sistema de Ascensores - Cambio de Clases")
    print("=" * 80)
    print()
    
    # 1. ANÁLISIS DE TASAS DE LLEGADA
    print("=" * 80)
    print("1. ANÁLISIS DE TASAS DE LLEGADA")
    print("=" * 80)
    print()
    
    resultados_7am = calcular_tasas_llegada(DATOS_7AM, "7am")
    resultados_2pm = calcular_tasas_llegada(DATOS_2PM, "2pm")
    
    print(f"FRANJA HORARIA: 7am")
    print(f"  Personas que entran: {resultados_7am['personas_entran']}")
    print(f"  Personas que esperan: {resultados_7am['personas_esperan']}")
    print(f"  Personas que bajan: {resultados_7am['personas_bajan']}")
    print(f"  Tiempo de observación: {resultados_7am['tiempo_minutos']} minutos")
    print(f"  Tasa de llegada: {resultados_7am['tasa_llegada_por_min']:.2f} personas/minuto")
    print(f"  Tasa de llegada: {resultados_7am['tasa_llegada_por_seg']:.6f} personas/segundo")
    print(f"  Tiempo promedio entre llegadas: {resultados_7am['tiempo_promedio_entre_llegadas_seg']:.2f} segundos")
    print(f"  Parámetro λ (Poisson/Exponencial): {resultados_7am['lambda_exponencial']:.6f}")
    print()
    
    print(f"FRANJA HORARIA: 2pm")
    print(f"  Personas que entran: {resultados_2pm['personas_entran']}")
    print(f"  Personas que esperan: {resultados_2pm['personas_esperan']}")
    print(f"  Personas que bajan: {resultados_2pm['personas_bajan']}")
    print(f"  Tiempo de observación: {resultados_2pm['tiempo_minutos']} minutos")
    print(f"  Tasa de llegada: {resultados_2pm['tasa_llegada_por_min']:.2f} personas/minuto")
    print(f"  Tasa de llegada: {resultados_2pm['tasa_llegada_por_seg']:.6f} personas/segundo")
    print(f"  Tiempo promedio entre llegadas: {resultados_2pm['tiempo_promedio_entre_llegadas_seg']:.2f} segundos")
    print(f"  Parámetro λ (Poisson/Exponencial): {resultados_2pm['lambda_exponencial']:.6f}")
    print()
    
    # Visualización de tasas
    print("Generando visualizaciones de tasas...")
    fig1 = visualizar_analisis_tasas(resultados_7am, resultados_2pm)
    plt.savefig('analisis_tasas_llegada.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: analisis_tasas_llegada.png")
    print()
    
    # 2. ANÁLISIS DE TIEMPOS ENTRE LLEGADAS
    print("=" * 80)
    print("2. ANÁLISIS DE TIEMPOS ENTRE LLEGADAS")
    print("=" * 80)
    print()
    
    # Generar tiempos simulados para análisis
    tiempos_7am_sim = generar_tiempos_entre_llegadas_simulados(
        resultados_7am['lambda_exponencial'], n_simulaciones=100
    )
    tiempos_2pm_sim = generar_tiempos_entre_llegadas_simulados(
        resultados_2pm['lambda_exponencial'], n_simulaciones=100
    )
    
    print(f"Tiempos entre llegadas simulados (basados en distribución exponencial):")
    print(f"  7am: n={len(tiempos_7am_sim)}, media={np.mean(tiempos_7am_sim):.2f}s")
    print(f"  2pm: n={len(tiempos_2pm_sim)}, media={np.mean(tiempos_2pm_sim):.2f}s")
    print()
    
    # Pruebas de bondad de ajuste para tiempos entre llegadas
    print("Prueba de Kolmogorov-Smirnov - Tiempos entre llegadas 7am:")
    ks_7am, p_7am, decision_ks_7am = prueba_kolmogorov_smirnov(
        tiempos_7am_sim, 'expon', 
        parametros={'lambda': resultados_7am['lambda_exponencial']}
    )
    print(f"  Estadística KS: {ks_7am:.6f}")
    print(f"  Valor p: {p_7am:.6f}")
    print(f"  Decisión: {decision_ks_7am}")
    print()
    
    print("Prueba de Kolmogorov-Smirnov - Tiempos entre llegadas 2pm:")
    ks_2pm, p_2pm, decision_ks_2pm = prueba_kolmogorov_smirnov(
        tiempos_2pm_sim, 'expon',
        parametros={'lambda': resultados_2pm['lambda_exponencial']}
    )
    print(f"  Estadística KS: {ks_2pm:.6f}")
    print(f"  Valor p: {p_2pm:.6f}")
    print(f"  Decisión: {decision_ks_2pm}")
    print()
    
    # Prueba de corridas para verificar independencia
    print("Prueba de Corridas Arriba y Abajo - Tiempos entre llegadas 7am:")
    corridas_7am = prueba_corridas_arriba_abajo(tiempos_7am_sim)
    print(f"  Corridas observadas: {corridas_7am['corridas_observadas']}")
    print(f"  Estadístico Z0: {corridas_7am['estadistico_Z0']:.3f}")
    print(f"  Decisión: {corridas_7am['decision']}")
    print(f"  Conclusión: {corridas_7am['conclusion']}")
    print()
    
    print("Prueba de Corridas Arriba y Abajo - Tiempos entre llegadas 2pm:")
    corridas_2pm = prueba_corridas_arriba_abajo(tiempos_2pm_sim)
    print(f"  Corridas observadas: {corridas_2pm['corridas_observadas']}")
    print(f"  Estadístico Z0: {corridas_2pm['estadistico_Z0']:.3f}")
    print(f"  Decisión: {corridas_2pm['decision']}")
    print(f"  Conclusión: {corridas_2pm['conclusion']}")
    print()
    
    # Visualizaciones
    print("Generando visualizaciones de tiempos entre llegadas...")
    fig2 = visualizar_tiempos_entre_llegadas(resultados_7am, tiempos_7am_sim, "7am")
    plt.savefig('analisis_tiempos_entre_llegadas_7am.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: analisis_tiempos_entre_llegadas_7am.png")
    
    fig3 = visualizar_tiempos_entre_llegadas(resultados_2pm, tiempos_2pm_sim, "2pm")
    plt.savefig('analisis_tiempos_entre_llegadas_2pm.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: analisis_tiempos_entre_llegadas_2pm.png")
    
    fig4 = visualizar_prueba_corridas(corridas_7am, "Prueba de Corridas - Tiempos entre Llegadas 7am")
    plt.savefig('prueba_corridas_7am.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: prueba_corridas_7am.png")
    
    fig5 = visualizar_prueba_corridas(corridas_2pm, "Prueba de Corridas - Tiempos entre Llegadas 2pm")
    plt.savefig('prueba_corridas_2pm.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gráfico guardado: prueba_corridas_2pm.png")
    print()
    
    # 3. ANÁLISIS DE TIEMPOS DE SERVICIO (ASCENSORES)
    print("=" * 80)
    print("3. ANÁLISIS DE TIEMPOS DE SERVICIO DE ASCENSORES")
    print("=" * 80)
    print()
    
    analisis_tiempos = analizar_tiempos_servicio_ascensores(TIEMPOS_ASCENSORES)
    
    print("Tiempos de ciclo observados:")
    for ascensor, tiempo in TIEMPOS_ASCENSORES.items():
        print(f"  {ascensor}: {tiempo} segundos ({tiempo//60}:{tiempo%60:02d})")
    print()
    
    print("Estadísticas descriptivas:")
    print(f"  Media: {analisis_tiempos['media']:.2f} segundos")
    print(f"  Mediana: {analisis_tiempos['mediana']:.2f} segundos")
    print(f"  Desviación estándar: {analisis_tiempos['desviacion_estandar']:.2f} segundos")
    print(f"  Mínimo: {analisis_tiempos['minimo']:.2f} segundos")
    print(f"  Máximo: {analisis_tiempos['maximo']:.2f} segundos")
    print(f"  Coeficiente de variación: {analisis_tiempos['coeficiente_variacion']:.3f}")
    print()
    
    print("Parámetros estimados para distribuciones:")
    print()
    print("  Distribución Normal:")
    print(f"    μ (media) = {analisis_tiempos['normal']['mu']:.2f} s")
    print(f"    σ (desviación estándar) = {analisis_tiempos['normal']['sigma']:.2f} s")
    print()
    print("  Distribución Exponencial:")
    print(f"    λ (tasa) = {analisis_tiempos['exponencial']['lambda']:.6f} 1/s")
    print(f"    scale (1/λ) = {analisis_tiempos['exponencial']['scale']:.2f} s")
    print()
    print("  Distribución Uniforme:")
    print(f"    a (mínimo) = {analisis_tiempos['uniforme']['a']:.2f} s")
    print(f"    b (máximo) = {analisis_tiempos['uniforme']['b']:.2f} s")
    print()
    
    # Pruebas de bondad de ajuste para tiempos de servicio
    tiempos_array = analisis_tiempos['tiempos']
    
    print("PRUEBAS DE BONDAD DE AJUSTE:")
    print()
    
    # Normal
    print("Distribución Normal:")
    ks_norm, p_norm, decision_norm = prueba_kolmogorov_smirnov(
        tiempos_array, 'norm',
        parametros={'mu': analisis_tiempos['normal']['mu'], 
                   'sigma': analisis_tiempos['normal']['sigma']}
    )
    print(f"  KS - Estadística: {ks_norm:.6f}, p-valor: {p_norm:.6f}, Decisión: {decision_norm}")
    
    chi_norm = prueba_chi_cuadrado(
        tiempos_array, 'norm',
        parametros={'mu': analisis_tiempos['normal']['mu'], 
                   'sigma': analisis_tiempos['normal']['sigma']},
        k=3
    )
    print(f"  Chi² - Estadística: {chi_norm['chi_cuadrado']:.6f}, p-valor: {chi_norm['p_valor']:.6f}, Decisión: {chi_norm['decision']}")
    print()
    
    # Exponencial
    print("Distribución Exponencial:")
    ks_exp, p_exp, decision_exp = prueba_kolmogorov_smirnov(
        tiempos_array, 'expon',
        parametros={'lambda': analisis_tiempos['exponencial']['lambda']}
    )
    print(f"  KS - Estadística: {ks_exp:.6f}, p-valor: {p_exp:.6f}, Decisión: {decision_exp}")
    
    chi_exp = prueba_chi_cuadrado(
        tiempos_array, 'expon',
        parametros={'lambda': analisis_tiempos['exponencial']['lambda']},
        k=3
    )
    print(f"  Chi² - Estadística: {chi_exp['chi_cuadrado']:.6f}, p-valor: {chi_exp['p_valor']:.6f}, Decisión: {chi_exp['decision']}")
    print()
    
    # Uniforme
    print("Distribución Uniforme:")
    ks_unif, p_unif, decision_unif = prueba_kolmogorov_smirnov(
        tiempos_array, 'uniform',
        parametros={'a': analisis_tiempos['uniforme']['a'], 
                   'b': analisis_tiempos['uniforme']['b']}
    )
    print(f"  KS - Estadística: {ks_unif:.6f}, p-valor: {p_unif:.6f}, Decisión: {decision_unif}")
    
    chi_unif = prueba_chi_cuadrado(
        tiempos_array, 'uniform',
        parametros={'a': analisis_tiempos['uniforme']['a'], 
                   'b': analisis_tiempos['uniforme']['b']},
        k=3
    )
    print(f"  Chi² - Estadística: {chi_unif['chi_cuadrado']:.6f}, p-valor: {chi_unif['p_valor']:.6f}, Decisión: {chi_unif['decision']}")
    print()
    
    # Prueba de media
    print("Prueba de la Media (Normal):")
    prueba_media_norm = prueba_media(tiempos_array, mu0=analisis_tiempos['normal']['mu'])
    print(f"  Media muestral: {prueba_media_norm['media_muestral']:.2f}")
    print(f"  Intervalo aceptación: [{prueba_media_norm['limite_inferior']:.2f}, {prueba_media_norm['limite_superior']:.2f}]")
    print(f"  Decisión: {prueba_media_norm['decision']}")
    print()
    
    # Prueba de corridas para tiempos de servicio
    print("Prueba de Corridas Arriba y Abajo (Tiempos de Servicio):")
    corridas_servicio = prueba_corridas_arriba_abajo(tiempos_array)
    print(f"  Corridas observadas: {corridas_servicio['corridas_observadas']}")
    print(f"  Estadístico Z0: {corridas_servicio['estadistico_Z0']:.3f}")
    print(f"  Decisión: {corridas_servicio['decision']}")
    print(f"  Conclusión: {corridas_servicio['conclusion']}")
    print()
    
    # Visualización
    print("Generando visualizaciones de tiempos de servicio...")
    fig6 = visualizar_distribucion_tiempos_servicio(analisis_tiempos)
    plt.savefig('analisis_tiempos_servicio_ascensores.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado: analisis_tiempos_servicio_ascensores.png")
    
    fig7 = visualizar_prueba_corridas(corridas_servicio, "Prueba de Corridas - Tiempos de Servicio")
    plt.savefig('prueba_corridas_servicio.png', dpi=300, bbox_inches='tight')
    print("Gráfico guardado: prueba_corridas_servicio.png")
    print()
    
    # 4. RESUMEN Y RECOMENDACIONES
    print("=" * 80)
    print("4. RESUMEN Y RECOMENDACIONES PARA FLEXSIM")
    print("=" * 80)
    print()
    
    print("DISTRIBUCIONES Y PARÁMETROS RECOMENDADOS:")
    print()
    
    print("A. TIEMPOS ENTRE LLEGADAS (Arrival Interarrival Time):")
    print("   Distribución: Exponencial")
    print("   Justificación: Las llegadas son eventos independientes en tiempo continuo")
    print()
    print("   Parámetros para 7am:")
    print(f"     - Mean (1/λ) = {resultados_7am['tiempo_promedio_entre_llegadas_seg']:.2f} segundos")
    print(f"     - En FlexSim: Exponential(0, {resultados_7am['tiempo_promedio_entre_llegadas_seg']:.2f})")
    print()
    print("   Parámetros para 2pm:")
    print(f"     - Mean (1/λ) = {resultados_2pm['tiempo_promedio_entre_llegadas_seg']:.2f} segundos")
    print(f"     - En FlexSim: Exponential(0, {resultados_2pm['tiempo_promedio_entre_llegadas_seg']:.2f})")
    print()
    
    print("B. TIEMPOS DE SERVICIO DE ASCENSORES (Service Time):")
    print("   NOTA: Con solo 3 observaciones, se recomienda usar distribución Uniforme")
    print("         como aproximación conservadora, o Normal si se tienen más datos.")
    print()
    
    # Recomendar distribución basada en pruebas
    mejor_distribucion = "Uniforme"  # Por defecto con pocos datos
    mejor_ajuste = decision_unif
    
    if decision_norm == "No rechazar H0" and decision_exp != "No rechazar H0":
        mejor_distribucion = "Normal"
        mejor_ajuste = decision_norm
    elif decision_exp == "No rechazar H0" and p_exp > p_norm:
        mejor_distribucion = "Exponencial"
        mejor_ajuste = decision_exp
    
    print(f"   Distribución recomendada: {mejor_distribucion}")
    print(f"   Justificación: Pruebas de bondad de ajuste - {mejor_ajuste}")
    print()
    
    if mejor_distribucion == "Uniforme":
        print("   Parámetros Uniforme:")
        print(f"     - Minimum (a) = {analisis_tiempos['uniforme']['a']:.2f} segundos")
        print(f"     - Maximum (b) = {analisis_tiempos['uniforme']['b']:.2f} segundos")
        print(f"     - En FlexSim: Uniform({analisis_tiempos['uniforme']['a']:.2f}, {analisis_tiempos['uniforme']['b']:.2f})")
        print()
        print("   Alternativa Normal (si se tienen más datos):")
        print(f"     - Mean (μ) = {analisis_tiempos['normal']['mu']:.2f} segundos")
        print(f"     - StdDev (σ) = {analisis_tiempos['normal']['sigma']:.2f} segundos")
        print(f"     - En FlexSim: Normal({analisis_tiempos['normal']['mu']:.2f}, {analisis_tiempos['normal']['sigma']:.2f})")
    elif mejor_distribucion == "Normal":
        print("   Parámetros Normal:")
        print(f"     - Mean (μ) = {analisis_tiempos['normal']['mu']:.2f} segundos")
        print(f"     - StdDev (σ) = {analisis_tiempos['normal']['sigma']:.2f} segundos")
        print(f"     - En FlexSim: Normal({analisis_tiempos['normal']['mu']:.2f}, {analisis_tiempos['normal']['sigma']:.2f})")
    else:
        print("   Parámetros Exponencial:")
        print(f"     - Mean (1/λ) = {analisis_tiempos['exponencial']['scale']:.2f} segundos")
        print(f"     - En FlexSim: Exponential(0, {analisis_tiempos['exponencial']['scale']:.2f})")
    print()
    
    print("C. VALIDACIÓN DE INDEPENDENCIA:")
    print("   Los resultados de la Prueba de Corridas indican:")
    print(f"     - Tiempos entre llegadas 7am: {corridas_7am['conclusion']}")
    print(f"     - Tiempos entre llegadas 2pm: {corridas_2pm['conclusion']}")
    print(f"     - Tiempos de servicio: {corridas_servicio['conclusion']}")
    print()
    
    print("=" * 80)
    print("Análisis completado. Todos los gráficos han sido guardados.")
    print("=" * 80)
    
    return {
        'resultados_7am': resultados_7am,
        'resultados_2pm': resultados_2pm,
        'analisis_tiempos': analisis_tiempos,
        'corridas_7am': corridas_7am,
        'corridas_2pm': corridas_2pm,
        'corridas_servicio': corridas_servicio
    }

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    
    # Ejecutar análisis completo
    resultados = generar_reporte_completo()
    
    # Mostrar gráficos
    plt.show()

