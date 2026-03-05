#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:31:13 2024

@author: llopez
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer
import pandas as pd
# import math as m
from scipy.interpolate import interp1d
from scipy.signal import periodogram
# import my_functions
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from SALib.analyze import sobol
from SALib.sample import saltelli
import pickle
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal
import math as m


# =============================================================================

def calculate_beta_combined(beta_0, t, k_imh, k_temp, k_mosquito, alpha,A,B):
    """
    Calculates the transmission rate (beta_mod) based on temperature, IMH, and mosquito density
    using a modified logistic function.

    Parameters:
    - beta_0: Minimum transmission rate.
    - t: Current time.
    - k_imh: Slope of the logistic function for IMH.
    - k_temp: Slope of the logistic function for temperature.
    - k_mosquito: Slope of the logistic function for mosquito density.
    - alpha: Modulation factor.

    Returns:
    - beta_mod: Transmission rate calculated based on temperature, IMH, and mosquito density.
    """
    # Restar 30 días al valor de t, asegurándote de que no sea menor que 0
    tc=t
    tm=t
    
    tc -= 21
    if tc < 0:
        tc = TimeMax-tc
     
    tm -= 7
    if tm < 0:
        tm = TimeMax-tm
    
    # Adjust t for 30-day lag
    t_imh_temp = tc
    
    # Adjust t for 14-day lag
    t_mosquito = tm

    # Interpolate values for IMH and temperature at the lagged time point
    coefficients_imh = coefficients_df.iloc[1].values
    imh_t = polynomial_function(t_imh_temp, *coefficients_imh)
    
    coefficients_temp = coefficients_df.iloc[2].values
    temp_t = polynomial_function(t_imh_temp, *coefficients_temp)
    
    # Interpolate value for mosquito density at the lagged time point
    coefficients_mosquito = coefficients_df.iloc[0].values
    mosquito_density_t = polynomial_function(t_mosquito, *coefficients_mosquito)
    
    # exponential_component0 = (A * np.exp(-k_imh * t)) +  ((1-A) * np.exp(-k_temp * t)) 
    logistic_component0 = A / (B + np.exp(-k_imh *  t)) + ((1-A)) / (B + np.exp(-k_temp *  t)) 
    
    # exponential_component1 = A * np.exp(-k_mosquito * t)
    logistic_component1 = A / (B + np.exp(-k_mosquito *  t))
    
    # beta_mod = beta_0 * ((logistic_component0) +( logistic_component1))
    beta_mod = beta_0 * ((imh_t + temp_t ) +  mosquito_density_t)
    # Calculate beta_mod using a modified logistic function
    # beta_mod = beta_0 + (A / (1 + np.exp(-k_imh * imh_t))) \
    #     + (B / (1 + np.exp(-k_temp * temp_t))) \
    #         + (A / (1 + np.exp(-k_mosquito * mosquito_density_t )))
    # beta_mod = beta_0 + (A  + np.exp(-k_imh * imh_t)) + \
    #     (B + np.exp(-k_temp * temp_t)) \
    #         + (A + np.exp(-k_mosquito * mosquito_density_t ))

    return beta_mod


# =============================================================================

def calculate_beta(beta_0, t, k_imh, k_temp,A,B):
    """
    Calcula la tasa de transmisión (beta_mod) en función de la humedad (IMH)
    y la temperatura utilizando una función logística modificada.

    Parámetros:
    - beta_0: Tasa de transmisión mínima.
    - t: Instante de tiempo actual. Un valor real entre 0 y n.
    - k_imh: Pendiente de la función logística para la humedad.
    - k_temp: Pendiente de la función logística para la temperatura.

    Retorna:
    - beta_mod: Tasa de transmisión calculada en función de la humedad y la temperatura.
    """
    
    # Restar 30 días al valor de t, asegurándote de que no sea menor que 0
    t -= 21
    if t < 0:
        t = TimeMax-t
    
    
    
    coefficients0 = coefficients_df.iloc[1].values
    imh_t = polynomial_function(t, *coefficients0)
    coefficients1 = coefficients_df.iloc[2].values
    temp_t = polynomial_function(t, *coefficients1)

    # exponential_component = (A * np.exp(-k_imh * t)) +  ((1-A) * np.exp(-k_temp * t)) 
    logistic_component = A / (B + np.exp(-k_imh  * t)) + ((1-A)) / (B + np.exp(-k_temp  * t)) 
    
    # beta_mod = beta_0 *( logistic_component)
    beta_mod = beta_0 *( (imh_t + temp_t))
    # beta_mod = beta_0 + (A / (1 + np.exp(-k_imh * imh_t))) + (B / (1 + np.exp(-k_temp * temp_t)))

    # beta_mod = beta_0 + (A1 + np.exp(-k_imh * imh_t)) + (B + np.exp(-k_temp * temp_t))

    return beta_mod


# =============================================================================
def calculate_beta_mosquito(beta_0, t, k_mosquito, alpha,A,B):
    """
    Calcula la tasa de transmisión (beta_mod) en función de la densidad de mosquitos
    utilizando una función logística modificada.

    Parámetros:
    - beta_0: Tasa de transmisión mínima.
    - t: Instante de tiempo actual. Puede ser un objeto datetime o un número entero que representa días desde el inicio.
    - k_mosquito: Pendiente de la función logística para la densidad de mosquitos.
    - alpha: Factor de modulación.

    Retorna:
    - beta_mod: Tasa de transmisión calculada en función de la densidad de mosquitos.
    """
    # Restar 30 días al valor de t, asegurándote de que no sea menor que 0
    t -= 7
    if t < 0:
        t = TimeMax-t
    # Obtener la densidad de mosquitos en el instante de tiempo t mediante interpolación
    coefficients = coefficients_df.iloc[0].values
    mosquito_density_t = polynomial_function(t, *coefficients)
    
    # exponential_component = A * np.exp(-k_mosquito * t)
    logistic_component = A / (B + np.exp(-k_mosquito * t))
    
    # beta_mod = beta_0 * (logistic_component)
    beta_mod = beta_0 * (mosquito_density_t)
    
    # Calcula beta_mod utilizando la función logística modificada
    # beta_mod = beta_0 + (A / (1 +np.exp(-k_mosquito * mosquito_density_t )))

    # beta_mod = beta_0 + (A +np.exp(-k_mosquito * mosquito_density_t ))

    return beta_mod


# =============================================================================
def modelNM(y, t, param):
    """
    Modelo de dengue de dos cepas con efecto ADE e inmunidad cruzada.

    Parámetros
    ----------
    y : array
        Condición inicial.
    t : array float
        Tiempo.
    param : dict
        beta_mh1_0, beta_mh2_0 = Tasa de transmisión de mosquitos a humanos para cada cepa
        delta = Tasa de progresión a la enfermedad sintomática en humanos
        gamma = Tasa de recuperación para los humanos
        omega = Tasa de mortalidad natural (Humanos)
        sigma = Tasa de infección secundaria (ADE)
        mu = Tasa de muerte por enfermedad en humanos
        nu = Tasa de desarrollo de inmunidad cruzada
        rho = Tasa de pérdida de inmunidad cruzada
        A, B = Parámetros climáticos
        k_imh, k_temp, k_mosquito, alpha = Parámetros para diferentes escenarios de transmisión

    Scenario : str
        Escenario específico para calcular beta_mh1 y beta_mh2 ('Constant', 'No Clima', 'Clima', 'Mosquito', 'Combined')

    Retorna
    -------
    list
        Dinámica del sistema para la población humana con dos cepas de dengue con ADE e inmunidad cruzada.

    Compartimentos
    --------------
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D :
    Susceptibles a ambas cepas, Expuestos, Infectados, Asintomáticos, Cross Immunity,
    Susceptibles a cepa 1, Susceptibles a cepa 2,
    Expuesto cepa 1 por previa infección por cepa 2, Expuesto cepa 2 por previa infección por cepa 1,
    Asintomático cepa 1 por previa infección por cepa 2, Asintomático cepa 2 por previa infección por cepa 1,
    Infectado cepa 1 por previa infección por cepa 2, Infectado cepa 2 por previa infección por cepa 1,
    Recuperados totales (los que cursaron ambas infecciones), Población fallecida por infecciones.

    Descripción
    -----------
    - S_both: Población susceptible a ambas cepas.
    - E1, E2: Poblaciones expuestas a la cepa 1 y cepa 2, respectivamente.
    - I1, I2: Poblaciones infectadas por la cepa 1 y cepa 2, respectivamente.
    - A1, A2: Poblaciones asintomáticas por la cepa 1 y cepa 2, respectivamente.
    - CI: Población con inmunidad cruzada.
    - S_1: Población susceptible a cepa 1 por previa infección a cepa 2.
    - S_2: Población susceptible a cepa 2 por previa infección a cepa 1.
    - E_12: Expuesto cepa 1 previa infección por cepa 2.
    - E_21: Expuesto cepa 2 previa infección por cepa 1.
    - A_12: Asintomático cepa 1 previa infección por cepa 2.
    - A_21: Asintomático cepa 2 previa infección por cepa 1.
    - I_12: Infectado cepa 1 previa infección por cepa 2.
    - I_21: Infectado cepa 2 previa infección por cepa 1.
    - R: Recuperados totales (los que cursaron ambas infecciones).
    - D: Población fallecida por infecciones (con aporte de todos los compartimentos de infectados sintomáticos).
    """
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
    seasonal_period = 30 * 6
    annual_period = 365 * 1
    A = param['A']
    B = param['B']

    if Scenario == 'Constant':        
        beta_mh1 =  param['beta_mh1_0']
        beta_mh2 =  param['beta_mh2_0']
        
    if Scenario == 'No Clima':        
        beta_mh1 =  param['beta_mh1_0'] * (1 + A * np.sin(2 * np.pi * t / seasonal_period) + 
                                           B * np.sin(2 * np.pi * t / annual_period))
        beta_mh2 = param['beta_mh2_0'] * (1 + A * np.sin(2 * np.pi * t / seasonal_period) + 
                                          B * np.sin(2 * np.pi * t / annual_period))

    if Scenario == 'Clima':
        k_imh = param['k_imh']
        k_temp = param['k_temp']
        beta_mh1 = calculate_beta(param['beta_mh1_0'], t, k_imh, k_temp, A, B)
        beta_mh2 = calculate_beta(param['beta_mh2_0'], t, k_imh, k_temp, A, B)

    if Scenario == 'Mosquito':
        k_mosquito = param['k_mosquito']
        alpha = param['alpha']
        beta_mh1 = calculate_beta_mosquito(param['beta_mh1_0'], t, k_mosquito, alpha, A, B)
        beta_mh2 = calculate_beta_mosquito(param['beta_mh2_0'], t, k_mosquito, alpha, A, B)

    if Scenario == 'Combined':
        k_imh = param['k_imh']
        k_temp = param['k_temp']
        k_mosquito = param['k_mosquito']
        alpha = param['alpha']
        beta_mh1 = calculate_beta_combined(param['beta_mh1_0'], t, k_imh, k_temp, k_mosquito, alpha, A, B)
        beta_mh2 = calculate_beta_combined(param['beta_mh2_0'], t, k_imh, k_temp, k_mosquito, alpha, A, B)
        
    delta = param['delta']
    gamma = param['gamma']
    omega = param['omega']
    sigma = param['sigma']
    mu = param['mu']
    nu = param['nu']
    rho = param['rho']
    
    # Número total de individuos
    N = S_both + E1 + E2 + I1 + I2 + A1 + A2 + CI + S_1 + S_2 + E_12 + E_21 + A_12 + A_21 + I_12 + I_21 + R + D
    
    # Proporciones de CI que pasan a S_1 y S_2
    CI_to_S_1 = rho * CI * (I2 + A2) / (I1 + A1 + I2 + A2) if (I1 + A1 + I2 + A2) != 0 else 0
    CI_to_S_2 = rho * CI * (I1 + A1) / (I1 + A1 + I2 + A2) if (I1 + A1 + I2 + A2) != 0 else 0
    
    # Ecuaciones del modelo
    dS_both_dt = omega * S_both - (beta_mh1 * S_both * (I1 + A1 + I_12 + A_12) + beta_mh2 * S_both * (I2 + A2 + I_21 + A_21)) / N \
        - omega * S_both
    dE1_dt = (beta_mh1 * S_both * (I1 + A1) + beta_mh1 * S_1 * (I1 + A1)) / N - delta * E1 - omega * E1
    dE2_dt = (beta_mh2 * S_both * (I2 + A2) + beta_mh2 * S_2 * (I2 + A2)) / N - delta * E2 - omega * E2
    dI1_dt = delta * E1 - (gamma + mu) * I1 - nu*I1
    dI2_dt = delta * E2 - (gamma + mu) * I2 - nu*I2
    dA1_dt = (1 - delta) * E1 - omega * A1 - nu * A1
    dA2_dt = (1 - delta) * E2 - omega * A2 - nu * A2
    dCI_dt = nu * (A1 + A2 +I1+I2 ) - rho * CI - omega * CI
    dS_1_dt = gamma * I2  - (beta_mh1 * (1 + sigma) * S_1 * (I1 + A1 + A_12)) / N - omega * S_1 + CI_to_S_1
    dS_2_dt = gamma * I1  - (beta_mh2 * (1 + sigma) * S_2 * (I2 + A2 + A_21)) / N - omega * S_2 + CI_to_S_2
    dE_12_dt = (beta_mh1 * (1 + sigma) * S_1 * (I1 + A1)) / N - delta * E_12 - omega * E_12
    dE_21_dt = (beta_mh2 * (1 + sigma) * S_2 * (I2 + A2)) / N - delta * E_21 - omega * E_21
    dA_12_dt = (1 - delta) * E_12 - omega * A_12 - sigma * A_12
    dA_21_dt = (1 - delta) * E_21 - omega * A_21 - sigma * A_21
    dI_12_dt = delta * E_12 - (gamma + mu) * I_12
    dI_21_dt = delta * E_21 - (gamma + mu) * I_21
    dR_dt = gamma * (I_12 + I_21) + (1 - delta) * (A_12 + A_21) - omega * R
    dD_dt = mu * (I1 + I2 + I_12 + I_21)
    
    return [dS_both_dt, dE1_dt, dE2_dt, dI1_dt, dI2_dt, dA1_dt, dA2_dt, dCI_dt, dS_1_dt, dS_2_dt,
            dE_12_dt, dE_21_dt, dA_12_dt, dA_21_dt, dI_12_dt, dI_21_dt, dR_dt, dD_dt]


# =============================================================================

# def modelNM(y, t, param):
#     """
#     Modelo de dengue de dos cepas con efecto ADE e inmunidad cruzada.

#     Parámetros
#     ----------
#     y : array
#         Condición inicial.
#     t : array float
#         Tiempo.
#     param : Array
#         beta_mh1, beta_mh2 = Tasa de transmisión de mosquitos a humanos para cada cepa
#         delta = Tasa de progresión a la enfermedad sintomática en humanos
#         gamma = Tasa de recuperación para los humanos
#         omega = Tasa de mortalidad natural (Humanos)
#         sigma = Tasa de infección secundaria (ADE)
#         mu = Tasa de muerte por enfermedad en humanos
#         nu = Tasa de desarrollo de inmunidad cruzada
#         rho = Tasa de pérdida de inmunidad cruzada

#     Retorna
#     -------
#     list
#         Dinámica del sistema para la población humana con dos cepas de dengue con ADE e inmunidad cruzada.

#     Compartimentos
#     --------------
#     S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D :
#     Susceptibles a ambas cepas, Expuestos, Infectados, Asintomáticos, Cross Immunity,
#     Susceptibles a cepa 1, Susceptibles a cepa 2,
#     Expuesto cepa 1 por previa infección por cepa 2, Expuesto cepa 2 por previa infección por cepa 1,
#     Asintomático cepa 1 por previa infección por cepa 2, Asintomático cepa 2 por previa infección por cepa 1,
#     Infectado cepa 1 por previa infección por cepa 2, Infectado cepa 2 por previa infección por cepa 1,
#     Recuperados totales (los que cursaron ambas infecciones), Población fallecida por infecciones.

#     Descripción
#     -----------
#     - S_both: Población susceptible a ambas cepas.
#     - E1, E2: Poblaciones expuestas a la cepa 1 y cepa 2, respectivamente.
#     - I1, I2: Poblaciones infectadas por la cepa 1 y cepa 2, respectivamente.
#     - A1, A2: Poblaciones asintomáticas por la cepa 1 y cepa 2, respectivamente.
#     - CI: Población con inmunidad cruzada.
#     - S_1: Población susceptible a cepa 1 por previa infección a cepa 2.
#     - S_2: Población susceptible a cepa 2 por previa infección a cepa 1.
#     - E_12: Expuesto cepa 1 previa infección por cepa 2.
#     - E_21: Expuesto cepa 2 previa infección por cepa 1.
#     - A_12: Asintomático cepa 1 previa infección por cepa 2.
#     - A_21: Asintomático cepa 2 previa infección por cepa 1.
#     - I_12: Infectado cepa 1 previa infección por cepa 2.
#     - I_21: Infectado cepa 2 previa infección por cepa 1.
#     - R: Recuperados totales (los que cursaron ambas infecciones).
#     - D: Población fallecida por infecciones (con aporte de todos los compartimentos de infectados sintomáticos).
#     """
#     S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
#     seasonal_period=30*6
#     annual_period=365*1
#     A=param['A'].value
#     B=param['B'].value
#     # A=0
#     # B=1
#     if Scenario == 'Constant':        
#         beta_mh1 =  param['beta_mh1_0'].value
#         beta_mh2 =  param['beta_mh2_0'].value
        
#     if Scenario == 'No Clima':        
#         beta_mh1 =  param['beta_mh1_0'].value* (1 + A* np.sin(2 * np.pi * t / seasonal_period) + 
#                                                 B*np.sin(2 * np.pi * t / annual_period))
#         beta_mh2 = param['beta_mh2_0'].value*(1 +  A* np.sin(2 * np.pi * t / seasonal_period) + 
#                                               B*np.sin(2 * np.pi * t / annual_period))

#     if Scenario == 'Clima':
#         k_imh= param['k_imh'].value
#         k_temp= param['k_temp'].value
#         # imh = 
#         beta_mh1 = calculate_beta(param['beta_mh1_0'].value, 
#                                   t, k_imh, k_temp,A,B)
#         beta_mh2 = calculate_beta(param['beta_mh2_0'].value, 
#                                   t, k_imh, k_temp,A,B)
#     if Scenario == 'Mosquito':
#         k_mosquito =param['k_mosquito'].value
#         alpha=param['alpha'].value
#         beta_mh1 = calculate_beta_mosquito(param['beta_mh1_0'].value, 
#                                            t, k_mosquito,alpha,A,B)
        
#         beta_mh2 = calculate_beta_mosquito(param['beta_mh2_0'].value, 
#                                            t, k_mosquito,alpha,A,B)
#         # print('BETA: ' , beta_mh1)
#     if Scenario == 'Combined':
#         k_imh= param['k_imh'].value
#         k_temp= param['k_temp'].value
#         k_mosquito =param['k_mosquito'].value
#         alpha=param['alpha'].value
#         beta_mh1 = calculate_beta_combined(param['beta_mh1_0'].value, 
#                                            t, k_imh, k_temp, k_mosquito,alpha,A,B)
#         beta_mh2 = calculate_beta_combined(param['beta_mh2_0'].value, 
#                                            t, k_imh, k_temp, k_mosquito,alpha,A,B)
#         # beta_mh1 = calculate_beta_combined(beta_0, t, k_imh, k_temp, k_mosquito, alpha)
        
#     delta = param['delta'].value
#     gamma = param['gamma'].value
#     omega = param['omega'].value
#     sigma = param['sigma'].value
#     mu = param['mu'].value
#     nu = param['nu'].value
#     rho = param['rho'].value
    
#     # Número total de individuos
#     N = S_both + E1 + E2 + I1 + I2 + A1 + A2 + CI + S_1 + S_2 + E_12 + E_21 + A_12 + A_21 + I_12 + I_21 + R + D
    
#     # Ecuaciones del modelo
#     dS_both_dt = omega * S_both- (beta_mh1 * S_both * I1 + beta_mh2 * S_both * I2) / N - omega * S_both
#     dE1_dt = (beta_mh1 * S_both * I1 + beta_mh1 * S_1 * I1) / N - delta * E1 - omega * E1
#     dE2_dt = (beta_mh2 * S_both * I2 + beta_mh2 * S_2 * I2) / N - delta * E2 - omega * E2
#     dI1_dt = delta * E1 - (gamma + mu) * I1
#     dI2_dt = delta * E2 - (gamma + mu) * I2
#     dA1_dt = (1 - delta) * E1 - omega * A1 - sigma * A1
#     dA2_dt = (1 - delta) * E2 - omega * A2 - sigma * A2
#     dCI_dt = sigma * (A1 + A2) + nu * (I_12 + I_21) - rho * CI - omega * CI
#     dS_1_dt = gamma * I_12 - (beta_mh1 * S_1 * I1) / N - omega * S_1
#     dS_2_dt = gamma * I_21 - (beta_mh2 * S_2 * I2) / N - omega * S_2
#     dE_12_dt = (beta_mh1 * S_1 * I1) / N - delta * E_12 - omega * E_12
#     dE_21_dt = (beta_mh2 * S_2 * I2) / N - delta * E_21 - omega * E_21
#     dA_12_dt = (1 - delta) * E_12 - omega * A_12 - sigma * A_12
#     dA_21_dt = (1 - delta) * E_21 - omega * A_21 - sigma * A_21
#     dI_12_dt = delta * E_12 - (gamma + mu) * I_12
#     dI_21_dt = delta * E_21 - (gamma + mu) * I_21
#     dR_dt = gamma * (I1 + I2) + gamma * (I_12 + I_21) - omega * R
#     dD_dt = mu * (I1 + I2 + I_12 + I_21)
    
#     return [dS_both_dt, dE1_dt, dE2_dt, dI1_dt, dI2_dt, dA1_dt, dA2_dt, dCI_dt, dS_1_dt, dS_2_dt,
#             dE_12_dt, dE_21_dt, dA_12_dt, dA_21_dt, dI_12_dt, dI_21_dt, dR_dt, dD_dt]
# =============================================================================
# =============================================================================
def modelNM_discrete_stochastic(y, t, param):
    """
    Modelo de dengue de dos cepas con efecto ADE e inmunidad cruzada.

    Parámetros
    ----------
    y : array
        Condición inicial.
    t : array float
        Tiempo.
    param : Array
        beta_mh1, beta_mh2 = Tasa de transmisión de mosquitos a humanos para cada cepa
        delta = Tasa de progresión a la enfermedad sintomática en humanos
        gamma = Tasa de recuperación para los humanos
        omega = Tasa de mortalidad natural (Humanos)
        sigma = Tasa de infección secundaria (ADE)
        mu = Tasa de muerte por enfermedad en humanos
        nu = Tasa de desarrollo de inmunidad cruzada
        rho = Tasa de pérdida de inmunidad cruzada

    Retorna
    -------
    list
        Dinámica del sistema para la población humana con dos cepas de dengue con ADE e inmunidad cruzada.

    Compartimentos
    --------------
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D :
    Susceptibles a ambas cepas, Expuestos, Infectados, Asintomáticos, Cross Immunity,
    Susceptibles a cepa 1, Susceptibles a cepa 2,
    Expuesto cepa 1 por previa infección por cepa 2, Expuesto cepa 2 por previa infección por cepa 1,
    Asintomático cepa 1 por previa infección por cepa 2, Asintomático cepa 2 por previa infección por cepa 1,
    Infectado cepa 1 por previa infección por cepa 2, Infectado cepa 2 por previa infección por cepa 1,
    Recuperados totales (los que cursaron ambas infecciones), Población fallecida por infecciones.

    Descripción
    -----------
    - S_both: Población susceptible a ambas cepas.
    - E1, E2: Poblaciones expuestas a la cepa 1 y cepa 2, respectivamente.
    - I1, I2: Poblaciones infectadas por la cepa 1 y cepa 2, respectivamente.
    - A1, A2: Poblaciones asintomáticas por la cepa 1 y cepa 2, respectivamente.
    - CI: Población con inmunidad cruzada.
    - S_1: Población susceptible a cepa 1 por previa infección a cepa 2.
    - S_2: Población susceptible a cepa 2 por previa infección a cepa 1.
    - E_12: Expuesto cepa 1 previa infección por cepa 2.
    - E_21: Expuesto cepa 2 previa infección por cepa 1.
    - A_12: Asintomático cepa 1 previa infección por cepa 2.
    - A_21: Asintomático cepa 2 previa infección por cepa 1.
    - I_12: Infectado cepa 1 previa infección por cepa 2.
    - I_21: Infectado cepa 2 previa infección por cepa 1.
    - R: Recuperados totales (los que cursaron ambas infecciones).
    - D: Población fallecida por infecciones (con aporte de todos los compartimentos de infectados sintomáticos).
    """
    # Recorrer la lista y reemplazar los valores negativos con 0
    y = [max(0, value) for value in y]
    
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
    seasonal_period=30*6
    annual_period=365*1
    A=param['A'].value
    B=param['B'].value
    # A=0
    # B=1
    if Scenario == 'Constant':        
        beta_mh1 =  param['beta_mh1_0'].value
        beta_mh2 =  param['beta_mh2_0'].value
        
    if Scenario == 'No Clima':        
        beta_mh1 =  param['beta_mh1_0'].value* (1 + A* np.sin(2 * np.pi * t / seasonal_period) + 
                                                B*np.sin(2 * np.pi * t / annual_period))
        beta_mh2 = param['beta_mh2_0'].value*(1 +  A* np.sin(2 * np.pi * t / seasonal_period) + 
                                              B*np.sin(2 * np.pi * t / annual_period))

    if Scenario == 'Clima':
        k_imh= param['k_imh'].value
        k_temp= param['k_temp'].value
        # imh = 
        beta_mh1 = calculate_beta(param['beta_mh1_0'].value, 
                                  t, k_imh, k_temp,A,B)
        beta_mh2 = calculate_beta(param['beta_mh2_0'].value, 
                                  t, k_imh, k_temp,A,B)
    if Scenario == 'Mosquito':
        k_mosquito =param['k_mosquito'].value
        alpha=param['alpha'].value
        beta_mh1 = calculate_beta_mosquito(param['beta_mh1_0'].value, 
                                           t, k_mosquito,alpha,A,B)
        
        beta_mh2 = calculate_beta_mosquito(param['beta_mh2_0'].value, 
                                           t, k_mosquito,alpha,A,B)
        # print('BETA: ' , beta_mh1)
    if Scenario == 'Combined':
        k_imh= param['k_imh'].value
        k_temp= param['k_temp'].value
        k_mosquito =param['k_mosquito'].value
        alpha=param['alpha'].value
        beta_mh1 = calculate_beta_combined(param['beta_mh1_0'].value, 
                                           t, k_imh, k_temp, k_mosquito,alpha,A,B)
        beta_mh2 = calculate_beta_combined(param['beta_mh2_0'].value, 
                                           t, k_imh, k_temp, k_mosquito,alpha,A,B)
        # beta_mh1 = calculate_beta_combined(beta_0, t, k_imh, k_temp, k_mosquito, alpha)
        
    delta = param['delta'].value
    gamma = param['gamma'].value
    omega = param['omega'].value
    sigma = param['sigma'].value
    mu = param['mu'].value
    nu = param['nu'].value
    rho = param['rho'].value
    # Número total de individuos
    N = S_both + E1 + E2 + I1 + I2 + A1 + A2 + CI + S_1 + S_2 + E_12 + E_21 + A_12 + A_21 + I_12 + I_21 + R + D
    BetaS1=(1/N)*(beta_mh1*(I1+A1))   
    BetaS2=(1/N)*(beta_mh2*I2+A2) 

    # Probabilidades de Transicion
    NacMuerProb=max(0,(1.0 - m.exp(-omega*dt))) 
    Exp1Prob   =max(0,(1.0 - m.exp(-BetaS1*dt))) 
    Exp2Prob   =max(0,(1.0 - m.exp(-BetaS2*dt))) 
    InfProb    =max(0,(1.0 - m.exp(-delta*dt))) 
    AsymProb   =max(0,(1.0 - m.exp(-(1-delta)*dt))) 
    ADEProb    =max(0,(1.0 - m.exp(-sigma*dt))) 
    CrossIProb =max(0,(1.0 - m.exp(-nu*dt))) 
    LossCIProb =max(0,(1.0 - m.exp(-rho*dt)))
    RecProb    =max(0,(1.0 - m.exp(-gamma*dt))) 
    DeathProb  =max(0,(1.0 - m.exp(-mu*dt))) 

    
    # Ecuaciones del modelo
    # S_both+= omega * S_both - (beta_mh1 * S_both * I1 + beta_mh2 * S_both * I2) / N -  omega * S_both
    # E1    += (beta_mh1 * S_both * I1 + beta_mh1 * S_1 * I1) / N - delta * E1 - omega * E1
    # E2    += (beta_mh2 * S_both * I2 + beta_mh2 * S_2 * I2) / N - delta * E2 - omega * E2
    # I1    += delta * E1 - (gamma + mu) * I1
    # I2    += delta * E2 - (gamma + mu) * I2
    # A1    += (1 - delta) * E1 - omega * A1 - sigma * A1
    # A2    += (1 - delta) * E2 - omega * A2 - sigma * A2
    # CI    += sigma * (A1 + A2) + nu * (I_12 + I_21) - rho * CI - omega * CI
    # S_1   += gamma * I_12 - (beta_mh1 * S_1 * I1) / N - omega * S_1
    # S_2   += gamma * I_21 - (beta_mh2 * S_2 * I2) / N - omega * S_2
    # E_12  += (beta_mh1 * S_1 * I1) / N - delta * E_12 - omega * E_12
    # E_21  += (beta_mh2 * S_2 * I2) / N - delta * E_21 - omega * E_21
    # A_12  += (1 - delta) * E_12 - omega * A_12 - sigma * A_12
    # A_21  += (1 - delta) * E_21 - omega * A_21 - sigma * A_21
    # I_12  += delta * E_12 - (gamma + mu) * I_12
    # I_21  += delta * E_21 - (gamma + mu) * I_21
    # R     += gamma * (I1 + I2) + gamma * (I_12 + I_21) - omega * R
    # D     += mu * (I1 + I2 + I_12 + I_21)
    
    S_both += np.random.binomial(max(0, S_both), NacMuerProb) - \
          np.random.binomial(max(0, S_both), Exp1Prob) - np.random.binomial(max(0, S_both), Exp2Prob) - \
          np.random.binomial(max(0, S_both), NacMuerProb)
   
    E1 += np.random.binomial(max(0, S_both), Exp1Prob) + np.random.binomial(max(0, S_1), Exp1Prob) - \
          np.random.binomial(max(0, E1), InfProb) - np.random.binomial(max(0, E1), NacMuerProb)
        
    E2 += np.random.binomial(max(0, S_both), Exp2Prob) + np.random.binomial(max(0, S_2), Exp2Prob) - \
          np.random.binomial(max(0, E2), InfProb) - np.random.binomial(max(0, E2), NacMuerProb)
        
    I1 += np.random.binomial(max(0, E1), InfProb) - np.random.binomial(max(0, I1), RecProb) - \
          np.random.binomial(max(0, I1), DeathProb)
        
    I2 += np.random.binomial(max(0, E2), InfProb) - np.random.binomial(max(0, I2), RecProb) - \
          np.random.binomial(max(0, I2), DeathProb)
        
    A1 += np.random.binomial(max(0, E1), AsymProb) - np.random.binomial(max(0, A1), NacMuerProb) - \
          np.random.binomial(max(0, A1), ADEProb)
        
    A2 += np.random.binomial(max(0, E2), AsymProb) - np.random.binomial(max(0, A2), NacMuerProb) - \
          np.random.binomial(max(0, A2), ADEProb)
        
    CI += np.random.binomial(max(0, A1), ADEProb) + np.random.binomial(max(0, A2), ADEProb) + \
          np.random.binomial(max(0, I_12), CrossIProb) + np.random.binomial(max(0, I_21), CrossIProb) - \
          np.random.binomial(max(0, CI), LossCIProb) - np.random.binomial(max(0, CI), NacMuerProb)
        
    S_1 += np.random.binomial(max(0, I_12), RecProb) - np.random.binomial(max(0, S_1), Exp1Prob) - \
           np.random.binomial(max(0, S_1), NacMuerProb)
        
    S_2 += np.random.binomial(max(0, I_21), RecProb) - np.random.binomial(max(0, S_2), Exp2Prob) - \
           np.random.binomial(max(0, S_2), NacMuerProb)
        
    E_12 += np.random.binomial(max(0, S_1), Exp1Prob) - np.random.binomial(max(0, E_12), InfProb) - \
            np.random.binomial(max(0, E_12), NacMuerProb)
        
    E_21 += np.random.binomial(max(0, S_2), Exp2Prob) - np.random.binomial(max(0, E_21), InfProb) - \
            np.random.binomial(max(0, E_21), NacMuerProb)
        
    A_12 += np.random.binomial(max(0, E_12), AsymProb) - np.random.binomial(max(0, A_12), NacMuerProb) - \
            np.random.binomial(max(0, A_12), ADEProb)
        
    A_21 += np.random.binomial(max(0, E_21), AsymProb) - np.random.binomial(max(0, A_21), NacMuerProb) - \
            np.random.binomial(max(0, A_21), ADEProb)
        
    I_12 += np.random.binomial(max(0, E_12), InfProb) - np.random.binomial(max(0, I_12), RecProb) - \
            np.random.binomial(max(0, I_12), DeathProb)
        
    I_21 += np.random.binomial(max(0, E_21), InfProb) - np.random.binomial(max(0, I_21), RecProb) - \
            np.random.binomial(max(0, I_21), DeathProb)
        
    R += np.random.binomial(max(0, I1), RecProb) + np.random.binomial(max(0, I2), RecProb) + \
         np.random.binomial(max(0, I_12), RecProb) + np.random.binomial(max(0, I_21), RecProb) - \
         np.random.binomial(max(0, R), NacMuerProb)
        
    D += np.random.binomial(max(0, I1), DeathProb) + np.random.binomial(max(0, I2), DeathProb) + \
         np.random.binomial(max(0, I_12), DeathProb) + np.random.binomial(max(0, I_21), DeathProb)

    
    Y= [S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2,
            E_12, E_21, A_12, A_21, I_12, I_21, R, D]

    return Y
# =============================================================================

# def valid_prob(p):
#     """Ensure the probability is between 0 and 1 and is not NaN."""
#     return np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
# def modelNM_discrete_stochastic(y, t, param):
#     """
#     Modelo estocástico de dengue de dos cepas con efecto ADE e inmunidad cruzada.

#     Parámetros
#     ----------
#     y : array
#         Condición inicial.
#     t : array float
#         Tiempo.
#     param : dict
#         Parámetros del modelo.
#     dt : float
#         Paso de tiempo.
#     Scenario : str
#         Escenario de simulación ('Constant', 'No Clima', 'Clima', 'Mosquito', 'Combined').

#     Retorna
#     -------
#     list
#         Dinámica del sistema para la población humana con dos cepas de dengue con ADE e inmunidad cruzada.

#     """
#     S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
#     seasonal_period = 30 * 6
#     annual_period = 365 * 1
#     A = param['A'].value
#     B = param['B'].value

#     if Scenario == 'Constant':        
#         beta_mh1 = param['beta_mh1_0'].value
#         beta_mh2 = param['beta_mh2_0'].value
        
#     elif Scenario == 'No Clima':        
#         beta_mh1 = param['beta_mh1_0'].value * (1 + A * np.sin(2 * np.pi * t / seasonal_period) + 
#                                           B * np.sin(2 * np.pi * t / annual_period))
#         beta_mh2 = param['beta_mh2_0'].value * (1 + A * np.sin(2 * np.pi * t / seasonal_period) + 
#                                           B * np.sin(2 * np.pi * t / annual_period))

#     elif Scenario == 'Clima':
#         k_imh = param['k_imh'].value
#         k_temp = param['k_temp'].value
#         beta_mh1 = calculate_beta(param['beta_mh1_0'].value, t, k_imh, k_temp, A, B)
#         beta_mh2 = calculate_beta(param['beta_mh2_0'].value, t, k_imh, k_temp, A, B)
        
#     elif Scenario == 'Mosquito':
#         k_mosquito = param['k_mosquito']
#         alpha = param['alpha'].value
#         beta_mh1 = calculate_beta_mosquito(param['beta_mh1_0'].value, t, k_mosquito, alpha, A, B)
#         beta_mh2 = calculate_beta_mosquito(param['beta_mh2_0'].value, t, k_mosquito, alpha, A, B)
        
#     elif Scenario == 'Combined':
#         k_imh = param['k_imh'].value
#         k_temp = param['k_temp'].value
#         k_mosquito = param['k_mosquito'].value
#         alpha = param['alpha'].value
#         beta_mh1 = calculate_beta_combined(param['beta_mh1_0'].value, t, k_imh, k_temp, k_mosquito, alpha, A, B)
#         beta_mh2 = calculate_beta_combined(param['beta_mh2_0'].value, t, k_imh, k_temp, k_mosquito, alpha, A, B)

#     delta = param['delta'].value
#     gamma = param['gamma'].value
#     omega = param['omega'].value
#     sigma = param['sigma'].value
#     mu = param['mu'].value
#     nu = param['nu'].value
#     rho = param['rho'].value
    
#     # Número total de individuos
#     N = S_both + E1 + E2 + I1 + I2 + A1 + A2 + CI + S_1 + S_2 + E_12 + E_21 + A_12 + A_21 + I_12 + I_21 + R + D
    
#     # Asegurarse de que N no sea cero para evitar división por cero
#     if N == 0:
#         N = NH
    
#     # Calcular los parámetros p para las distribuciones binomiales y asegurarse de que estén en el rango [0, 1]
#     # (1.0 - m.exp(-betaw*dt))   
#     Beta1=1/N*beta_mh1 * I1
#     Beta2=1/N*beta_mh2 * I2
#     # p_S_both_to_E1 = min(max(0, (beta_mh1 * I1 / N + rho) * dt), 1)
#     # p_S_both_to_E2 = min(max(0, (beta_mh2 * I2 / N + rho) * dt), 1)
#     p_S_both_to_E1 = min(max(0, m.exp(-rho*dt)-m.exp(-Beta1*dt ) ), 1)
#     p_S_both_to_E2 = min(max(0, m.exp(-rho*dt)-m.exp(-Beta2*dt ) ), 1)
#     p_E1_to_I1 = min(max(0, (1.0 - m.exp(-delta * dt))), 1)
#     p_E2_to_I2 = min(max(0, (1.0 - m.exp(-delta * dt))), 1)
#     p_I1_to_R = min(max(0, (1.0 - m.exp(-gamma * dt))), 1)
#     p_I1_to_D = min(max(0,(1.0 - m.exp(- mu * dt))), 1)
#     p_I2_to_R = min(max(0,(1.0 - m.exp(- gamma * dt))), 1)
#     p_I2_to_D = min(max(0,(1.0 - m.exp(- mu * dt))), 1)
#     p_A1_to_CI = min(max(0, (1.0 - m.exp(-sigma * dt))), 1)
#     p_A2_to_CI = min(max(0, (1.0 - m.exp(-sigma * dt))), 1)
#     p_CI_to_S1 = min(max(0, (1.0 - m.exp(-nu * dt))), 1)
#     p_CI_to_S2 = min(max(0, (1.0 - m.exp(-nu * dt))), 1)
#     # p_S1_to_E12 = min(max(0, (beta_mh1 * I1 / N + rho) * dt), 1)
#     # p_S2_to_E21 = min(max(0, (beta_mh2 * I2 / N + rho) * dt), 1)
#     p_S1_to_E12 = min(max(0,m.exp(-rho*dt)-m.exp(-Beta1*dt)), 1)
#     p_S2_to_E21 = min(max(0, m.exp(-rho*dt)-m.exp(-Beta2*dt ) ), 1)
#     p_E12_to_I12 = min(max(0, (1.0 - m.exp(-delta * dt))), 1)
#     p_E21_to_I21 = min(max(0, (1.0 - m.exp(-delta * dt))), 1)
#     p_I12_to_R = min(max(0, (1.0 - m.exp(-gamma * dt))), 1)
#     p_I12_to_D = min(max(0, (1.0 - m.exp(-mu * dt))), 1)
#     p_I21_to_R = min(max(0, (1.0 - m.exp(-gamma * dt))), 1)
#     p_I21_to_D = min(max(0, (1.0 - m.exp(-mu * dt))), 1)
    
#     # Transiciones estocásticas usando distribuciones binomiales
#     dS_both_to_E1 = np.random.binomial(S_both, p_S_both_to_E1)
#     dS_both_to_E2 = np.random.binomial(S_both, p_S_both_to_E2)
#     dE1_to_I1 = np.random.binomial(E1, p_E1_to_I1)
#     dE2_to_I2 = np.random.binomial(E2, p_E2_to_I2)
#     dI1_to_R = np.random.binomial(I1, p_I1_to_R)
#     dI1_to_D = np.random.binomial(I1, p_I1_to_D)
#     dI2_to_R = np.random.binomial(I2, p_I2_to_R)
#     dI2_to_D = np.random.binomial(I2, p_I2_to_D)
#     dA1_to_CI = np.random.binomial(A1, p_A1_to_CI)
#     dA2_to_CI = np.random.binomial(A2, p_A2_to_CI)
#     dCI_to_S1 = np.random.binomial(CI, p_CI_to_S1)
#     dCI_to_S2 = np.random.binomial(CI, p_CI_to_S2)
#     dS1_to_E12 = np.random.binomial(S_1, p_S1_to_E12)
#     dS2_to_E21 = np.random.binomial(S_2, p_S2_to_E21)
#     dE12_to_I12 = np.random.binomial(E_12, p_E12_to_I12)
#     dE21_to_I21 = np.random.binomial(E_21, p_E21_to_I21)
#     dI12_to_R = np.random.binomial(I_12, p_I12_to_R)
#     dI12_to_D = np.random.binomial(I_12, p_I12_to_D)
#     dI21_to_R = np.random.binomial(I_21, p_I21_to_R)
#     dI21_to_D = np.random.binomial(I_21, p_I21_to_D)
    
#     # Actualización de los compartimentos asegurando no negatividad
#     S_both = max(0, S_both + np.random.poisson(omega * N * dt) - dS_both_to_E1 - dS_both_to_E2 - np.random.binomial(max(0, S_both),  min(max(0, (1.0 - m.exp(-omega * dt))), 1)) )
#     E1 = max(0, E1 + dS_both_to_E1 - dE1_to_I1 - np.random.binomial(max(0, E1), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     E2 = max(0, E2 + dS_both_to_E2 - dE2_to_I2 - np.random.binomial(max(0, E2), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     I1 = max(0, I1 + dE1_to_I1 - dI1_to_R - dI1_to_D - np.random.binomial(max(0, I1), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     I2 = max(0, I2 + dE2_to_I2 - dI2_to_R - dI2_to_D - np.random.binomial(max(0, I2), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     A1 = max(0, A1 + dE1_to_I1 - dA1_to_CI - np.random.binomial(max(0, A1), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     A2 = max(0, A2 + dE2_to_I2 - dA2_to_CI - np.random.binomial(max(0, A2), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     CI = max(0, CI + dA1_to_CI + dA2_to_CI - dCI_to_S1 - dCI_to_S2 - np.random.binomial(max(0, CI), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     S_1 = max(0, S_1 + dCI_to_S1 - dS1_to_E12 - np.random.binomial(max(0, S_1), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     S_2 = max(0, S_2 + dCI_to_S2 - dS2_to_E21 - np.random.binomial(max(0, S_2), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     E_12 = max(0, E_12 + dS1_to_E12 - dE12_to_I12 - np.random.binomial(max(0, E_12), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     E_21 = max(0, E_21 + dS2_to_E21 - dE21_to_I21 - np.random.binomial(max(0, E_21), min(max(0, (1.0 - m.exp(-omega * dt))), 1)))
#     I_12 = max(0, I_12 + dE12_to_I12 - dI12_to_R - dI12_to_D - np.random.binomial(max(0, I_12), min(max(0,(1.0 - m.exp(- omega * dt))), 1)))
#     I_21 = max(0, I_21 + dE21_to_I21 - dI21_to_R - dI21_to_D - np.random.binomial(max(0, I_21), min(max(0,(1.0 - m.exp(- omega * dt))), 1)))
#     R = max(0, R + dI1_to_R + dI2_to_R + dI12_to_R + dI21_to_R)
#     D = max(0, D + dI1_to_D + dI2_to_D + dI12_to_D + dI21_to_D)
    
#     return [S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D]





def simulate_model_stochastic(y0, t, param):
    """
    Simula el modelo estocástico durante un período de tiempo definido por el vector t.

    Parámetros
    ----------
    y0 : array
        Condiciones iniciales del sistema.
    t : array
        Vector de tiempo.
    param : lmfit.Parameters
        Parámetros del modelo definidos con lmfit.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la dinámica del sistema a lo largo del tiempo.
    """
    results = []

    # Inicializar condiciones iniciales
    y = y0[:]
    results.append(y)

    # Simular paso a paso
    for i in range(1, len(t)):
        # dt = t[i] - t[i-1]
        y = modelNM_discrete_stochastic(y, t[i], param)
        results.append(y)

    # Convertir resultados a DataFrame
    columns = ['S_both', 'E1', 'E2', 'I1', 'I2', 'A1', 'A2', 'CI', 'S_1', 'S_2', 'E_12', 'E_21', 'A_12', 'A_21', 'I_12', 'I_21', 'R', 'D']
    df_results = pd.DataFrame(results, columns=columns)

    return df_results


# def simulate_model_stochastic(y0, t, ps):
#     results = np.zeros((len(t), len(y0)))
#     results[0] = y0
#     for i in range(1, len(t)):
#         results[i] = modelNM_discrete_stochastic(results[i-1], t[i], ps)
#     return results

def simulateSC(t, y0, ps):
    # Simula el modelo estocástico
    results = simulate_model_stochastic(y0, t, ps)
    
    # Estructura de datos para el resultado
    return {
        't': t,
        'S_both': results['S_both'],
        'E1': results['E1'],
        'E2': results['E2'],
        'I1': results['I1'],
        'I2': results['I2'],
        'A1': results['A1'],
        'A2': results['A2'],
        'CI': results['CI'],
        'S_1': results['S_1'],
        'S_2': results['S_2'],
        'E_12': results['E_12'],
        'E_21': results['E_21'],
        'A_12': results['A_12'],
        'A_21': results['A_21'],
        'I_12': results['I_12'],
        'I_21': results['I_21'],
        'R': results['R'],
        'D': results['D']
    }

# =============================================================================

# # Time-stepping loop to simulate the model
# def simulate_model_stochastic(y0, t, param, scenario='Constant'):
#     # Create an array to store the results
#     results = np.zeros((len(t), len(y0)))
#     results[0] = y0

#     for i in range(1, len(t)):
#         # Get the new state using the discrete stochastic model
#         results[i] = modelNM_discrete_stochastic(results[i-1], t[i], param, scenario)
    
#     return results
# =============================================================================
def modelNMSA(y, t, ps):
    """
    Modelo de dengue de dos cepas con efecto ADE e inmunidad cruzada.

    Parámetros
    ----------
    y : array
        Condición inicial.
    t : array float
        Tiempo.
    param : Array
        beta_mh1, beta_mh2 = Tasa de transmisión de mosquitos a humanos para cada cepa
        delta = Tasa de progresión a la enfermedad sintomática en humanos
        gamma = Tasa de recuperación para los humanos
        omega = Tasa de mortalidad natural (Humanos)
        sigma = Tasa de infección secundaria (ADE)
        mu = Tasa de muerte por enfermedad en humanos
        nu = Tasa de desarrollo de inmunidad cruzada
        rho = Tasa de pérdida de inmunidad cruzada

    Retorna
    -------
    list
        Dinámica del sistema para la población humana con dos cepas de dengue con ADE e inmunidad cruzada.

    Compartimentos
    --------------
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D :
    Susceptibles a ambas cepas, Expuestos, Infectados, Asintomáticos, Cross Immunity,
    Susceptibles a cepa 1, Susceptibles a cepa 2,
    Expuesto cepa 1 por previa infección por cepa 2, Expuesto cepa 2 por previa infección por cepa 1,
    Asintomático cepa 1 por previa infección por cepa 2, Asintomático cepa 2 por previa infección por cepa 1,
    Infectado cepa 1 por previa infección por cepa 2, Infectado cepa 2 por previa infección por cepa 1,
    Recuperados totales (los que cursaron ambas infecciones), Población fallecida por infecciones.

    Descripción
    -----------
    - S_both: Población susceptible a ambas cepas.
    - E1, E2: Poblaciones expuestas a la cepa 1 y cepa 2, respectivamente.
    - I1, I2: Poblaciones infectadas por la cepa 1 y cepa 2, respectivamente.
    - A1, A2: Poblaciones asintomáticas por la cepa 1 y cepa 2, respectivamente.
    - CI: Población con inmunidad cruzada.
    - S_1: Población susceptible a cepa 1 por previa infección a cepa 2.
    - S_2: Población susceptible a cepa 2 por previa infección a cepa 1.
    - E_12: Expuesto cepa 1 previa infección por cepa 2.
    - E_21: Expuesto cepa 2 previa infección por cepa 1.
    - A_12: Asintomático cepa 1 previa infección por cepa 2.
    - A_21: Asintomático cepa 2 previa infección por cepa 1.
    - I_12: Infectado cepa 1 previa infección por cepa 2.
    - I_21: Infectado cepa 2 previa infección por cepa 1.
    - R: Recuperados totales (los que cursaron ambas infecciones).
    - D: Población fallecida por infecciones (con aporte de todos los compartimentos de infectados sintomáticos).
    """
    S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
    seasonal_period=30*3
    annual_period=365
    # Extraer parámetros
    # beta_mh1 = param['beta_mh1'].value
    # beta_mh2 = param['beta_mh2'].value
    # beta_mh1 =  param['beta_mh1_0'].value*m.exp(-param['beta_mh1_1'].value*t)
    # beta_mh2 = param['beta_mh2_0'].value*m.exp(-param['beta_mh2_1'].value*t)
    if Scenario == 'No Clima':
        # A=param['A'].value
        # B=param['B'].value
        A=1
        B=1
        beta_mh1 =  ps['beta_mh1_0'] (1 + A* np.sin(2 * np.pi * t / seasonal_period) + B*np.sin(2 * np.pi * t / annual_period))
        beta_mh2 = ps['beta_mh2_0'](1 +  A* np.sin(2 * np.pi * t / seasonal_period) +  B*np.sin(2 * np.pi * t / annual_period))
    
    
    if Scenario == 'Clima':
        k_imh= ps['k_imh']
        k_temp= ps['k_temp']
        # imh = 
        beta_mh1 = calculate_beta(ps['beta_mh1_0'], 
                                  t, k_imh, k_temp)
        beta_mh2 = calculate_beta(ps['beta_mh2_0'], 
                                  t, k_imh, k_temp)
    if Scenario == 'Mosquito':
        k_mosquito =ps['k_mosquito']
        alpha=ps['alpha']
        beta_mh1 = calculate_beta_mosquito(ps['beta_mh1_0'], 
                                           t, k_mosquito,alpha)
        beta_mh2 = calculate_beta_mosquito(ps['beta_mh2_0'], 
                                           t, k_mosquito,alpha)
        # print('BETA: ' , beta_mh1)
        
    delta = ps['delta']
    gamma = ps['gamma']
    omega = ps['omega']
    sigma = ps['sigma']
    mu = ps['mu']
    nu = ps['nu']
    rho = ps['rho']
    
    # Número total de individuos
    N = S_both + E1 + E2 + I1 + I2 + A1 + A2 + CI + S_1 + S_2 + E_12 + E_21 + A_12 + A_21 + I_12 + I_21 + R + D
    
    # Ecuaciones del modelo
    dS_both_dt = omega * S_both- (beta_mh1 * S_both * I1 + beta_mh2 * S_both * I2) / N - omega * S_both
    dE1_dt = (beta_mh1 * S_both * I1 + beta_mh1 * S_1 * I1) / N - delta * E1 - omega * E1
    dE2_dt = (beta_mh2 * S_both * I2 + beta_mh2 * S_2 * I2) / N - delta * E2 - omega * E2
    dI1_dt = delta * E1 - (gamma + mu) * I1
    dI2_dt = delta * E2 - (gamma + mu) * I2
    dA1_dt = (1 - delta) * E1 - omega * A1 - sigma * A1
    dA2_dt = (1 - delta) * E2 - omega * A2 - sigma * A2
    dCI_dt = sigma * (A1 + A2) + nu * (I_12 + I_21) - rho * CI - omega * CI
    dS_1_dt = gamma * I_12 - (beta_mh1 * S_1 * I1) / N - omega * S_1
    dS_2_dt = gamma * I_21 - (beta_mh2 * S_2 * I2) / N - omega * S_2
    dE_12_dt = (beta_mh1 * S_1 * I1) / N - delta * E_12 - omega * E_12
    dE_21_dt = (beta_mh2 * S_2 * I2) / N - delta * E_21 - omega * E_21
    dA_12_dt = (1 - delta) * E_12 - omega * A_12 - sigma * A_12
    dA_21_dt = (1 - delta) * E_21 - omega * A_21 - sigma * A_21
    dI_12_dt = delta * E_12 - (gamma + mu) * I_12
    dI_21_dt = delta * E_21 - (gamma + mu) * I_21
    dR_dt = gamma * (I1 + I2) + gamma * (I_12 + I_21) - omega * R
    dD_dt = mu * (I1 + I2 + I_12 + I_21)
    
    return [dS_both_dt, dE1_dt, dE2_dt, dI1_dt, dI2_dt, dA1_dt, dA2_dt, dCI_dt, dS_1_dt, dS_2_dt,
            dE_12_dt, dE_21_dt, dA_12_dt, dA_21_dt, dI_12_dt, dI_21_dt, dR_dt, dD_dt]
# =============================================================================

def model(y, t, param):
    """
    Parameters
    ----------
    y : array
        Condición inicial.
    t : array float
        Tiempo.
    param : Array
        beta_mh =  Tasa de transmisión de mosquitos a humanos
        beta_hm =  Probabilidad de infección al ser picado por mosquito infectado
        beta_hh =  Probabilidad de infección al picar a un humano infectado
        delta =  Tasa de progresión a la enfermedad sintomática en humanos
        gamma =  Tasa de recuperación para los humanos
        omega =  Tasa de mortalidad natural (Humanos)
        sigma =  Tasa de infección secundaria (ADE)
        mu =  Tasa de muerte por enfermedad en humanos
        nu =  Tasa de desarrollo de inmunidad cruzada
        rho =  Tasa de pérdida de inmunidad cruzada
        alpha = Tasa de reproducción de los mosquitos
        delta_m =  Tasa de progresión a la infección en mosquitos
        gamma_m =  Tasa de recuperación en mosquitos
    """
    S_h, E_h1, I_h1, A_h1, R_h1, D_h1, E_h2, I_h2, A_h2, R_h2, D_h2, S_m, I_m1, I_m2 = y
    
    beta_mh = param['beta_mh']
    beta_hm = param['beta_hm']
    beta_hh = param['beta_hh']
    delta = param['delta']
    gamma = param['gamma']
    omega = param['omega']
    sigma = param['sigma']
    mu = param['mu']
    nu = param['nu']
    rho = param['rho']
    alpha = param['alpha']
    delta_m = param['delta_m']
    gamma_m = param['gamma_m']
    
    dS_hdt = omega * S_h - (beta_mh * S_h * (I_h1 + sigma * I_h2 + A_h1 + sigma * A_h2)) \
        - mu * S_h + rho * (I_h1 + I_h2 + A_h1 + A_h2) \
            - beta_hm * S_h * (I_m1 + I_m2)
    dE_h1dt = (beta_mh * S_h * (I_h1 + sigma * I_h2 + A_h1 + sigma * A_h2)) - delta * E_h1 - nu * E_h1
    dI_h1dt = delta * E_h1 - gamma * I_h1 - mu * I_h1
    dA_h1dt = sigma * delta * E_h1 - gamma * A_h1
    dR_h1dt = gamma * (I_h1 + A_h1) - rho * R_h1
    dD_h1dt = mu * I_h1
    
    dE_h2dt = (beta_mh * S_h * (I_h2 + sigma * I_h1 + A_h2 + sigma * A_h1)) - delta * E_h2 - nu * E_h2
    dI_h2dt = delta * E_h2 - gamma * I_h2 - mu * I_h2
    dA_h2dt = sigma * delta * E_h2 - gamma * A_h2
    dR_h2dt = gamma * (I_h2 + A_h2) - rho * R_h2
    dD_h2dt = mu * I_h2
    
    dS_mdt = alpha * S_m - (beta_hh * S_m * (I_h1 + I_h2 + A_h1 + A_h2)) + gamma_m * ((S_m + I_m1 + I_m2) - S_m)
    dI_m1dt = delta_m * S_m * (I_m1 + sigma * I_m2)
    dI_m2dt = delta_m * sigma * S_m * I_m1
    
    return [dS_hdt, dE_h1dt, dI_h1dt, dA_h1dt, dR_h1dt, dD_h1dt, 
            dE_h2dt, dI_h2dt, dA_h2dt, dR_h2dt, dD_h2dt, 
            dS_mdt, dI_m1dt, dI_m2dt]


# =============================================================================
def BetaH(T,p):
    Beta=T
    return Beta

# =============================================================================
def g(t, y0, ps):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(modelNM, y0, t, args=(ps,))
    return x
# =====================NRMSE===================================================

# def residual(param, t, y0, data):
#     y_model = g(t, y0, param)

#     I_1 = y_model[:, 3]
#     I_2 = y_model[:, 4]
#     I_12 = y_model[:, 14]
#     I_21 = y_model[:, 15]

#     # Calcula el NRMSE para los casos confirmados
#     nrmse_cases = np.sqrt(np.mean((I_1 + I_2 + I_12 + I_21 - data[0, :]) ** 2)) / (np.max(data[0, :]) - np.min(data[0, :]))

#     # Calcula el NRMSE para los casos acumulados
#     nrmse_accumulated = np.sqrt(np.mean((I_1.cumsum() + I_2.cumsum() + I_12.cumsum() + I_21.cumsum() - data[1, :]) ** 2)) / (np.max(data[1, :]) - np.min(data[1, :]))

#     # Calcular el periodograma de las señales real y modelo
#     f_real, Pxx_real = periodogram(data[0, :])
#     f_model, Pxx_model = periodogram(I_1 + I_2 + I_12 + I_21)

#     # Calcular la diferencia en los periodogramas
#     periodogram_difference = np.mean(np.abs(Pxx_model - Pxx_real))

#     # Ponderar las métricas y devolver la combinación
#     alpha = 0.6
#     beta = 0.4
#     combined_residual = (
#         alpha * nrmse_cases + 
#         beta * nrmse_accumulated + 
#         (1 - alpha - beta) * periodogram_difference
#     )

#     return combined_residual
# =====================WMAE====================================================

def residual(param, t, y0, data):
    # Calcula la solución del modelo
    y_model = g(t, y0, param)

    # Extrae las variables de interés del modelo
    I_1 = y_model[:, 3]
    I_2 = y_model[:, 4]
    I_12 = y_model[:, 14]
    I_21 = y_model[:, 15]

    # Calcula el WMAE para los casos confirmados
    wmae_cases = np.mean(np.abs(I_1 + I_2 + I_12 + I_21 - data[0, :]) / (np.max(data[0, :]) - np.min(data[0, :])))

    # Calcula el WMAE para los casos acumulados
    wmae_accumulated = np.mean(np.abs(I_1.cumsum() + I_2.cumsum() + I_12.cumsum() + I_21.cumsum() - data[1, :]) / (np.max(data[1, :]) - np.min(data[1, :])))

    # Calcular la diferencia logarítmica entre la señal modelada y los datos reales
    # log_difference = np.mean(np.abs(np.log(I_1 + I_2 + I_12 + I_21) - np.log(data[0, :])))
    # Calcular la diferencia logarítmica entre la señal modelada y los datos reales
    if np.any(I_1 + I_2 + I_12 + I_21):
        log_difference = np.mean(np.abs(np.log(I_1 + I_2 + I_12 + I_21) - np.log(data[0, :])))
    else:
        log_difference = 0

    # Ponderar las métricas y devolver la combinación
    alpha = 0.6
    beta = 0.4
    
    combined_residual = (
        alpha * wmae_cases + 
        beta * wmae_accumulated 
        # (1 - alpha - beta) * log_difference
    )
    
    return combined_residual

# =============================================================================



# def residual(param, t, y0, data):
#     y_model = g(t, y0, param)

#     I_1 = y_model[:, 3]
#     I_2 = y_model[:, 4]
#     I_12 = y_model[:, 14]
#     I_21 = y_model[:, 15]

#     # Residuos de Infección Diaria
#     Inf = (I_1 + I_2 + I_12 + I_21 - data[0, :]) / NH

#     # Residuos de Infección Acumulada
#     InfT = (I_1.cumsum() + I_2.cumsum() + I_12.cumsum() + I_21.cumsum() - data[1, :]) / NH

#     # Peso para dar más importancia a ciertos períodos (ajustar según necesidad)
#     weights = np.array([1.0, 2.0])

#     # Aplicar pesos y combinar residuos
#     weighted_residuals = weights.dot(np.array([Inf * 100, InfT]))

#     # Error Cuadrático Medio (MSE) solo para las salidas del modelo
#     mse = np.mean((np.array([I_1+ I_2+I_12+I_21]).T - data[0, :]) ** 2)

#     # Combina los residuos ponderados y el MSE para la función de residuos
#     residuals = np.concatenate((weighted_residuals, [mse]))

#     return residuals


# def residual(param, t, y0, data):
    
#     y_model = g(t, y0, param)

#     I_1 = y_model[:, 3]
#     I_2 = y_model[:, 4]
#     I_12= y_model[:, 14]
#     I_21= y_model[:, 15]

#     Inf = ((I_1 + I_2 + I_12 + I_21- data[0, :]) / NH).ravel()
    
#     InfT = ((I_1.cumsum() + I_2.cumsum()+I_12.cumsum() + I_21.cumsum() - data[1, :]) / NH).ravel()
#     residuals = np.array([Inf*100,InfT])
#     return residuals
# =============================================================================
def LoadData(path):
    Data = pd.read_csv(path,header=0)
    return Data

# =============================================================================
# Time-stepping loop to simulate the model
# def simulate_model_stochastic(y0, t, ps):
#     # Create an array to store the results
#     results = np.zeros((len(t), len(y0)))
#     results[0] = y0

#     for i in range(1, len(t)):
#         # Get the new state using the discrete stochastic model
#         results[i] = modelNM_discrete_stochastic(results[i-1], t[i], param)
    
#     return results
# # =============================================================================

# def simulateSC(t, y0, ps, Time):
#     # Simula el modelo estocástico
#     results = simulate_model_stochastic(y0, t, ps)
    
#     # Estructura de datos para el resultado
#     return {
#         't': t,
#         'S_both': results[:, 0],
#         'E1': results[:, 1],
#         'E2': results[:, 2],
#         'I1': results[:, 3],
#         'I2': results[:, 4],
#         'A1': results[:, 5],
#         'A2': results[:, 6],
#         'CI': results[:, 7],
#         'S_1': results[:, 8],
#         'S_2': results[:, 9],
#         'E_12': results[:, 10],
#         'E_21': results[:, 11],
#         'A_12': results[:, 12],
#         'A_21': results[:, 13],
#         'I_12': results[:, 14],
#         'I_21': results[:, 15],
#         'R': results[:, 16],
#         'D': results[:, 17]
#     }


# =============================================================================

def simulate(t, u, ps,Time):
    #    np.random.seed(int(ps['seed'].value))
    u = odeint(modelNM, y0, t,args=(ps,))
    return{'t': t, 'S_both' : u[:,0], 'E1' : u[:,1], 'E2' : u[:,2], 'I1' : u[:,3], 'I2' : u[:,4], 
           'A1' : u[:,5], 'A2' : u[:,6], 'CI' : u[:,7], 'S_1' : u[:,8], 'S_2' : u[:,9], 
           'E_12' : u[:,10], 'E_21' : u[:,11], 'A_12' : u[:,12], 'A_21' : u[:,13], 
           'I_12' : u[:,14], 'I_21' : u[:,15], 'R' : u[:,16], 'D' : u[:,17]}
# =============================================================================

def Smodel(y, t, param):
    for i in range(len(y)):
        if y[i] < 0:
            y[i] = 0

    S_h, E_h1, I_h1, A_h1, R_h1, D_h1, E_h2, I_h2, A_h2, R_h2, D_h2, S_m, I_m1, I_m2 = y
    
    beta_mh = max(0, param['beta_mh'].value)
    beta_hm = max(0, param['beta_hm'].value)
    beta_hh = max(0, param['beta_hh'].value)
    delta = max(0, param['delta'].value)
    gamma = max(0, param['gamma'].value)
    omega = max(0, param['omega'].value)
    sigma = max(0, param['sigma'].value)
    mu = max(0, param['mu'].value)
    nu = max(0, param['nu'].value)
    rho = max(0, param['rho'].value)
    alpha = max(0, param['alpha'].value)
    delta_m = max(0, param['delta_m'].value)
    gamma_m = max(0, param['gamma_m'].value)
    p = max(0, min(1, param['p'].value))  # Restringir p al rango [0, 1]
    
    dt = t[1] - t[0]
    # dt = 1/12

    # Probabilidades de transición
    prob_S_h_to_E_h1 = beta_mh * S_h * (I_h1 + sigma * I_h2 + A_h1 + sigma * A_h2) * dt
    prob_S_h_to_E_h1 *= beta_hm * (I_m1 + I_m2) * p
    
    prob_E_h1_to_I_h1 = delta * E_h1 * dt
    prob_I_h1_to_A_h1 = gamma * I_h1 * dt
    prob_A_h1_to_R_h1 = rho * A_h1 * dt
    prob_E_h2_to_I_h2 = delta * E_h2 * dt
    prob_I_h2_to_A_h2 = gamma * I_h2 * dt
    prob_A_h2_to_R_h2 = rho * A_h2 * dt
    prob_S_m_to_I_m1 = alpha * S_m * (I_h1 + I_h2 + A_h1 + A_h2) * dt
    prob_S_m_to_I_m2 = alpha * S_m * (I_h1 + I_h2 + A_h1 + A_h2) * dt
    
    prob_S_h_to_E_h2 = beta_hh * S_m * (I_h1 + I_h2 + A_h1 + A_h2) * dt
    prob_S_h_to_E_h2 *= gamma_m * (1 - S_m)
    
    prob_I_m1_to_I_h1 = delta_m * S_m * (I_m1 + sigma * I_m2) * dt
    prob_I_m2_to_I_h2 = delta_m * sigma * S_m * I_m1 * dt
    
    # Calcular el número de nuevos individuos en cada compartimiento
    new_S_h = np.random.binomial(S_h, 1 - min(1, prob_S_h_to_E_h1 + prob_S_h_to_E_h2))
    new_E_h1 = np.random.binomial(np.random.binomial(E_h1, min(1, prob_S_h_to_E_h1)), 1 - min(1, prob_E_h1_to_I_h1))
    new_I_h1 = np.random.binomial(np.random.binomial(I_h1, min(1, prob_E_h1_to_I_h1)), 1 - min(1, prob_I_h1_to_A_h1))
    new_A_h1 = np.random.binomial(np.random.binomial(A_h1, min(1, prob_I_h1_to_A_h1)), 1 - min(1, prob_A_h1_to_R_h1))
    new_R_h1 = np.random.binomial(R_h1, 1 - min(1, prob_A_h1_to_R_h1))
    # new_D_h1 = np.random.binomial(D_h1, 1 - min(1, prob_A_h1_to_R_h1))
    
    new_E_h2 = np.random.binomial(np.random.binomial(E_h2, min(1, prob_E_h2_to_I_h2)), 1 - min(1, prob_E_h2_to_I_h2))
    new_I_h2 = np.random.binomial(np.random.binomial(I_h2, min(1, prob_E_h2_to_I_h2)), 1 - min(1, prob_I_h2_to_A_h2))
    new_A_h2 = np.random.binomial(np.random.binomial(A_h2, min(1, prob_I_h2_to_A_h2)), 1 - min(1, prob_A_h2_to_R_h2))
    new_R_h2 = np.random.binomial(R_h2, 1 - min(1, prob_A_h2_to_R_h2))
    # new_D_h2 = np.random.binomial(D_h2, 1 - min(1, prob_A_h2_to_R_h2))
    
    new_S_m = np.random.binomial(S_m, 1 - min(1, prob_S_m_to_I_m1 + prob_S_m_to_I_m2))
    new_I_m1 = np.random.binomial(np.random.binomial(I_m1, min(1, prob_S_m_to_I_m1)), 1 - min(1, prob_I_m1_to_I_h1))
    new_I_m2 = np.random.binomial(np.random.binomial(I_m2, min(1, prob_I_m2_to_I_h2)), 1)
    
    # Incorporar los parámetros en las ecuaciones
    dS_hdt = omega - beta_mh * new_S_h * (new_I_h1 + new_I_h2 + new_A_h1 + new_A_h2) - mu * new_S_h + rho * (new_I_h1 + new_I_h2 + new_A_h1 + new_A_h2) - p * beta_hm * new_S_h * (new_I_m1 + new_I_m2)
    dE_h1dt = beta_mh * new_S_h * (new_I_h1 + sigma * new_I_h2 + new_A_h1 + sigma * new_A_h2) - delta * new_E_h1 - nu * new_E_h1
    dI_h1dt = delta * new_E_h1 - gamma * new_I_h1 - mu * new_I_h1
    dA_h1dt = sigma * delta * new_E_h1 - gamma * new_A_h1
    dR_h1dt = gamma * (new_I_h1 + new_A_h1) - rho * new_R_h1
    dD_h1dt = mu * new_I_h1
    
    dE_h2dt = beta_mh * new_S_h * (new_I_h2 + sigma * new_I_h1 + new_A_h2 + sigma * new_A_h1) - delta * new_E_h2 - nu * new_E_h2
    dI_h2dt = delta * new_E_h2 - gamma * new_I_h2 - mu * new_I_h2
    dA_h2dt = sigma * delta * new_E_h2 - gamma * new_A_h2
    dR_h2dt = gamma * (new_I_h2 + new_A_h2) - rho * new_R_h2
    dD_h2dt = mu * new_I_h2
    
    # Ecuaciones para la población de mosquitos
    dS_mdt = alpha - beta_hh * new_S_m * (new_I_h1 + new_I_h2 + new_A_h1 + new_A_h2) + gamma_m * (1 - new_S_m)
    dI_m1dt = delta_m * new_S_m * (new_I_m1 + sigma * new_I_m2)
    dI_m2dt = delta_m * sigma * new_S_m * new_I_m1
    
    return [dS_hdt, dE_h1dt, dI_h1dt, dA_h1dt, dR_h1dt, dD_h1dt, 
            dE_h2dt, dI_h2dt, dA_h2dt, dR_h2dt, dD_h2dt, 
            dS_mdt, dI_m1dt, dI_m2dt]
# =============================================================================

def interpolate_dataframe(df, num_points):
    """
    Interpola los datos de un DataFrame a más puntos en el tiempo.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de datos donde la primera columna es el tiempo en formato datetime.
    num_points : int, opcional
        Número de puntos a interpolar entre los puntos originales. Por defecto, 2.

    Retorna
    -------
    pandas.DataFrame
        DataFrame interpolado con más puntos en el tiempo.
    """
    # Eliminar filas con valores faltantes
    df_clean = df.dropna()

    # Extraer el tiempo y las variables
    Time = df_clean.iloc[:, 0]
    variables = df_clean.iloc[:, 1:]

    # Convertir el tiempo a valores numéricos
    time_numeric = np.arange(len(Time))

    # Crear un nuevo conjunto de puntos de tiempo
    new_time_numeric = np.linspace(0, len(Time) - 1, len(Time) * (num_points + 1))

    # Interpolar cada variable
    interpolated_variables = []
    for col_name, variable in variables.items():
        f = interp1d(time_numeric, variable, kind='cubic', fill_value='extrapolate')
        interpolated_variable = f(new_time_numeric)
        interpolated_variables.append(interpolated_variable)


    # Crear el DataFrame interpolado
    columns = [Time.name] + [f"{var_name}" for var_name in variables.columns]
    interpolated_df = pd.DataFrame(np.column_stack([np.interp(new_time_numeric, time_numeric, time_numeric), np.column_stack(interpolated_variables)]), columns=columns)

    return interpolated_df
# ==========================================================================
def resample_daily(DF):
    # Asegurarse de que la columna 'time' es de tipo datetime
    DF['time'] = pd.to_datetime(DF['time'])
    
    # Establecer 'time' como índice
    DF.set_index('time', inplace=True)
    
    # Remuestrear a intervalo diario y aplicar la interpolación cúbica
    # df_resampled = DF.resample('D').interpolate(method='spline', order=1)
    df_resampled = DF.resample('D').interpolate(method='nearest')

    
    # Resetear el índice para obtener 'time' como columna
    df_resampled.reset_index(inplace=True)
    
    # Rellenar los valores faltantes en 'time' si es necesario
    df_resampled['time'] = df_resampled['time'].ffill().bfill()
    return df_resampled
# ==========================================================================
# Función polinómica para el ajuste de datos climáticos y mosquitos
def polynomial_function(x, *coefficients):
    return sum(coefficients[i] * x**i for i in range(len(coefficients)))
# ==========================================================================
# Define la función objetivo (la salida del modelo que te interesa)
def objective_function(output):
    # Puedes elegir la salida que desees analizar, por ejemplo, el número total de infectados
    return np.sum(output[:, 3:7], axis=1)
# ==========================================================================
# Scenario='Constant'
Scenario='No Clima'
# Scenario= 'Clima'
# Scenario= 'Mosquito'
# Scenario= 'Combined'

# Uso de la función
# Supongamos que 'df' es tu DataFrame con la columna de tiempo en formato datetime
# dfinterp = interpolate_dataframe(df, num_points=2)

# Parámetros
# param = {
#     'beta_mh': 0.1, # Tasa de transmisión de mosquitos a humanos
#     'beta_hm': 0.1, # Probabilidad de infección al ser picado por mosquito infectado
#     'beta_hh': 0.1, # Probabilidad de infección al picar a un humano infectado
#     'delta': 0.1,   # Tasa de progresión a la enfermedad sintomática
#     'gamma': 0.05,  # Tasa de recuperación
#     'omega': 0.02,  # Tasa de mortalidad natural (Humanos)
#     'sigma': 0.1,   # Tasa de infección secundaria (ADE)
#     'mu': 0.02,     # Tasa de pérdida de inmunidad
#     'nu': 0.02,     # Tasa de desarrollo de inmunidad cruzada
#     'rho': 0.02,    # Tasa de pérdida de inmunidad cruzada
#     'alpha': 0.001, # Tasa de introducción de nuevas cepas
#     'delta_m': 0.1, # Tasa de progresión a la infección en mosquitos
#     'gamma_m': 0.05,# Tasa de re cuperación en mosquitos 
#     'p': 0.2        # Probabilidad de picadura de mosquito infectado a humano
# }
# year=2018
Data=LoadData('timeSeries/city/variables.csv') 
Data['time'] = pd.to_datetime(Data['time'])


# ==============================================
# Convertir la columna 'time' a tipo datetime si aún no lo está
Data['time'] = pd.to_datetime(Data['time'])

# Extraer el día y el mes de la columna 'time'
Data['day'] = Data['time'].dt.day
Data['month'] = Data['time'].dt.month

# Agrupar los datos por día y mes y calcular la media
daily_monthly_mean = Data.groupby(['month', 'day']).mean()
daily_monthly_mean = daily_monthly_mean.assign(time=lambda x: pd.to_datetime('2024-' + x.index.get_level_values('month').astype(str) + '-' + x.index.get_level_values('day').astype(str)))

# Mostrar los resultados
# print(daily_monthly_mean)

# # Plotear los datos reales
# # Convertir la columna 'time' a tipo de dato datetime si no está en ese formato
# # Convertir el índice de daily_monthly_mean a un objeto DatetimeIndex
# daily_monthly_mean.index = pd.to_datetime(daily_monthly_mean.index)

# # Crear subplots
# fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# # Plotear los datos reales en el primer subplot
# axs[0].plot(Data['time'], Data['notificacoes_total'], label='Real Data', color='blue')
# axs[0].set_xlabel('Date')
# axs[0].set_ylabel('Dengue Cases')
# axs[0].set_title('Real Data')

# # Plotear los datos de la media en el segundo subplot
# axs[1].plot(daily_monthly_mean['time'], daily_monthly_mean['notificacoes_total'], label='Mean Data', color='red')
# axs[1].set_xlabel('Date')
# axs[1].set_ylabel('Dengue Cases')
# axs[1].set_title('Mean Data')

# # Ajustar diseño de los subplots
# plt.tight_layout()

# # Mostrar el gráfico
# plt.show()
# ==============================================


# year_to_select = 2019
years=Data['time'].dt.year.unique()

for year_to_select in years:
    Datai = Data[Data['time'].dt.year == year_to_select]# Visualizar las primeras filas del DataFrame
    # Datai = Data[Data['time'].dt.year <= 2020]# Visualizar las primeras filas del DataFrame
    # Datai = Datai[Datai['time'].dt.year >= 2018]# Visualizar las primeras filas del DataFrame
    
    
    
    # Verificar si el primer elemento de la columna 'notificacoes_total' es mayor que el valor más pequeño de la serie
    if Datai['notificacoes_total'].iloc[0] > Datai['notificacoes_total'].min():
        # Obtener el valor mínimo de la serie que no sea 0
        min_value = Datai['notificacoes_total'].replace(0, np.nan).min()
        # Actualizar el primer elemento si el mínimo encontrado es menor que el primer elemento actual
        Datai['notificacoes_total'].iloc[0] = min_value
    
    
    # plt.plot(Datai['month'], Datai['20'], label='Reportados')
    # plt.plot(t, result[:, 1] + result[:, 6], label='Expuesto')
    # plt.plot(t, result[:, 2] + result[:, 3] + result[:, 7] + result[:, 8], label='Infectado')
    # plt.plot(t, result[:, 4] + result[:, 9], label='Recuperado')
    # plt.plot(t, result[:, 5] + result[:, 10], label='Fallecido')
    
    # plt.xlabel('Mes')
    # plt.ylabel('Casos')
    # plt.legend()
    # plt.show()
    
    
    # Cargar tu DataFrame df y definir las columnas de interés
    # df = pd.read_csv('tu_archivo.csv')
    columns_of_interest = ['mosquitos aedes aegypti total', 'imh', 'temp_area_ocupada satélite 30m pixel bimestre']
    # columns_of_interest = ['mosquitos aedes aegypti total']
    # columns_of_interest = ['imh', 'temp_area_ocupada satélite 30m pixel bimestre']
    
    
    
    # Ajustar la función polinómica a cada columna de interés
    degree = 5  # Puedes ajustar el grado del polinomio según tus necesidades
    initial_guess = np.ones(degree + 1)  # Establecer suposiciones iniciales para los coeficientes
    
    # Normalizar los datos
    scaler = MinMaxScaler()
    df_normalized = Datai.copy()
    df_normalized[columns_of_interest] = scaler.fit_transform(Datai[columns_of_interest])
    
    # Crear un DataFrame para almacenar los coeficientes de ajuste
    coefficients_df = pd.DataFrame(index=columns_of_interest, columns=['Coefficient_{}'.format(i) for i in range(degree + 1)])
    Type=' '
    # Realizar el ajuste y almacenar los coeficientes
    for column in columns_of_interest:
        # x = df_normalized['time'].astype(np.int64) // 10**9  # Convertir la columna de tiempo a segundos desde la época
        x = np.arange(0,len(df_normalized['time']))*60
        y = df_normalized[column]
        
        coefficients, _ = curve_fit(polynomial_function, x, y, p0=initial_guess)
        coefficients_df.loc[column] = coefficients
    
        # Generar valores predichos usando la función ajustada
        x_pred = np.linspace(min(x), max(x), 100)
        y_pred = polynomial_function(x_pred, *coefficients)
    
        # # Visualizar los resultados para cada columna
        # plt.figure(figsize=(8, 5))
        # plt.scatter(x, y, label='Datos reales')
        # plt.plot(x_pred, y_pred, label='Ajuste polinómico', color='red')
        # plt.title('Ajuste Polinómico a Datos Normalizados de {}'.format(column))
        # plt.xlabel('Tiempo (segundos desde la época)')
        # plt.ylabel(column)
        # plt.legend()
        # plt.show()
    
    # Mostrar los coeficientes de ajuste
    print(coefficients_df)
    
    
    
    # Ajustar y visualizar el ajuste
    
    
    
    NH=260000.00
    # NM=NH*10
    # Condiciones iniciales
    
    # E_h1 = 1        # Población expuesta a la cepa 1
    # I_h1 = np.array(Datai[prov])[0]       # Población sintomática por la cepa 1
    # A_h1 = 1        # Población asintomática por la cepa 1
    # R_h1 = 1       # Población recuperada de la cepa 1
    # D_h1 = 1       # Población fallecida por la cepa 1
    
    # E_h2 =1        # Población expuesta a la cepa 2
    # I_h2 = 1     # Población sintomática por la cepa 2
    # A_h2 = 1       # Población asintomática por la cepa 2
    # R_h2 = 1      # Població2n recuperada de la cepa 2
    # D_h2 = 1        # Población fallecida por la cepa 2
    
    # S_h = NH - E_h1 - I_h1 - A_h1 - R_h1 - D_h1 \
    #     - E_h2- I_h2 - A_h2 - R_h2 - D_h2 # Población susceptible
    
    
    
    # I_m1 = 1e2        # Población de mosquitos infectados por la cepa 1
    # I_m2 = 1e2        # Población de mosquitos infectados por la cepa 2
    # S_m = NM-I_m1  - I_m2         # Población de mosquitos susceptible
    # # Tiempo
    # # t = np.linspace(0, 200, 1000)
    
    # Condiciones iniciales
    # y0 = [S_h, E_h1, I_h1, A_h1, R_h1, D_h1, 
    #       E_h2, I_h2, A_h2, R_h2, D_h2, 
    #       S_m, I_m1, I_m2]
    
    # y0 = [S_h, E_h1, I_h1, A_h1, R_h1, D_h1, 
    #       E_h2, I_h2, A_h2, R_h2, D_h2]
    
    # topt=t
    
    # DataiInterp=interpolate_dataframe(Datai, num_points=6)
    DataiInterp=resample_daily(Datai)
    DataiInterp['Date']=DataiInterp['time']
    DataiInterp['time']=np.arange(1,len(DataiInterp['time'])+1)
    
    IMH = DataiInterp[['time', 'imh']].rename(columns={'imh': 'variable'})
    TEMP = DataiInterp[['time', 'temp_area_ocupada satélite 30m pixel bimestre']].\
        rename(columns={'temp_area_ocupada satélite 30m pixel bimestre': 'variable'})
    Mosquito= DataiInterp[['time', 'mosquitos aedes aegypti total']].\
        rename(columns={'mosquitos aedes aegypti total': 'variable'})
    IMH['time']=np.arange(1,len(IMH['time'])+1)    
    TEMP['time']=np.arange(1,len(TEMP['time'])+1)    
    Mosquito['time']=np.arange(1,len(Mosquito['time'])+1)    
    
    Mosquito['variable']=Mosquito['variable']/Mosquito['variable'].median()
    
    DataM = np.array([DataiInterp['notificacoes_total'].values, 
                      DataiInterp['notificacoes_total'].cumsum().values])
    
    # DataM = DataM/NH
    DataM = DataM
    # DataM=Datai
    Datai['time']=Datai.index
    # Time=Datai['time']
    # time=np.linspace(1, len(Datai['time']),len(Datai['time']))
    
    timeInterp=DataiInterp['time'].values
    
    dt=1/10
    # tf = len(time)
    # tl = int(tf/dt)
    tf = timeInterp[-1]-1
    tl = int(tf/dt)
    # tl = timeInterp[0]
    # t = np.linspace(1, tf, tl)
    t = np.linspace( timeInterp[0], tf, tl)
    
    # topt = np.linspace(0, tf+1, len(timeInterp))
    
    topt=timeInterp
    
    tfs = tf
    tls = int(tfs/dt)
    ts = np.linspace(timeInterp[0], tfs, tls)
      
    # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='least_squares', max_nfev=100000,
    #                  ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', diff_step=1e-4, verbose=2, tr_solver='lsmr')
    
    Time=np.linspace(1,timeInterp[-1],len(Datai['time']))
    FullOutputs = pd.DataFrame()
    FullSetOptParam = list()
    iterations=10
    Outputs = pd.DataFrame()
    TimeMax=Time[-1]
    
    n = np.size(DataM,1)
    percent = 0.8
    n_train = int(n * percent)
     # Seleccionar el 60% de los datos
    topt_train = topt[:n_train]
    DataM_train = DataM[:,:n_train]
    for i in np.arange(iterations):
    
        # Valor de I1 obtenido de tu dataset
        I1_value = np.array(Datai['notificacoes_total'])[0]
            # Inicialización de variables con valores aleatorios
        E1 = np.random.randint(0, int(0.1 * I1_value) + 1)
        E2 = np.random.randint(0, int(0.1 * I1_value) + 1)
        I1 = I1_value
        I2 = np.random.randint(0, int(0.1 * I1_value/3) + 1)
        A1 = np.random.randint(0, int(0.1 * I1_value/2) + 1)
        A2 = np.random.randint(0, int(0.1 * I1_value/4) + 1)
        CI = np.random.randint(0, int(0.1 * I1_value) + 1)
        S_1 = 0
        S_2 = 0
        # S_1 = np.random.randint(0, int(0.1 * I1_value) + 1)
        # S_2 = np.random.randint(0, int(0.1 * I1_value) + 1)
        E_12 = 0
        E_21 = 0
        A_12 = 0
        A_21 = 0
        # I_12 = 0
        # I_21 = 0
        # E_12 = np.random.randint(0, int(0.1 * I1_value) + 1)
        # E_21 = np.random.randint(0, int(0.1 * I1_value) + 1)
        # A_12 = np.random.randint(0, int(0.1 * I1_value) + 1)
        # A_21 = np.random.randint(0, int(0.1 * I1_value) + 1)
        I_12 = np.random.randint(0, int(0.1 * I1_value/4) + 1)
        I_21 = np.random.randint(0, int(0.1 * I1_value/4) + 1)
        R = 0
        D = 0
    
        S_both=NH- E1 -  E2 -  I1 -  I2 -  A1 -  A2 -  CI -  S_1 -  S_2 -  E_12 -  E_21 -  A_12 -  A_21 -  I_12 -  I_21 -  R -  D
        y0 = [S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, 
              E_12, E_21, A_12, A_21, I_12, I_21, R, D]
    
        y0opt=y0
    
        param = Parameters()
        param.add('beta_mh1_0', value=np.random.uniform(1e-2, 1), min=1e-2, max=5)
        param.add('beta_mh2_0', value=np.random.uniform(1e-2, 1), min=1e-2, max=5)
        param.add('k_imh', value=np.random.uniform(1e-2, 1), min=1e2, max=1)
        param.add('k_temp', value=np.random.uniform(1e-2, 1), min=1e-2, max=1) 
        param.add('k_mosquito', value=np.random.uniform(1e-2, 1), min=1e-2, max=1) 
        param.add('alpha', value=np.random.uniform(1e-2,1), min=1e-2, max=1)
        param.add('delta', value=np.random.uniform(1e-2,0.5), min=1e-2, max=0.5)
        param.add('gamma', value=np.random.uniform(1e-2, 0.5), min=1e-2, max=1)
        param.add('omega', value=np.random.uniform(1e-2, 0.2), min=1e-2, max=1)
        param.add('sigma',value=np.random.uniform(1e-2, 0.2), min=1e-2, max=1)
        param.add('mu', value=np.random.uniform(1e-3, 0.01), min=1e-3, max=0.01)
        param.add('nu', value=np.random.uniform(1e-6, 0.001), min=1e-6, max=1e3)
        param.add('rho', value=np.random.uniform(1e-2, 0.01), min=1e-2, max=1)
        param.add('A', value=np.random.uniform(1e-2, 0.01), min=1e-2, max=3)
        param.add('B', value=np.random.uniform(1e-2, 0.01), min=1e-2, max=3)
        # param.add('duration_cross_immunity', value=np.random.uniform(1e-2, 1), min=1e-6, max=1)
    
        
       # 
        # print(topt)
        # print(Mosquito['time'])
        # sol = minimize(residual, param, args=(topt,y0opt,DataM),method='least_squares',max_nfev=10000,nan_policy='omit',
                            # ftol=1e-10,gtol=1e-10,xtol=1e-10,loss='huber',diff_step=1e-4,verbose=2,tr_solver='exact')
        # 
        # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='BFGS', max_nfev=10000,
                       # loss='huber', ftol=1e-10, gtol=1e-10, xtol=1e-10, verbose=2)
    
        
        # sol = minimize(residual, param, args=(topt,y0opt,DataM),method='ampgo',max_nfev=10000)
    
    # 
        # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='least_squares', max_nfev=100000,nan_policy='omit',
                            # ftol=1e-12, gtol=1e-12, xtol=1e-12, loss='arctan', diff_step=1e-4, verbose=2, tr_solver='lsmr')
        
        sol = minimize(residual, param, args=(topt_train, y0opt, DataM_train), method='least_squares', max_nfev=10000, nan_policy='omit',
                       ftol=1e-10, gtol=1e-10, xtol=1e-10, loss='arctan', diff_step=1e-4, verbose=2, tr_solver='exact')
        # 
        # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='least_squares', max_nfev=100000, nan_policy='omit',
                    # ftol=1e-12, gtol=1e-12, xtol=1e-12, loss='arctan', diff_step=1e-4, verbose=2, tr_solver='trf')
        
        # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='least_squares', max_nfev=100000,
                    # nan_policy='omit', ftol=1e-12, gtol=1e-12, xtol=1e-12, loss='arctan', diff_step=1e-4,
                    # verbose=2, tr_solver=None)
    
    
        # sol = minimize(residual, param, args=(topt, y0opt, DataM), method='least_squares', max_nfev=100000,
        #            nan_policy='omit', ftol=1e-12, gtol=1e-12, xtol=1e-12, loss='arctan', diff_step=1e-4,
        #            verbose=2, tr_solver=year_to_select'lsmr')
    
    
    
        
        paropt = sol.params
        FullSetOptParam.append(paropt)
        # if len(FullSetOptParam) == 0:
        #     # Si FullOutputs está vacío, copiar 'result' a FullOutputs
        #     FullSetOptParam = paropt.copy()
        # else:
        #     # Si FullOutputs ya contiene datos, combinar 'result' con FullOutputs
        #     FullSetOptParam.append(paropt)
        # result = odeint(model, y0, t,args=(paropt,))
        # result=simulate(t, y0, paropt,time)
        result=  pd.DataFrame(simulate(t, y0, paropt,Time))
        # result=  pd.DataFrame(simulateSC(t, y0, paropt))
        # IModel= result[:, 2] + result[:, 7]
        
        # FullOutputs = FullOutputs.append(result, ignore_index=True)
        # Verificar si FullOutputs está vacío
        if FullOutputs.empty:
            # Si FullOutputs está vacío, copiar 'result' a FullOutputs
            FullOutputs = result.copy()
        else:
            # Si FullOutputs ya contiene datos, combinar 'result' con FullOutputs
            # FullOutputs = FullOutputs.append(result, ignore_index=True)
            FullOutputs = pd.concat([FullOutputs, result], ignore_index=True)
    
        # Outputs = Outputs.append(result)
        if Outputs.empty:
            # Si FullOutputs está vacío, copiar 'result' a FullOutputs
            Outputs = result.copy()
        else:
            # Si FullOutputs ya contiene datos, combinar 'result' con FullOutputs
            # Outputs = Outputs.append(result, ignore_index=True)
            Outputs = pd.concat([Outputs, result], ignore_index=True)
    
    # Resolución del sistema de ecuaciones diferenciales
    # S_both, E1, E2, I1, I2, A1, A2, CI, S_1, S_2, E_12, E_21, A_12, A_21, I_12, I_21, R, D = y
    
    
    Results = np.array(FullOutputs)
    vector = FullOutputs.groupby('t', as_index=False).mean()
    std = FullOutputs.groupby('t').std()
    timesimul=vector['t']
    
    # I= np.array(vector['I_h1']+vector['I_h2'])
    # i= np.array(std['I_h1']+std['I_h2'])
    I= np.array(vector['I1']+vector['I2']+vector['I_21']+vector['I_12'])
    i= np.array(std['I1']+std['I2']+std['I_21']+std['I_12'])
    # Gráficas
    # plt.style.use('seaborn-darkgrid')
    
    fig = plt.figure(1, facecolor='w')
    ax = fig.add_subplot(111, facecolor='w', axisbelow=True)
    ax.plot(timesimul, I, 'r', alpha=0.75, lw=2, label='Model')
    ax.fill_between(timesimul,  I-i, I+i, color='r', alpha=0.2)
    
    ax.plot(Time,  Datai['notificacoes_total'], 'or', alpha=0.5, lw=2, label='Cases')
    
    ax.set_xlabel('Time /months')
    ax.set_ylabel('Reported Cases')
    # ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(visible=True, which='major', c='w', lw=2, ls='-')
    
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()
    
    
    
    FigName = str(year_to_select)+Type +' '+Scenario+'.png'
    # Directory = 'Re-Infection Model Simplified/Plots/Fit/Constant/Data/'
    File = 'Report'+str(year_to_select)+Type+Scenario+'.csv'
    Path = File
    # Report.to_csv(Path, sep=',')
    File = 'FullOutputs'+str(year_to_select)+Type+Scenario+'.csv'
    Path = File
    # FullOutputs.to_csv(Path, sep=',')
    
    Save=Scenario+' '+str(year_to_select)+Type+'.pkl'
    with open(Save, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([FullSetOptParam,Outputs,year_to_select], f)
        
    fig.savefig(FigName, dpi=600)
    plt.close(fig)
   




