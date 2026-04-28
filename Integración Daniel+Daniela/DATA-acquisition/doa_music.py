import numpy as np
from scipy.signal import find_peaks
import config_doa                                                           # Importamos nuestro archivo de configuración

def doa_music(signals, micpos, fs):
    """
    Estimador DOA MUSIC con detección dinámica del número de fuentes (L_est).
    
    Parámetros:
    - signals: array NumPy [N_muestras, M_micros]
    - micpos: array NumPy [M_micros, 2] con coordenadas (x,y)
    - fs: frecuencia de muestreo
    
    Retorna:
    - theta_est: lista de ángulos estimados
    """
    c = 343.0
    N, M = signals.shape

    Nfft = int(2 ** np.ceil(np.log2(N)))                                # 1. FFT y Selección de banda
    X = np.fft.fft(signals, n=Nfft, axis=0)
    
    f = np.arange(Nfft) * fs / Nfft                                     # Equivalente en MATLAB a: f = (0:Nfft-1)' * fs/Nfft
    
    f_idx = (f >= config_doa.MUSIC['f_min']) & (f <= config_doa.MUSIC['f_max'])
    f_sel = f[f_idx]
    X_filt = X[f_idx, :]
    Nf = len(f_sel)

    if Nf == 0:
        return [0.0], np.zeros(720), np.arange(0, 360, 0.5)

    Rxx = (X_filt.conj().T @ X_filt) / Nf                               # 2. Covarianza Vectorizada
    Rxx = Rxx + (0.02 * np.trace(Rxx).real / M) * np.eye(M)             # Añadimos una fracción de la potencia media (traza) a la diagonal

    eigenvalues, eigenvectors = np.linalg.eigh(Rxx)                     # 3. Subespacio de Ruido
    
    idx = np.argsort(eigenvalues)[::-1]                                 # Ordenamos de mayor a menor (descend)
    V = eigenvectors[:, idx]
    
    L_est = config_doa.MUSIC['l_est_default']
    Un = V[:, L_est:]

    theta_scan = np.arange(0, 360, 0.5)                                 # 4. Escaneo Vectorizado
    P_music = np.zeros(len(theta_scan))
    
    U_vecs = np.vstack((np.cos(np.radians(theta_scan)), np.sin(np.radians(theta_scan))))        # Precalculamos los vectores de dirección unitarios
    
    Tau = -(micpos @ U_vecs) / c                                        # Precalculamos todos los retardos

    for i in range(len(theta_scan)):
        A = np.exp(-1j * 2 * np.pi * np.outer(f_sel, Tau[:, i]))        # Matriz de dirección para todas las frecuencias a la vez    
        denominador = np.sum(np.abs(A @ Un)**2) / Nf                    # Proyección sobre ruido
        P_music[i] = (1 / (denominador + np.finfo(float).eps)).real

    p_min = np.min(P_music)
    p_max = np.max(P_music)
    if p_max > p_min:                                                   # 5. Normalización Min-Max (Rango exacto 0 a 1)
        P_music = (P_music - p_min) / (p_max - p_min)
    else:
        P_music = np.zeros_like(P_music)
    
    L_out = config_doa.MUSIC['l_out']                                       # 6. Búsqueda de Picos y Fiabilidad
    locs_idx, _ = find_peaks(P_music)                                   # Buscar picos (SciPy devuelve índices, no los valores en grados)
    
    pks_all = P_music[locs_idx]
    sort_idx = np.argsort(pks_all)[::-1]                                # Ordenar los picos de mayor a menor altura ('SortStr', 'descend' de MATLAB)
    locs = theta_scan[locs_idx[sort_idx]]                               # Convertimos índices a grados ordenados
    pks = pks_all[sort_idx]

    if len(locs) >= L_out:                                              # Tomamos los L_out mejores
        theta_est = list(locs[:L_out])
        conf_est  = list(pks[:L_out])
    else:                                                               # Rellenamos con el máximo absoluto si hay menos picos
        theta_est = list(locs)
        conf_est  = list(pks)
        
        max_idx = np.argmax(P_music)
        val_max = theta_scan[max_idx]
        max_val = P_music[max_idx]
        
        while len(theta_est) < L_out:
            theta_est.append(val_max)
            conf_est.append(max_val)

    theta_est_arr = np.array(theta_est)                                 # Ordenamos espacialmente para mantener el formato de salida
    conf_est_arr  = np.array(conf_est)
    
    sort_spatial_idx = np.argsort(theta_est_arr)
    
    theta_est_final = theta_est_arr[sort_spatial_idx].tolist()
    conf_est_final  = conf_est_arr[sort_spatial_idx].tolist()

    return theta_est_final, conf_est_final, P_music, theta_scan