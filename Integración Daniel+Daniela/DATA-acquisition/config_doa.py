# ==========================================
# CONFIGURACIÓN DEL SISTEMA TDOA - LPRO
# ==========================================

# --- Parámetros de Audio y Hardware ---
AUDIO = {
    'channels_hw': 6,       # Canales que pedimos al driver de Windows
    'rate': 16000,          # Frecuencia de muestreo (Hz)
    't_frame': 0.15,        # Tamaño de la trama en segundos (ej. 150ms)
    'radius_m': 0.044       # Radio del array ReSpeaker en metros
}

# --- Parámetros del Tracker (Lógica Alpha/Salud) ---
TRACKER = {
    'alpha': 0.4,           # Suavizado (0 = mucha inercia, 1 = sin filtro)
    'gate_deg': 45.0,       # Grados máximos para asociar un pico a una pista
    'conf_keep': 0.35,      # Fiabilidad mínima para mantener viva una pista
    'conf_start': 0.9,      # Fiabilidad mínima para que nazca una nueva pista
    'health_max': 99.0,     # Batería máxima de una pista
    'health_damage': 2,     # Daño por frame si no encuentra medidas
    'min_age_confirm': 4    # Frames consecutivos para salir a pantalla
}

# --- Parámetros del Estimador (MUSIC) ---
MUSIC = {
    'f_min': 500,           # Frecuencia mínima de corte (Hz)
    'f_max': 1800,          # Frecuencia máxima de corte (Hz)
    'l_est_default': 3,     # Fuentes base para el subespacio de ruido
    'l_out': 2              # Máximo de picos que devuelve al tracker
}