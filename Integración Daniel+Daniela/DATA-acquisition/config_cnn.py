# ==========================================
# CONFIGURACIÓN DEL DETECTOR CNN
# ==========================================

MODEL = {
    'path': 'modelo_sirenas_cnn_optimo.keras',
    'api_url': 'http://192.168.1.138:8000',
    'threshold': 0.9       # Umbral de probabilidad para activar alerta de sirena
}

# Parámetros fijos de la arquitectura con la que se entrenó la red
SIGNAL = {
    'chunk_sec': 0.5,       # Tamaño de la ventana de captura (500 ms)
    'step_sec': 0.375,      # Avance de la ventana (determina el solape)
    'pad_to': 8192,         # Relleno de ceros para estabilizar la FFT
    'n_fft': 1024,          # Tamaño de la Transformada Rápida de Fourier
    'hop_length': 512,      # Salto entre frames del espectrograma
    'n_mels': 359,          # Número de bandas de frecuencia Mel
    'fmin': 0,              # Frecuencia mínima del espectrograma
    'fmax': 8000            # Frecuencia máxima del espectrograma (SR / 2)
}