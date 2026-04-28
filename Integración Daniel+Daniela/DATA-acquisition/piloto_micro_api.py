import numpy as np
import pyaudio
import librosa
import tensorflow as tf
import time
import threading
import requests

# --- Importamos configuraciones ---
import config_doa as cdoa
import config_cnn as ccnn

from doa_tracker_single import DOATrackerSingle 
from doa_music import doa_music

MODEL_PATH = ccnn.MODEL['path']
API_URL = ccnn.MODEL['api_url']
SR = cdoa.AUDIO['rate']  

# =========================================================
# CONFIGURACIÓN MULTITASA (MULTIRATE PROCESSING)
# =========================================================
# --- RELOJ 1: CNN (Baja Frecuencia) ---
CNN_WINDOW_SEC = ccnn.SIGNAL['chunk_sec']            
CNN_STEP_SEC = ccnn.SIGNAL['step_sec']        
CNN_WINDOW_SAMP = int(CNN_WINDOW_SEC * SR)                 
CNN_STEP_SAMP = int(CNN_STEP_SEC * SR)  

# --- RELOJ 2: DoA (Alta Frecuencia) ---
DOA_WINDOW_SEC = cdoa.AUDIO['t_frame']
DOA_WINDOW_SAMP = int(DOA_WINDOW_SEC * SR)
DOA_STEP_SAMP = DOA_WINDOW_SAMP 

# --- BUFFER MAESTRO ---
MAX_BUFFER = max(CNN_WINDOW_SAMP, DOA_WINDOW_SAMP)
MICRO_CHUNK_SAMP = int(0.025 * SR) # Leemos el ReSpeaker en trozos de 25ms

PAD_TO = ccnn.SIGNAL['pad_to']
N_FFT = ccnn.SIGNAL['n_fft']
HOP_LENGTH = ccnn.SIGNAL['hop_length']
THRESHOLD = ccnn.MODEL['threshold']

# =========================================================
# INICIALIZACIÓN DE MODELOS
# =========================================================
print(">>> [1/3] Cargando Modelo CNN...")
modelo_cnn = tf.keras.models.load_model(MODEL_PATH)

print(">>> [2/3] Inicializando Motor DoA y Tracker...")
tracker = DOATrackerSingle(cdoa.TRACKER)
phi_mics = np.arange(4) * (2 * np.pi / 4)
micpos = np.column_stack((cdoa.AUDIO['radius_m'] * np.cos(phi_mics), 
                          cdoa.AUDIO['radius_m'] * np.sin(phi_mics)))

# =========================================================
# INICIALIZACIÓN DE PYAUDIO
# =========================================================
print(f">>> [3/3] Configurando PyAudio ({cdoa.AUDIO['channels_hw']} canales)...")
audio = pyaudio.PyAudio()

dev_index = -1
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "ReSpeaker" in info['name'] and info['maxInputChannels'] > 0:
        dev_index = i
        print(f"    ✅ ReSpeaker detectado en ID {i}: {info['name']}")
        break

if dev_index == -1:
    print("\n❌ ERROR: ReSpeaker no encontrado. Conecta el USB y verifica los permisos.")
    audio.terminate()
    exit()

stream = audio.open(format=pyaudio.paInt16, 
                    channels=cdoa.AUDIO['channels_hw'],
                    rate=SR, 
                    input=True,
                    input_device_index=dev_index,
                    frames_per_buffer=MICRO_CHUNK_SAMP)

print(f"    - Tasa DoA dinámica: {DOA_WINDOW_SEC*1000:.0f} ms ({1/DOA_WINDOW_SEC:.1f} Hz)")
print(f"    - Tasa CNN estática: {CNN_STEP_SEC*1000:.0f} ms ({1/CNN_STEP_SEC:.1f} Hz)")

# --- FUNCIONES DE RED ---
def enviar_post_async(endpoint, data):
    def task():
        try: requests.post(f"{API_URL}{endpoint}", json=data, timeout=0.5)
        except: pass
    threading.Thread(target=task, daemon=True).start()

# --- MEMORIA Y ESTADO COMPARTIDO ---
buffer_audio = np.zeros((MAX_BUFFER, cdoa.AUDIO['channels_hw']), dtype=np.float32)
acumulador_cnn = 0
acumulador_doa = 0
estado_sirena = False
probabilidad_actual = 0.0

# =========================================================
# BUCLE PRINCIPAL
# =========================================================
print("\n🚀 SISTEMA SIRA ACTIVO - Escuchando...")
try:
    while True:
        # 1. Leer micro-bloque de 25ms
        data = stream.read(MICRO_CHUNK_SAMP, exception_on_overflow=False)
        t0_captura = time.time()
        
        # 2. Convertir bytes a matriz NumPy
        nuevo_bloque = np.frombuffer(data, dtype=np.int16).reshape(-1, cdoa.AUDIO['channels_hw'])
        # Normalizamos el audio a floats (-1.0 a 1.0) para la FFT y la UI
        nuevo_bloque_float = nuevo_bloque.astype(np.float32) / 32768.0
        n_nuevos = len(nuevo_bloque_float)

        # 3. Desplazar buffer y añadir muestras
        buffer_audio = np.roll(buffer_audio, -n_nuevos, axis=0)
        buffer_audio[-n_nuevos:, :] = nuevo_bloque_float
        
        acumulador_cnn += n_nuevos
        acumulador_doa += n_nuevos

        # ---------------------------------------------------------
        # RELOJ 1: DoA (Ejecuta rápido y a su ritmo)
        # ---------------------------------------------------------
        if acumulador_doa >= DOA_STEP_SAMP:
            acumulador_doa -= DOA_STEP_SAMP
            
            ventana_doa = buffer_audio[-DOA_WINDOW_SAMP:, :]
            mics_raw = ventana_doa[:, 1:5].astype(np.float64)
            
            angulo_oficial = None
            
            if estado_sirena: 
                mics_norm = mics_raw - np.mean(mics_raw, axis=0)
                mics_norm = mics_norm / (np.max(np.abs(mics_norm), axis=0) + np.finfo(float).eps)
                
                theta_est, conf_est, P_music, _ = doa_music(mics_norm, micpos, SR)
                angulos_validos = tracker.actualizar(theta_est, conf_est)
                if len(angulos_validos) > 0: angulo_oficial = int(angulos_validos[0])
            else:
                tracker.actualizar([], [])

            # CORRECCIÓN 1: Formato exacto que pide la API para DatosDOA
            angulo_a_enviar = int(angulo_oficial) if angulo_oficial is not None else 0
            tendencia_a_enviar = "Estable" if angulo_oficial is not None else "Buscando"

            enviar_post_async("/update_doa", {
                "angulo": angulo_a_enviar, 
                "tendencia": tendencia_a_enviar
            })

        # ---------------------------------------------------------
        # RELOJ 2: CNN (Ejecuta lento para dar fiabilidad)
        # ---------------------------------------------------------
        if acumulador_cnn >= CNN_STEP_SAMP:
            acumulador_cnn -= CNN_STEP_SAMP
            t_inicio_cnn = time.time()
            
            ventana_cnn = buffer_audio[-CNN_WINDOW_SAMP:, :]
            chunk_cnn = ventana_cnn[:, 0].astype(np.float32) 
            
            if len(chunk_cnn) < PAD_TO: chunk_cnn = np.pad(chunk_cnn, (0, PAD_TO - len(chunk_cnn)))
            S_mel = librosa.feature.melspectrogram(y=chunk_cnn, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0, n_mels=ccnn.SIGNAL['n_mels'], fmin=ccnn.SIGNAL['fmin'], fmax=ccnn.SIGNAL['fmax'])
            S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
            S_mel_db = (S_mel_db - np.min(S_mel_db)) / (np.max(S_mel_db) - np.min(S_mel_db) + 1e-6)
            
            prediccion = modelo_cnn.predict(S_mel_db.reshape(1, S_mel_db.shape[0], S_mel_db.shape[1], 1), verbose=0)
            probabilidad_actual = float(prediccion[0][0])
            
            estado_sirena = probabilidad_actual >= THRESHOLD
            latencia_ms = (time.time() - t_inicio_cnn) * 1000

            # CORRECCIÓN 2: Tipos forzados a float/bool
            enviar_post_async("/update_deteccion", {
                "sirena": bool(estado_sirena), 
                "probabilidad": float(probabilidad_actual), 
                "tipo_vehiculo": "Ambulancia" if estado_sirena else "Ninguno", 
                "latencia_inferencia_ms": float(latencia_ms), 
                "fps": float(1.0 / (CNN_STEP_SAMP/SR)), 
                "t0_captura": float(t0_captura)
            })
            
            # CORRECCIÓN 3: Cálculo real de FFT y envío del schema exacto de DatosAudio
            # Extraemos 80 muestras para la onda temporal
            onda_ui = [float(x) for x in chunk_cnn[::int(len(chunk_cnn)/80)]]
            
            # Calculamos la FFT para la gráfica de frecuencias (Ej: 64 barras)
            fft_complex = np.fft.rfft(chunk_cnn)
            fft_mag = np.abs(fft_complex)[:64]
            if np.max(fft_mag) > 0:
                fft_mag = fft_mag / np.max(fft_mag) # Normalizamos de 0 a 1
            fft_ui = [float(x) for x in fft_mag] 

            enviar_post_async("/update_audio", {
                "waveform_summary": onda_ui, 
                "fft_data": fft_ui,     # Ahora mandamos la FFT real
                "mfcc_features": []     # La API lo exige, lo mandamos vacío si no lo usas en UI
            })

            if estado_sirena: 
                print(f"🚨 SIRENA [{probabilidad_actual*100:.0f}%] | Muestreo DoA a {1/DOA_WINDOW_SEC:.0f} FPS...")
            else: 
                print(f"✅ Ruido   [{probabilidad_actual*100:.0f}%]")

except KeyboardInterrupt:
    print("\nDeteniendo sistema SIRA...")
    stream.stop_stream()
    stream.close()
    audio.terminate()