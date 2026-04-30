import time
import threading

import numpy as np
import pyaudio
import requests

# --- Importamos configuraciones ---
import config_doa as cdoa
import config_cnn as ccnn
import detection_manager

from doa_tracker_single import DOATrackerSingle
from doa_music import doa_music


API_URL = ccnn.MODEL['api_url']
SR = cdoa.AUDIO['rate']


# =========================================================
# CONFIGURACION DE MODELOS DE DETECCION
# =========================================================
CNN_MODEL_DIR = detection_manager.resolve_cnn_directory()
TRAD_MODEL_DIR = detection_manager.resolve_traditional_directory()

print(">>> [1/4] Cargando modelos de deteccion...")
cnn_states = detection_manager.load_cnn_models(CNN_MODEL_DIR)
traditional_states = detection_manager.load_traditional_models(TRAD_MODEL_DIR)
detection_states = cnn_states + traditional_states

VOTE_WINDOW_SEC = detection_manager.validate_detection_states(detection_states)
VOTE_STEP_SAMP = int(round(VOTE_WINDOW_SEC * SR))
if VOTE_STEP_SAMP <= 0:
    raise ValueError("La ventana global de votacion de deteccion no es valida.")

print(f"    - CNN cargadas: {len(cnn_states)}")
for state in cnn_states:
    config = state['config']
    print(f"      Modelo: {state['path']}")
    print(f"      Config: {state['config_path']}")
    print(
        "      Frontend: "
        f"{config['spectral_frontend']} | "
        f"Representacion: {config['feature_representation']} | "
        f"Chunk: {config['chunk_length_s']:.3f}s | "
        f"Paso: {config['decision_step_s']:.3f}s | "
        f"Umbral: {config['chunk_threshold']:.3f} | "
        f"Racha+: {state['required_positive_s']:.3f}s"
    )

print(f"    - Clasificadores tradicionales cargados: {len(traditional_states)}")
for state in traditional_states:
    config = state['config']
    print(f"      Bundle: {state['path']}")
    print(f"      Config: {state['config_path']}")
    print(
        "      Modelo: "
        f"{config['model_name']} | "
        f"Chunk: {config['chunk_length_s']:.3f}s | "
        f"Paso: {config['decision_step_s']:.3f}s | "
        f"Umbral: {config['chunk_threshold']:.3f} | "
        f"Racha+: {state['required_positive_s']:.3f}s | "
        f"Probabilidad: {'si' if config['has_probability'] else 'no'}"
    )
    for warning in config.get('load_warnings', []):
        print(f"      Aviso sklearn: {warning}")

# =========================================================
# CONFIGURACION MULTITASA (MULTIRATE PROCESSING)
# =========================================================
DOA_WINDOW_SEC = cdoa.AUDIO['t_frame']
DOA_WINDOW_SAMP = int(DOA_WINDOW_SEC * SR)
DOA_STEP_SAMP = DOA_WINDOW_SAMP

MAX_DETECTION_WINDOW_SAMP = max(state['window_samp'] for state in detection_states)
MAX_BUFFER = max(MAX_DETECTION_WINDOW_SAMP, DOA_WINDOW_SAMP, VOTE_STEP_SAMP)
MICRO_CHUNK_SAMP = int(0.025 * SR)  # Leemos el ReSpeaker en trozos de 25ms
DETECTION_CHANNEL = detection_manager.resolve_detection_channel(
    cdoa.AUDIO['channels_hw'],
    is_respeaker=True,
)

# =========================================================
# INICIALIZACION DE MODELOS DOA
# =========================================================
print(">>> [2/4] Inicializando Motor DoA y Tracker...")
tracker = DOATrackerSingle(cdoa.TRACKER)
phi_mics = np.arange(4) * (2 * np.pi / 4)
micpos = np.column_stack((cdoa.AUDIO['radius_m'] * np.cos(phi_mics),
                          cdoa.AUDIO['radius_m'] * np.sin(phi_mics)))

# =========================================================
# INICIALIZACION DE PYAUDIO
# =========================================================
print(f">>> [3/4] Configurando PyAudio ({cdoa.AUDIO['channels_hw']} canales)...")
audio = pyaudio.PyAudio()

dev_index = -1
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "ReSpeaker" in info['name'] and info['maxInputChannels'] > 0:
        dev_index = i
        print(f"    ReSpeaker detectado en ID {i}: {info['name']}")
        break

if dev_index == -1:
    print("\nERROR: ReSpeaker no encontrado. Conecta el USB y verifica los permisos.")
    audio.terminate()
    exit()

stream = audio.open(format=pyaudio.paInt16,
                    channels=cdoa.AUDIO['channels_hw'],
                    rate=SR,
                    input=True,
                    input_device_index=dev_index,
                    frames_per_buffer=MICRO_CHUNK_SAMP)

print(">>> [4/4] Sistema configurado.")
print(f"    - Tasa DoA dinamica: {DOA_WINDOW_SEC*1000:.0f} ms ({1/DOA_WINDOW_SEC:.1f} Hz)")
print(f"    - Ventana votacion deteccion: {VOTE_WINDOW_SEC*1000:.0f} ms ({1/VOTE_WINDOW_SEC:.1f} Hz)")
print(f"    - Canal deteccion: {DETECTION_CHANNEL}")


# --- FUNCIONES DE RED ---
def enviar_post_async(endpoint, data):
    def task():
        try:
            requests.post(f"{API_URL}{endpoint}", json=data, timeout=0.5)
        except Exception:
            pass
    threading.Thread(target=task, daemon=True).start()


# --- MEMORIA Y ESTADO COMPARTIDO ---
buffer_audio = np.zeros((MAX_BUFFER, cdoa.AUDIO['channels_hw']), dtype=np.float32)
acumulador_votacion = 0
acumulador_doa = 0
estado_sirena = False
probabilidad_actual = 0.0

# =========================================================
# BUCLE PRINCIPAL
# =========================================================
print("\nSISTEMA SIRA ACTIVO - Escuchando...")
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

        # 3. Desplazar buffer y anadir muestras
        buffer_audio = np.roll(buffer_audio, -n_nuevos, axis=0)
        buffer_audio[-n_nuevos:, :] = nuevo_bloque_float

        detection_manager.update_detection_models(
            detection_states,
            buffer_audio,
            n_nuevos,
            audio_channel=DETECTION_CHANNEL,
        )
        acumulador_votacion += n_nuevos
        acumulador_doa += n_nuevos

        # ---------------------------------------------------------
        # RELOJ 1: DoA (Ejecuta rapido y a su ritmo)
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
                if len(angulos_validos) > 0:
                    angulo_oficial = int(angulos_validos[0])
            else:
                tracker.actualizar([], [])

            # Formato exacto que pide la API para DatosDOA
            angulo_a_enviar = int(angulo_oficial) if angulo_oficial is not None else 0
            tendencia_a_enviar = "Estable" if angulo_oficial is not None else "Buscando"

            enviar_post_async("/update_doa", {
                "angulo": angulo_a_enviar,
                "tendencia": tendencia_a_enviar
            })

        # ---------------------------------------------------------
        # RELOJ 2: Votacion de deteccion
        # ---------------------------------------------------------
        if acumulador_votacion >= VOTE_STEP_SAMP:
            acumulador_votacion -= VOTE_STEP_SAMP

            estado_sirena, probabilidad_actual, latencia_ms, detalle_votacion = detection_manager.vote_detection_models(
                detection_states,
                buffer_audio,
                audio_channel=DETECTION_CHANNEL,
            )

            enviar_post_async("/update_deteccion", {
                "sirena": bool(estado_sirena),
                "probabilidad": float(probabilidad_actual),
                "tipo_vehiculo": "Ambulancia" if estado_sirena else "Ninguno",
                "latencia_inferencia_ms": float(latencia_ms),
                "fps": float(1.0 / VOTE_WINDOW_SEC),
                "t0_captura": float(t0_captura)
            })

            chunk_ui = buffer_audio[-VOTE_STEP_SAMP:, DETECTION_CHANNEL].astype(np.float32)
            salto_onda = max(1, int(len(chunk_ui) / 80))
            onda_ui = [float(x) for x in chunk_ui[::salto_onda]][:80]

            fft_complex = np.fft.rfft(chunk_ui)
            fft_mag = np.abs(fft_complex)[:64]
            if np.max(fft_mag) > 0:
                fft_mag = fft_mag / np.max(fft_mag)
            fft_ui = [float(x) for x in fft_mag]

            enviar_post_async("/update_audio", {
                "waveform_summary": onda_ui,
                "fft_data": fft_ui,
                "mfcc_features": []
            })

            print(detalle_votacion)

except KeyboardInterrupt:
    print("\nDeteniendo sistema SIRA...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
