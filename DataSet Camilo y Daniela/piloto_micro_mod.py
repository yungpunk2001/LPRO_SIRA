import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import queue

# =========================
# CONFIGURACIÓN
# =========================
MODEL_PATH = "modelo_sirenas_margin_3.keras"
SR = 16000

CHUNK_SEC = 0.5
OVERLAP_SEC = 0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)                 # 8000 muestras en total
STEP_SAMPLES = int((CHUNK_SEC - OVERLAP_SEC) * SR)  # 7840 muestras nuevas por bloque
OVERLAP_SAMPLES = CHUNK_SAMPLES - STEP_SAMPLES      # 160 muestras arrastradas (solape)

PAD_TO = 8192
N_FFT = 1024
HOP_LENGTH = 512
WINDOW = "hamming"

HPSS_MARGIN = 3.0
FREQ_BINS = 359
TIME_FRAMES = 17
THRESHOLD = 0.9

# Cola para comunicación asíncrona entre el micrófono y la red neuronal
audio_queue = queue.Queue()

# =========================
# CAPTURA DE AUDIO CONTINUA
# =========================
def audio_callback(indata, frames, time, status):
    """Se ejecuta automáticamente cada vez que el micro capta 7840 muestras nuevas."""
    if status:
        print(status, flush=True)
    # Extraemos los datos brutos y los metemos en la cola de procesamiento
    audio_queue.put(indata.copy().squeeze())

# =========================
# EXTRACCIÓN DE CARACTERÍSTICAS
# =========================
def extract_features_from_array(y: np.ndarray) -> np.ndarray:
    if len(y) != CHUNK_SAMPLES:
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)))
        else:
            y = y[:CHUNK_SAMPLES]

    y_padded = np.pad(y, (0, PAD_TO - len(y)))
    S = librosa.stft(y_padded, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW)
    H, P = librosa.decompose.hpss(S, margin=HPSS_MARGIN)
    H_sliced = H[:FREQ_BINS, :TIME_FRAMES]
    H_db = librosa.amplitude_to_db(np.abs(H_sliced), ref=np.max)
    features = np.expand_dims(H_db, axis=-1).astype(np.float32)

    return features

# =========================
# FLUJO PRINCIPAL EN TIEMPO REAL
# =========================
def main():
    print(f"📦 Cargando modelo: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado y listo.\n")

    # Memoria inicial vacía para guardar los 10 ms finales del audio anterior
    buffer_solapamiento = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)

    print("🎙️ Iniciando escucha en tiempo real. Presiona Ctrl+C para detener la monitorización.")

    try:
        # Abrimos el micrófono exigiendo bloques exactos de 7840 muestras
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=STEP_SAMPLES, callback=audio_callback):
            while True:
                # 1. Esperamos a que llegue un bloque acústico nuevo
                nuevo_audio = audio_queue.get()

                # 2. Ensamblamos los 10ms anteriores con los datos nuevos (160 + 7840 = 8000 muestras)
                chunk_completo = np.concatenate((buffer_solapamiento, nuevo_audio))

                # 3. Guardamos los últimos 10ms de la trama actual para la siguiente vuelta
                buffer_solapamiento = chunk_completo[-OVERLAP_SAMPLES:]

                # 4. Transformación matemática a espectrograma
                feat = extract_features_from_array(chunk_completo)
                if feat is None:
                    continue

                # 5. Inferencia directa de la red neuronal
                x = feat[np.newaxis, ...]
                p = float(model.predict(x, verbose=0)[0, 0])

                # 6. Decisión inmediata en pantalla
                if p >= THRESHOLD:
                    print(f"🚨 ¡SIRENA DETECTADA! (Confianza: {p*100:.1f}%)")
                else:
                    print(f"✅ Ruido de fondo (Confianza: {p*100:.1f}%)")

    except KeyboardInterrupt:
        print("\n🛑 Escucha ininterrumpida detenida por el usuario.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado en el stream de audio: {e}")

if __name__ == "__main__":
    main()