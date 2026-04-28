import time
import threading
from pathlib import Path

import numpy as np
import pyaudio
import requests

# --- Importamos configuraciones ---
import config_doa as cdoa
import config_cnn as ccnn
import cnn_runtime
import trad_runtime

from doa_tracker_single import DOATrackerSingle
from doa_music import doa_music


BASE_DIR = Path(__file__).resolve().parent
API_URL = ccnn.MODEL['api_url']
SR = cdoa.AUDIO['rate']


def resolve_configured_directory(config_key: str, default_value: str) -> Path:
    configured_dir = ccnn.MODEL.get(config_key, default_value)
    return Path(cnn_runtime.resolve_path(configured_dir, BASE_DIR))


def resolve_cnn_directory() -> Path:
    legacy_dir = ccnn.MODEL.get('directory')
    return resolve_configured_directory('cnn_directory', legacy_dir or '../Modelos/CNN')


def resolve_traditional_directory() -> Path:
    return resolve_configured_directory('traditional_directory', '../Modelos/Tradicionales')


def discover_model_paths(model_dir: Path, pattern: str, model_kind: str) -> list[Path]:
    if not model_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos {model_kind}: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"La ruta de modelos {model_kind} no es una carpeta: {model_dir}")
    return sorted(path for path in model_dir.glob(pattern) if path.is_file())


def threshold_override_for(model_path: Path) -> float | None:
    overrides = ccnn.MODEL.get('threshold_overrides', {})
    if model_path.name in overrides:
        return float(overrides[model_path.name])
    if model_path.stem in overrides:
        return float(overrides[model_path.stem])
    global_threshold = ccnn.MODEL.get('threshold')
    return None if global_threshold is None else float(global_threshold)


def build_detection_state(
    *,
    kind: str,
    name: str,
    path: str,
    config_path: str,
    config: dict,
    model=None,
    bundle=None,
) -> dict:
    window_samp = int(round(config['chunk_length_s'] * SR))
    step_samp = int(round(config['decision_step_s'] * SR))
    if window_samp <= 0 or step_samp <= 0:
        raise ValueError(
            f"Configuracion temporal invalida para {name}: "
            f"window_samp={window_samp}, step_samp={step_samp}"
        )

    return {
        'kind': kind,
        'name': name,
        'path': path,
        'config_path': config_path,
        'model': model,
        'bundle': bundle,
        'config': config,
        'window_samp': window_samp,
        'step_samp': step_samp,
        'accumulator': 0,
        'period_results': [],
        'period_latency_ms': 0.0,
    }


def load_cnn_models(model_dir: Path) -> list[dict]:
    pattern = ccnn.MODEL.get('cnn_pattern', ccnn.MODEL.get('pattern', '**/*.keras'))
    states = []
    for model_path in discover_model_paths(model_dir, pattern, 'CNN'):
        model_path_str = str(model_path.resolve())
        runtime_config, config_path = cnn_runtime.load_runtime_config(
            model_path_str,
            threshold_override=threshold_override_for(model_path),
        )
        model = cnn_runtime.load_model_for_inference(model_path_str)
        runtime_config = cnn_runtime.apply_model_shape_hints(model, runtime_config)
        cnn_runtime.validate_model_against_runtime(model, runtime_config)

        states.append(
            build_detection_state(
                kind='cnn',
                name=f"CNN:{model_path.relative_to(model_dir).with_suffix('').as_posix()}",
                path=model_path_str,
                config_path=config_path,
                config=runtime_config,
                model=model,
            )
        )
    return states


def load_traditional_models(model_dir: Path) -> list[dict]:
    pattern = ccnn.MODEL.get(
        'traditional_pattern',
        '**/clasificador_tradicional_*_bundle.joblib',
    )
    states = []
    for bundle_path in discover_model_paths(model_dir, pattern, 'tradicionales'):
        bundle_path_str = str(bundle_path.resolve())
        bundle, runtime_config, config_path = trad_runtime.load_traditional_bundle(
            bundle_path_str
        )
        states.append(
            build_detection_state(
                kind='traditional',
                name=f"TRAD:{bundle_path.relative_to(model_dir).with_suffix('').as_posix()}",
                path=bundle_path_str,
                config_path=config_path,
                config=runtime_config,
                bundle=bundle,
            )
        )
    return states


def validate_detection_states(detection_states: list[dict]) -> float:
    if not detection_states:
        raise FileNotFoundError(
            "No se encontro ningun modelo valido de deteccion.\n"
            f"Carpeta CNN: {resolve_cnn_directory()}\n"
            f"Carpeta tradicionales: {resolve_traditional_directory()}\n"
            "Coloca las CNN en Modelos/CNN y los bundles tradicionales en "
            "Modelos/Tradicionales, cada uno con su JSON de postprocesado."
        )

    max_vote_window_sec = max(
        state['config']['chunk_length_s'] for state in detection_states
    )
    for state in detection_states:
        if state['config']['decision_step_s'] > max_vote_window_sec:
            raise ValueError(
                f"El paso de decision de {state['name']} "
                f"({state['config']['decision_step_s']:.3f}s) supera la ventana "
                f"global de votacion ({max_vote_window_sec:.3f}s)."
            )
    return max_vote_window_sec


def infer_cnn_state(state: dict, audio_buffer: np.ndarray) -> dict:
    chunk_cnn = audio_buffer[-state['window_samp']:, 0].astype(np.float32)
    runtime_config = state['config']
    chunk_modelo = cnn_runtime.resample_and_pad(
        chunk_cnn,
        orig_sr=SR,
        target_sr=runtime_config['sample_rate'],
        target_length=runtime_config['chunk_samples'],
    )
    features_cnn = cnn_runtime.extract_features_from_array(
        chunk_modelo,
        runtime_config,
    )
    probability = cnn_runtime.predict_chunk_probability(state['model'], features_cnn)
    return {
        'probability': probability,
        'positive': probability >= runtime_config['chunk_threshold'],
        'label': 'siren' if probability >= runtime_config['chunk_threshold'] else 'background',
    }


def infer_traditional_state(state: dict, audio_buffer: np.ndarray) -> dict:
    chunk_trad = audio_buffer[-state['window_samp']:, 0].astype(np.float32)
    runtime_config = state['config']
    chunk_modelo = cnn_runtime.resample_and_pad(
        chunk_trad,
        orig_sr=SR,
        target_sr=runtime_config['sample_rate'],
        target_length=runtime_config['chunk_samples'],
    )
    features = trad_runtime.extract_feature_vector(
        chunk_modelo,
        sr=runtime_config['sample_rate'],
    )
    return trad_runtime.predict_traditional(
        state['bundle'],
        features,
        runtime_config['chunk_threshold'],
    )


def infer_detection_state(state: dict, audio_buffer: np.ndarray) -> dict:
    if state['kind'] == 'cnn':
        return infer_cnn_state(state, audio_buffer)
    if state['kind'] == 'traditional':
        return infer_traditional_state(state, audio_buffer)
    raise ValueError(f"Tipo de modelo de deteccion no soportado: {state['kind']}")


def update_detection_models(
    detection_states: list[dict],
    audio_buffer: np.ndarray,
    n_samples: int,
) -> None:
    for state in detection_states:
        state['accumulator'] += n_samples
        if state['accumulator'] >= state['step_samp']:
            state['accumulator'] -= state['step_samp']
            t_inicio = time.time()
            result = infer_detection_state(state, audio_buffer)
            state['period_latency_ms'] += (time.time() - t_inicio) * 1000
            state['period_results'].append(result)


def select_period_result(state: dict, audio_buffer: np.ndarray) -> dict:
    if not state['period_results']:
        t_inicio = time.time()
        result = infer_detection_state(state, audio_buffer)
        state['period_latency_ms'] += (time.time() - t_inicio) * 1000
        state['period_results'].append(result)

    numeric_results = [
        result for result in state['period_results']
        if result.get('probability') is not None
    ]
    if numeric_results:
        return max(numeric_results, key=lambda item: item['probability'])

    positive = any(result.get('positive', False) for result in state['period_results'])
    return {
        'probability': None,
        'positive': positive,
        'label': 'siren' if positive else 'background',
    }


def format_vote_part(state: dict, result: dict) -> str:
    sign = '+' if result['positive'] else '-'
    probability = result.get('probability')
    if probability is None:
        return f"{state['name']}={result.get('label', 'background')}({sign})"
    return f"{state['name']}={probability*100:.1f}%({sign})"


def vote_detection_models(
    detection_states: list[dict],
    audio_buffer: np.ndarray,
) -> tuple[bool, float, float, str]:
    positive_votes = 0
    positive_probs = []
    binary_positive_votes = 0
    detail_parts = []
    total_latency_ms = 0.0

    for state in detection_states:
        result = select_period_result(state, audio_buffer)
        if result['positive']:
            positive_votes += 1
            if result.get('probability') is not None:
                positive_probs.append(float(result['probability']))
            else:
                binary_positive_votes += 1

        detail_parts.append(format_vote_part(state, result))
        total_latency_ms += state['period_latency_ms']
        state['period_results'].clear()
        state['period_latency_ms'] = 0.0

    total_models = len(detection_states)
    final_detection = positive_votes * 2 >= total_models
    final_probability = (
        float(sum(positive_probs) / len(positive_probs))
        if positive_probs
        else (1.0 if binary_positive_votes else 0.0)
    )
    detail = " | ".join(detail_parts)
    detail = (
        f"{detail} => votos {positive_votes}/{total_models} => "
        f"{'SIRENA' if final_detection else 'Ruido'} "
        f"(prob+ media={final_probability*100:.1f}%)"
    )
    return final_detection, final_probability, total_latency_ms, detail


# =========================================================
# CONFIGURACION DE MODELOS DE DETECCION
# =========================================================
CNN_MODEL_DIR = resolve_cnn_directory()
TRAD_MODEL_DIR = resolve_traditional_directory()

print(">>> [1/4] Cargando modelos de deteccion...")
cnn_states = load_cnn_models(CNN_MODEL_DIR)
traditional_states = load_traditional_models(TRAD_MODEL_DIR)
detection_states = cnn_states + traditional_states

VOTE_WINDOW_SEC = validate_detection_states(detection_states)
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
        f"Umbral: {config['chunk_threshold']:.3f}"
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

        update_detection_models(detection_states, buffer_audio, n_nuevos)
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

            estado_sirena, probabilidad_actual, latencia_ms, detalle_votacion = vote_detection_models(
                detection_states,
                buffer_audio,
            )

            enviar_post_async("/update_deteccion", {
                "sirena": bool(estado_sirena),
                "probabilidad": float(probabilidad_actual),
                "tipo_vehiculo": "Ambulancia" if estado_sirena else "Ninguno",
                "latencia_inferencia_ms": float(latencia_ms),
                "fps": float(1.0 / VOTE_WINDOW_SEC),
                "t0_captura": float(t0_captura)
            })

            chunk_ui = buffer_audio[-VOTE_STEP_SAMP:, 0].astype(np.float32)
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
