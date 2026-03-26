import json
import os

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf


# =========================
# Configuracion base
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "modelo_sirenas_margin_3.keras")

SAMPLE_RATE = 16000
CHUNK_LENGTH_S = 0.5
OVERLAP_S = 0.0
PAD_TO = 8192
N_FFT = 1024
HOP_LENGTH = 512
WINDOW = "hamming"
HPSS_MARGIN = 3.0
FREQ_BINS = 359
TIME_FRAMES = 17

USE_FREQUENCY_NORMALIZATION = True
REFERENCE_CHUNK_THRESHOLD = 0.5
REFERENCE_HIT_RATIO = 0.25
DEFAULT_LABELS = ["background", "siren"]
EXPECTED_OUTPUT_MODE = "chunk_probability"
POSITIVE_EMOJI_LABEL = "🚨 SIRENA"
NEGATIVE_EMOJI_LABEL = "✅ no_sirena"

RECORD_SECONDS = 8.0


def derive_postprocessing_path(model_path: str) -> str:
    base_path, _ = os.path.splitext(model_path)
    return f"{base_path}_postprocesado.json"


def load_runtime_config(model_path: str) -> tuple[dict, str | None]:
    """
    Construye la configuracion de inferencia a partir del script y, si existe,
    del JSON de postprocesado guardado por entrenar_modelo_margin_3.py.
    """
    runtime_config = {
        "sample_rate": SAMPLE_RATE,
        "chunk_length_s": CHUNK_LENGTH_S,
        "overlap_s": OVERLAP_S,
        "hpss_margin": HPSS_MARGIN,
        "use_frequency_normalization": USE_FREQUENCY_NORMALIZATION,
        "chunk_threshold": REFERENCE_CHUNK_THRESHOLD,
        "labels": list(DEFAULT_LABELS),
        "output_mode": EXPECTED_OUTPUT_MODE,
    }

    config_path = derive_postprocessing_path(model_path)
    if not os.path.exists(config_path):
        return finalize_runtime_config(runtime_config), None

    with open(config_path, "r", encoding="utf-8") as file_handle:
        saved_config = json.load(file_handle)

    runtime_config["chunk_length_s"] = float(saved_config.get("chunk_length_s", runtime_config["chunk_length_s"]))
    runtime_config["overlap_s"] = float(saved_config.get("overlap_s", runtime_config["overlap_s"]))
    runtime_config["chunk_threshold"] = float(
        saved_config.get("recommended_chunk_threshold", runtime_config["chunk_threshold"])
    )
    runtime_config["labels"] = list(saved_config.get("labels", runtime_config["labels"]))
    runtime_config["output_mode"] = saved_config.get("output_mode", runtime_config["output_mode"])
    runtime_config["use_frequency_normalization"] = bool(
        saved_config.get("use_frequency_normalization", runtime_config["use_frequency_normalization"])
    )

    return finalize_runtime_config(runtime_config), config_path


def finalize_runtime_config(runtime_config: dict) -> dict:
    chunk_samples = int(runtime_config["chunk_length_s"] * runtime_config["sample_rate"])
    step_samples = int((runtime_config["chunk_length_s"] - runtime_config["overlap_s"]) * runtime_config["sample_rate"])
    if step_samples <= 0:
        raise ValueError("El paso entre chunks debe ser mayor que cero.")

    labels = runtime_config["labels"]
    if len(labels) < 2:
        labels = list(DEFAULT_LABELS)

    runtime_config["labels"] = labels
    runtime_config["negative_label"] = labels[0]
    runtime_config["positive_label"] = labels[-1]
    runtime_config["chunk_samples"] = chunk_samples
    runtime_config["step_samples"] = step_samples
    runtime_config["decision_step_s"] = step_samples / runtime_config["sample_rate"]
    return runtime_config


def record_audio(seconds: float, sr: int) -> np.ndarray:
    print(f"Grabando {seconds:.1f}s a {sr} Hz...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Grabacion finalizada.")
    return audio.squeeze()


def normalize_spectrogram(db_spectrogram: np.ndarray) -> np.ndarray:
    """
    Replica la normalizacion por banda activada en entrenar_modelo_margin_3.py.
    """
    freq_mean = np.mean(db_spectrogram, axis=1, keepdims=True)
    freq_std = np.std(db_spectrogram, axis=1, keepdims=True)
    normalized = (db_spectrogram - freq_mean) / (freq_std + 1e-6)
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def extract_features_from_array(audio_chunk: np.ndarray, runtime_config: dict) -> np.ndarray | None:
    """
    Convierte un chunk mono en la entrada (359, 17, 1) usada por la CNN actual.
    """
    chunk_samples = runtime_config["chunk_samples"]

    if len(audio_chunk) != chunk_samples:
        if len(audio_chunk) < chunk_samples:
            return None
        audio_chunk = audio_chunk[:chunk_samples]

    audio_chunk = audio_chunk.astype(np.float32, copy=False)
    audio_chunk_padded = np.pad(audio_chunk, (0, PAD_TO - len(audio_chunk)))
    stft = librosa.stft(audio_chunk_padded, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW)

    harmonic, _ = librosa.decompose.hpss(stft, margin=runtime_config["hpss_margin"])
    harmonic_sliced = harmonic[:FREQ_BINS, :TIME_FRAMES]

    harmonic_db = librosa.amplitude_to_db(np.abs(harmonic_sliced), ref=np.max)
    if runtime_config["use_frequency_normalization"]:
        harmonic_db = normalize_spectrogram(harmonic_db)

    return np.expand_dims(harmonic_db, axis=-1).astype(np.float32)


def sliding_chunks(audio: np.ndarray, runtime_config: dict) -> tuple[list[np.ndarray], list[float]]:
    """
    Genera chunks con el mismo paso temporal que usa el entrenamiento.
    """
    chunk_samples = runtime_config["chunk_samples"]
    step_samples = runtime_config["step_samples"]
    sample_rate = runtime_config["sample_rate"]

    chunks = []
    times = []
    for start in range(0, max(1, len(audio) - chunk_samples + 1), step_samples):
        audio_chunk = audio[start : start + chunk_samples]
        if len(audio_chunk) < chunk_samples:
            continue
        chunks.append(audio_chunk)
        times.append(start / sample_rate)
    return chunks, times


def load_model_for_inference(model_path: str) -> tf.keras.Model:
    """
    El piloto solo necesita inferencia, asi que se carga sin recompilar.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No se encontro el modelo en la ruta esperada. "
            "Entrena primero margin_3 o actualiza MODEL_PATH en piloto_micro.py."
        )

    print(f"Cargando modelo: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def predict_chunk_probability(model: tf.keras.Model, features: np.ndarray) -> float:
    """
    La red actual debe devolver una unica probabilidad por chunk.
    """
    raw_prediction = np.asarray(model.predict(features[np.newaxis, ...], verbose=0))
    flattened_prediction = raw_prediction.reshape(-1)

    if flattened_prediction.size != 1:
        raise ValueError(
            "Se esperaba una unica probabilidad por chunk, "
            f"pero el modelo devolvio una salida con shape {raw_prediction.shape}."
        )

    return float(flattened_prediction[0])


def format_visual_label(is_positive: bool) -> str:
    return POSITIVE_EMOJI_LABEL if is_positive else NEGATIVE_EMOJI_LABEL


def main() -> None:
    model_path = os.path.abspath(MODEL_PATH)
    runtime_config, config_path = load_runtime_config(model_path)

    model = load_model_for_inference(model_path)
    print("Modelo cargado.")
    print("Input esperado:", model.input_shape, "| Output:", model.output_shape)

    if config_path is not None:
        print(f"Usando configuracion de postprocesado: {config_path}")
    else:
        print("No se encontro JSON de postprocesado. Se usan los parametros de entrenar_modelo_margin_3.py.")

    print(
        "Chunk: {chunk:.2f}s | Solape: {overlap:.2f}s | Paso: {step:.2f}s | Umbral ref.: {threshold:.2f}".format(
            chunk=runtime_config["chunk_length_s"],
            overlap=runtime_config["overlap_s"],
            step=runtime_config["decision_step_s"],
            threshold=runtime_config["chunk_threshold"],
        )
    )
    print(
        "Salida esperada: {output_mode} | Clase positiva: {positive_label}".format(
            output_mode=runtime_config["output_mode"],
            positive_label=runtime_config["positive_label"],
        )
    )
    if runtime_config["output_mode"] != EXPECTED_OUTPUT_MODE:
        print(
            "Aviso: el JSON indica un output_mode distinto al esperado. "
            "Este piloto asumira igualmente una probabilidad escalar por chunk."
        )

    audio = record_audio(RECORD_SECONDS, runtime_config["sample_rate"])

    chunks, times = sliding_chunks(audio, runtime_config)
    if not chunks:
        print("No se pudieron generar chunks validos. La grabacion es demasiado corta.")
        return

    print(
        "\nAnalizando {count} chunks de {chunk:.2f}s con paso {step:.2f}s...\n".format(
            count=len(chunks),
            chunk=runtime_config["chunk_length_s"],
            step=runtime_config["decision_step_s"],
        )
    )

    probabilities = []
    chunk_hits = 0

    for index, (chunk, start_time) in enumerate(zip(chunks, times), start=1):
        features = extract_features_from_array(chunk, runtime_config)
        if features is None:
            continue

        probability = predict_chunk_probability(model, features)
        probabilities.append(probability)

        is_positive = probability >= runtime_config["chunk_threshold"]
        chunk_hits += int(is_positive)
        reference_label = runtime_config["positive_label"] if is_positive else runtime_config["negative_label"]
        visual_label = format_visual_label(is_positive)

        print(
            "[{index:02d}] t={start:5.2f}s -> {visual_label} | p({positive_label})={prob:.3f} | ref={reference_label}".format(
                index=index,
                start=start_time,
                visual_label=visual_label,
                positive_label=runtime_config["positive_label"],
                prob=probability,
                reference_label=reference_label,
            )
        )

    if not probabilities:
        print("No hubo predicciones validas.")
        return

    hit_ratio = chunk_hits / len(probabilities)
    probability_mean = float(np.mean(probabilities))
    probability_max = float(np.max(probabilities))
    ratio_reference = hit_ratio >= REFERENCE_HIT_RATIO

    print("\n========================")
    print(
        "Chunks por encima del umbral de referencia: "
        f"{chunk_hits}/{len(probabilities)} -> {hit_ratio * 100:.1f}%"
    )
    print(
        "Media p({positive_label}): {probability_mean:.3f} | Max p({positive_label}): {probability_max:.3f}".format(
            positive_label=runtime_config["positive_label"],
            probability_mean=probability_mean,
            probability_max=probability_max,
        )
    )
    print(
        "Indicador auxiliar por ratio (>= {ratio:.2f}): {label}".format(
            ratio=REFERENCE_HIT_RATIO,
            label=format_visual_label(ratio_reference),
        )
    )
    print(
        "Nota: el modelo actual se interpreta como probabilidad por chunk. "
        "La decision binaria final debe hacerse con logica causal externa."
    )
    print("========================\n")


if __name__ == "__main__":
    main()
