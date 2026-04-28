from __future__ import annotations

import json
import os
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
from scipy.signal import resample_poly


SAMPLE_RATE = 16000
CHUNK_LENGTH_S = 0.5
OVERLAP_S = 0.0
N_FFT = 1024
HOP_LENGTH = 512
WINDOW = "hamming"
HPSS_MARGIN = 3.0
LINEAR_FREQ_BINS = 359
MEL_BINS = 128

DEFAULT_SPECTRAL_FRONTEND = "linear_stft"
DEFAULT_FEATURE_REPRESENTATION = "harmonic"
DEFAULT_SPECTROGRAM_NORMALIZATION = "frequency"
DEFAULT_DECISION_THRESHOLD = 0.7
DEFAULT_LABELS = ["background", "siren"]
EXPECTED_OUTPUT_MODE = "chunk_probability"
REQUIRED_POSTPROCESSING_KEYS = ("chunk_length_s",)


def resolve_path(path_value: str | os.PathLike, base_dir: str | os.PathLike) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def derive_postprocessing_path(model_path: str) -> str:
    base_path, _ = os.path.splitext(model_path)
    return f"{base_path}_postprocesado.json"


def get_missing_required_postprocessing_keys(saved_config: dict) -> list[str]:
    missing_keys = [
        key for key in REQUIRED_POSTPROCESSING_KEYS if key not in saved_config
    ]
    if "overlap_s" not in saved_config and "decision_step_s" not in saved_config:
        missing_keys.append("overlap_s|decision_step_s")
    return missing_keys


def compute_default_time_frames(
    chunk_length_s: float,
    sample_rate: int,
    hop_length: int,
) -> int:
    chunk_samples = int(round(chunk_length_s * sample_rate))
    return max(1, int(np.ceil(chunk_samples / hop_length)) + 1)


def compute_padded_chunk_samples(
    chunk_length_s: float,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    time_frames: int,
) -> int:
    return int(
        max(
            int(round(chunk_length_s * sample_rate)),
            n_fft,
            hop_length * max(0, time_frames - 1),
        )
    )


def build_default_runtime_config() -> dict:
    time_frames = compute_default_time_frames(CHUNK_LENGTH_S, SAMPLE_RATE, HOP_LENGTH)
    return {
        "sample_rate": SAMPLE_RATE,
        "chunk_length_s": CHUNK_LENGTH_S,
        "overlap_s": OVERLAP_S,
        "decision_step_s": CHUNK_LENGTH_S - OVERLAP_S,
        "hpss_margin": HPSS_MARGIN,
        "window": WINDOW,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "spectral_frontend": DEFAULT_SPECTRAL_FRONTEND,
        "feature_representation": DEFAULT_FEATURE_REPRESENTATION,
        "spectrogram_normalization": DEFAULT_SPECTROGRAM_NORMALIZATION,
        "linear_freq_bins": LINEAR_FREQ_BINS,
        "time_frames": time_frames,
        "mel_bins": MEL_BINS,
        "padded_chunk_samples": compute_padded_chunk_samples(
            CHUNK_LENGTH_S,
            SAMPLE_RATE,
            N_FFT,
            HOP_LENGTH,
            time_frames,
        ),
        "chunk_threshold": DEFAULT_DECISION_THRESHOLD,
        "labels": list(DEFAULT_LABELS),
        "output_mode": EXPECTED_OUTPUT_MODE,
        "_loaded_from_json": False,
        "_has_saved_frontend": False,
        "_has_saved_feature_representation": False,
        "_has_saved_linear_freq_bins": False,
        "_has_saved_time_frames": False,
        "_has_saved_mel_bins": False,
        "_has_saved_padded_chunk_samples": False,
    }


def finalize_runtime_config(runtime_config: dict) -> dict:
    runtime_config["sample_rate"] = int(runtime_config["sample_rate"])
    runtime_config["chunk_length_s"] = float(runtime_config["chunk_length_s"])
    runtime_config["overlap_s"] = float(runtime_config["overlap_s"])
    runtime_config["hpss_margin"] = float(runtime_config["hpss_margin"])
    runtime_config["window"] = str(runtime_config["window"])
    runtime_config["n_fft"] = int(runtime_config["n_fft"])
    runtime_config["hop_length"] = int(runtime_config["hop_length"])
    runtime_config["chunk_threshold"] = float(runtime_config["chunk_threshold"])
    runtime_config["spectral_frontend"] = str(runtime_config["spectral_frontend"])
    feature_representation = runtime_config.get("feature_representation")
    runtime_config["feature_representation"] = (
        None if feature_representation is None else str(feature_representation)
    )
    runtime_config["spectrogram_normalization"] = str(
        runtime_config["spectrogram_normalization"]
    )

    if runtime_config["n_fft"] <= 0 or runtime_config["hop_length"] <= 0:
        raise ValueError("n_fft y hop_length deben ser positivos.")

    if runtime_config["time_frames"] is None:
        runtime_config["time_frames"] = compute_default_time_frames(
            runtime_config["chunk_length_s"],
            runtime_config["sample_rate"],
            runtime_config["hop_length"],
        )
    else:
        runtime_config["time_frames"] = int(runtime_config["time_frames"])

    runtime_config["linear_freq_bins"] = int(
        runtime_config.get("linear_freq_bins", LINEAR_FREQ_BINS)
    )
    runtime_config["mel_bins"] = int(runtime_config.get("mel_bins", MEL_BINS))

    padded_chunk_samples = runtime_config.get("padded_chunk_samples")
    if padded_chunk_samples is None:
        runtime_config["padded_chunk_samples"] = compute_padded_chunk_samples(
            runtime_config["chunk_length_s"],
            runtime_config["sample_rate"],
            runtime_config["n_fft"],
            runtime_config["hop_length"],
            runtime_config["time_frames"],
        )
    else:
        runtime_config["padded_chunk_samples"] = int(padded_chunk_samples)

    chunk_samples = int(
        round(runtime_config["chunk_length_s"] * runtime_config["sample_rate"])
    )
    step_samples = int(
        round(
            (runtime_config["chunk_length_s"] - runtime_config["overlap_s"])
            * runtime_config["sample_rate"]
        )
    )
    if step_samples <= 0:
        raise ValueError("El paso entre chunks debe ser mayor que cero.")

    labels = [str(label) for label in runtime_config["labels"]]
    if len(labels) < 2:
        labels = list(DEFAULT_LABELS)

    runtime_config["labels"] = labels
    runtime_config["negative_label"] = labels[0]
    runtime_config["positive_label"] = labels[-1]
    runtime_config["chunk_samples"] = chunk_samples
    runtime_config["step_samples"] = step_samples
    runtime_config["decision_step_s"] = step_samples / runtime_config["sample_rate"]
    return runtime_config


def load_runtime_config(
    model_path: str,
    postprocessing_path: str | None = None,
    threshold_override: float | None = None,
) -> tuple[dict, str]:
    runtime_config = build_default_runtime_config()
    config_path = postprocessing_path or derive_postprocessing_path(model_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "No se encontro el JSON de postprocesado asociado al modelo.\n"
            f"Modelo: {model_path}\n"
            f"Esperado: {config_path}\n"
            "Copia junto al .keras el JSON que genera entrenar_modelo_margin_3.py "
            "o define MODEL['postprocessing_path'] en config_cnn.py."
        )

    with open(config_path, "r", encoding="utf-8") as file_handle:
        saved_config = json.load(file_handle)

    missing_required_keys = get_missing_required_postprocessing_keys(saved_config)
    if missing_required_keys:
        raise ValueError(
            "El JSON de postprocesado no incluye toda la configuracion minima "
            "necesaria para una inferencia segura.\n"
            f"Fichero: {config_path}\n"
            f"Campos ausentes: {', '.join(missing_required_keys)}"
        )

    runtime_config["_loaded_from_json"] = True
    runtime_config["sample_rate"] = saved_config.get(
        "sample_rate",
        runtime_config["sample_rate"],
    )
    runtime_config["chunk_length_s"] = saved_config.get(
        "chunk_length_s",
        runtime_config["chunk_length_s"],
    )
    runtime_config["overlap_s"] = saved_config.get(
        "overlap_s",
        runtime_config["overlap_s"],
    )
    if "overlap_s" not in saved_config and "decision_step_s" in saved_config:
        runtime_config["overlap_s"] = float(runtime_config["chunk_length_s"]) - float(
            saved_config["decision_step_s"]
        )

    runtime_config["chunk_threshold"] = saved_config.get(
        "recommended_chunk_threshold",
        runtime_config["chunk_threshold"],
    )
    runtime_config["hpss_margin"] = saved_config.get(
        "hpss_margin",
        runtime_config["hpss_margin"],
    )
    runtime_config["window"] = saved_config.get("window", runtime_config["window"])
    runtime_config["n_fft"] = saved_config.get("n_fft", runtime_config["n_fft"])
    runtime_config["hop_length"] = saved_config.get(
        "hop_length",
        runtime_config["hop_length"],
    )
    runtime_config["spectral_frontend"] = saved_config.get(
        "spectral_frontend",
        runtime_config["spectral_frontend"],
    )
    runtime_config["feature_representation"] = saved_config.get(
        "feature_representation",
        runtime_config["feature_representation"],
    )
    runtime_config["spectrogram_normalization"] = saved_config.get(
        "spectrogram_normalization",
        runtime_config["spectrogram_normalization"],
    )
    runtime_config["linear_freq_bins"] = saved_config.get(
        "linear_freq_bins",
        runtime_config["linear_freq_bins"],
    )
    runtime_config["time_frames"] = saved_config.get(
        "time_frames",
        runtime_config["time_frames"],
    )
    runtime_config["mel_bins"] = saved_config.get(
        "mel_bins",
        runtime_config["mel_bins"],
    )
    runtime_config["padded_chunk_samples"] = saved_config.get(
        "padded_chunk_samples",
        runtime_config["padded_chunk_samples"],
    )
    runtime_config["labels"] = list(
        saved_config.get("labels", runtime_config["labels"])
    )
    runtime_config["output_mode"] = saved_config.get(
        "output_mode",
        runtime_config["output_mode"],
    )

    runtime_config["_has_saved_frontend"] = "spectral_frontend" in saved_config
    runtime_config["_has_saved_feature_representation"] = (
        "feature_representation" in saved_config
    )
    runtime_config["_has_saved_linear_freq_bins"] = "linear_freq_bins" in saved_config
    runtime_config["_has_saved_time_frames"] = "time_frames" in saved_config
    runtime_config["_has_saved_mel_bins"] = "mel_bins" in saved_config
    runtime_config["_has_saved_padded_chunk_samples"] = (
        "padded_chunk_samples" in saved_config
    )

    if "spectrogram_normalization" not in saved_config:
        use_frequency_normalization = bool(
            saved_config.get("use_frequency_normalization", True)
        )
        runtime_config["spectrogram_normalization"] = (
            "frequency" if use_frequency_normalization else "none"
        )

    if threshold_override is not None:
        runtime_config["chunk_threshold"] = float(threshold_override)

    return finalize_runtime_config(runtime_config), config_path


def normalize_spectrogram(db_spectrogram: np.ndarray, mode: str) -> np.ndarray:
    if mode == "frequency":
        freq_mean = np.mean(db_spectrogram, axis=1, keepdims=True)
        freq_std = np.std(db_spectrogram, axis=1, keepdims=True)
        normalized = (db_spectrogram - freq_mean) / (freq_std + 1e-6)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    if mode == "minmax":
        min_db = np.min(db_spectrogram)
        max_db = np.max(db_spectrogram)
        if max_db - min_db > 0:
            normalized = (db_spectrogram - min_db) / (max_db - min_db)
        else:
            normalized = db_spectrogram - min_db
        return normalized.astype(np.float32)

    if mode == "none":
        return db_spectrogram.astype(np.float32)

    raise ValueError(
        "La normalizacion del espectrograma debe ser 'frequency', 'minmax' o 'none'."
    )


def pad_or_trim_time_frames(feature_map: np.ndarray, target_frames: int) -> np.ndarray:
    current_frames = feature_map.shape[1]
    if current_frames < target_frames:
        padding_shape = [(0, 0)] * feature_map.ndim
        padding_shape[1] = (0, target_frames - current_frames)
        return np.pad(feature_map, padding_shape)
    if current_frames > target_frames:
        return feature_map[:, :target_frames]
    return feature_map


def build_feature_tensor_from_linear_stft(
    stft_matrix: np.ndarray,
    runtime_config: dict,
) -> np.ndarray:
    freq_bins = runtime_config["linear_freq_bins"]
    time_frames = runtime_config["time_frames"]

    full_sliced = pad_or_trim_time_frames(stft_matrix[:freq_bins, :], time_frames)
    harmonic, _ = librosa.decompose.hpss(
        stft_matrix,
        margin=runtime_config["hpss_margin"],
    )
    harmonic_sliced = pad_or_trim_time_frames(harmonic[:freq_bins, :], time_frames)

    full_db = librosa.amplitude_to_db(np.abs(full_sliced), ref=np.max)
    harmonic_db = librosa.amplitude_to_db(np.abs(harmonic_sliced), ref=np.max)

    normalization_mode = runtime_config["spectrogram_normalization"]
    full_db = normalize_spectrogram(full_db, normalization_mode)
    harmonic_db = normalize_spectrogram(harmonic_db, normalization_mode)

    feature_representation = runtime_config["feature_representation"]
    if feature_representation == "harmonic":
        return np.expand_dims(harmonic_db, axis=-1).astype(np.float32)
    if feature_representation == "full":
        return np.expand_dims(full_db, axis=-1).astype(np.float32)
    if feature_representation == "harmonic_full":
        return np.stack([harmonic_db, full_db], axis=-1).astype(np.float32)

    raise ValueError(
        "feature_representation debe ser 'harmonic', 'full' o 'harmonic_full'."
    )


def extract_features_from_array(
    audio_chunk: np.ndarray,
    runtime_config: dict,
) -> np.ndarray:
    chunk_samples = runtime_config["chunk_samples"]
    audio_chunk = pad_or_trim(audio_chunk.astype(np.float32, copy=False), chunk_samples)
    audio_chunk_padded = pad_or_trim(
        audio_chunk,
        runtime_config["padded_chunk_samples"],
    ).astype(np.float32, copy=False)

    if runtime_config["spectral_frontend"] == "linear_stft":
        stft = librosa.stft(
            audio_chunk_padded,
            n_fft=runtime_config["n_fft"],
            hop_length=runtime_config["hop_length"],
            window=runtime_config["window"],
        )
        return build_feature_tensor_from_linear_stft(stft, runtime_config)

    if runtime_config["spectral_frontend"] == "log_mel":
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_chunk_padded,
            sr=runtime_config["sample_rate"],
            n_fft=runtime_config["n_fft"],
            hop_length=runtime_config["hop_length"],
            n_mels=runtime_config["mel_bins"],
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_db = pad_or_trim_time_frames(mel_db, runtime_config["time_frames"])
        mel_db = normalize_spectrogram(
            mel_db,
            runtime_config["spectrogram_normalization"],
        )
        return np.expand_dims(mel_db, axis=-1).astype(np.float32)

    raise ValueError("spectral_frontend debe ser 'linear_stft' o 'log_mel'.")


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    if len(audio) > target_length:
        return audio[:target_length]
    return audio


def resample_and_pad(
    audio_chunk: np.ndarray,
    orig_sr: int,
    target_sr: int,
    target_length: int,
) -> np.ndarray:
    if orig_sr == target_sr:
        return pad_or_trim(audio_chunk.astype(np.float32, copy=False), target_length)

    ratio = Fraction(target_sr, orig_sr).limit_denominator()
    resampled = resample_poly(audio_chunk, ratio.numerator, ratio.denominator).astype(
        np.float32
    )
    return pad_or_trim(resampled, target_length)


def load_model_for_inference(model_path: str) -> tf.keras.Model:
    import tensorflow as tf

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontro el modelo: {model_path}")
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise
        return load_keras_model_stripping_quantization_config(model_path)


def strip_quantization_config(config_node) -> None:
    if isinstance(config_node, dict):
        config_node.pop("quantization_config", None)
        for value in config_node.values():
            strip_quantization_config(value)
    elif isinstance(config_node, list):
        for item in config_node:
            strip_quantization_config(item)


def load_keras_model_stripping_quantization_config(model_path: str) -> tf.keras.Model:
    import tensorflow as tf

    with zipfile.ZipFile(model_path) as archive:
        model_config = json.loads(archive.read("config.json"))
        strip_quantization_config(model_config)
        weights_bytes = archive.read("model.weights.h5")

    model = tf.keras.models.model_from_json(json.dumps(model_config))
    weights_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp.write(weights_bytes)
            weights_path = tmp.name
        model.load_weights(weights_path)
    finally:
        if weights_path and os.path.exists(weights_path):
            os.remove(weights_path)

    return model


def infer_spectral_frontend_from_model(
    model_input_shape: tuple[int, int, int],
    runtime_config: dict,
) -> str:
    freq_bins, _, channels = model_input_shape
    if channels == 2:
        return "linear_stft"
    if runtime_config.get("_has_saved_feature_representation") and runtime_config.get(
        "feature_representation"
    ) in {"harmonic", "full"}:
        return "linear_stft"
    if freq_bins == runtime_config["mel_bins"]:
        return "log_mel"
    return "linear_stft"


def apply_model_shape_hints(model: tf.keras.Model, runtime_config: dict) -> dict:
    input_shape = tuple(model.input_shape[1:])
    if len(input_shape) != 3:
        raise ValueError(
            "Se esperaba un modelo con entrada 3D (freq, time, channels), "
            f"pero se ha encontrado input_shape={model.input_shape}."
        )

    freq_bins, time_frames, channels = input_shape
    if not runtime_config["_has_saved_frontend"]:
        runtime_config["spectral_frontend"] = infer_spectral_frontend_from_model(
            input_shape,
            runtime_config,
        )

    if runtime_config["spectral_frontend"] == "linear_stft":
        if not runtime_config["_has_saved_linear_freq_bins"]:
            runtime_config["linear_freq_bins"] = freq_bins
        if not runtime_config["_has_saved_time_frames"]:
            runtime_config["time_frames"] = time_frames
        if channels == 2:
            runtime_config["feature_representation"] = "harmonic_full"
        elif not runtime_config["_has_saved_feature_representation"]:
            runtime_config["feature_representation"] = "harmonic"
    elif runtime_config["spectral_frontend"] == "log_mel":
        if not runtime_config["_has_saved_mel_bins"]:
            runtime_config["mel_bins"] = freq_bins
        if not runtime_config["_has_saved_time_frames"]:
            runtime_config["time_frames"] = time_frames
        runtime_config["feature_representation"] = "log_mel"
    else:
        raise ValueError("spectral_frontend debe ser 'linear_stft' o 'log_mel'.")

    if not runtime_config["_has_saved_padded_chunk_samples"]:
        runtime_config["padded_chunk_samples"] = compute_padded_chunk_samples(
            runtime_config["chunk_length_s"],
            runtime_config["sample_rate"],
            runtime_config["n_fft"],
            runtime_config["hop_length"],
            runtime_config["time_frames"],
        )

    return finalize_runtime_config(runtime_config)


def expected_feature_shape(runtime_config: dict) -> tuple[int, int, int]:
    if runtime_config["spectral_frontend"] == "linear_stft":
        channels = (
            2 if runtime_config["feature_representation"] == "harmonic_full" else 1
        )
        return (
            runtime_config["linear_freq_bins"],
            runtime_config["time_frames"],
            channels,
        )
    if runtime_config["spectral_frontend"] == "log_mel":
        return (
            runtime_config["mel_bins"],
            runtime_config["time_frames"],
            1,
        )
    raise ValueError("spectral_frontend debe ser 'linear_stft' o 'log_mel'.")


def validate_model_against_runtime(
    model: tf.keras.Model,
    runtime_config: dict,
) -> None:
    input_shape = tuple(model.input_shape[1:])
    expected_shape = expected_feature_shape(runtime_config)
    if input_shape != expected_shape:
        raise ValueError(
            "El modelo espera input_shape="
            f"{input_shape}, pero el preprocesado produce {expected_shape}. "
            "Revisa el JSON de postprocesado o el modelo cargado."
        )

    output_shape = tuple(model.output_shape)
    if output_shape[-1] != 1:
        raise ValueError(
            "Se esperaba una salida escalar por chunk, pero el modelo devuelve "
            f"{output_shape}."
        )


def predict_chunk_probability(model: tf.keras.Model, features: np.ndarray) -> float:
    raw_prediction = np.asarray(model.predict(features[np.newaxis, ...], verbose=0))
    flattened_prediction = raw_prediction.reshape(-1)

    if flattened_prediction.size != 1:
        raise ValueError(
            "Se esperaba una unica probabilidad por chunk, "
            f"pero el modelo devolvio una salida con shape {raw_prediction.shape}."
        )

    return float(flattened_prediction[0])
