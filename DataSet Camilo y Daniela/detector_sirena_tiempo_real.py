from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
from fractions import Fraction

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from scipy.signal import resample_poly


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    r"C:\Users\marti\Documents\UVigo\4_Cuarto\LPRO\Detección\DataSet Camilo y Daniela\Modelos Atlas\barrido_margin_3_20260406_205256\artefactos\exp_006_solo_espectrograma_completo_20260408_130621.keras",
)

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
PREFERRED_HOSTAPIS = (
    "Windows WASAPI",
    "Windows DirectSound",
    "MME",
    "Windows WDM-KS",
)
REQUIRED_POSTPROCESSING_KEYS = (
    "chunk_length_s",
)

# El valor hardcodeado original puede quedar obsoleto o incluso romperse por
# rutas absolutas con caracteres no ASCII. A partir de aqui se resuelve en
# tiempo de ejecucion el modelo mas reciente que tenga su JSON asociado.
MODELS_ROOT = os.path.join(SCRIPT_DIR, "Modelos Atlas")
DEFAULT_MODEL_PATH = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Captura audio de una tarjeta de entrada y clasifica cada chunk "
            "segun el preprocesado con el que se entreno el modelo de Keras."
        )
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Ruta al archivo .keras del modelo. Si no se indica, se intenta "
            "usar el mas reciente que tenga JSON de postprocesado."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Muestra las entradas de audio disponibles y sale.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="Indice del dispositivo de entrada a usar.",
    )
    parser.add_argument(
        "--device-name",
        help="Subcadena del nombre del dispositivo de entrada a usar.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help=(
            "Canales a abrir del dispositivo. Si eliges un canal N, el script "
            "abrira al menos N+1 canales. Por defecto: 1."
        ),
    )
    parser.add_argument(
        "--device-channel",
        type=int,
        help=(
            "Canal concreto del dispositivo que se analizara. Si no se indica, "
            "se muestra una segunda pantalla para escogerlo."
        ),
    )
    parser.add_argument(
        "--capture-samplerate",
        type=float,
        help=(
            "Frecuencia de captura del dispositivo. Si no se indica, se "
            "intenta 16000 Hz y, si no es valido, se usa la frecuencia "
            "por defecto del dispositivo."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Umbral binario por chunk. Si no se indica, se usa el valor "
            "recomendado en el JSON de postprocesado; si no existe, "
            f"se usa {DEFAULT_DECISION_THRESHOLD:.2f}."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Procesa como mucho este numero de chunks y termina.",
    )
    return parser.parse_args()


def derive_postprocessing_path(model_path: str) -> str:
    base_path, _ = os.path.splitext(model_path)
    return f"{base_path}_postprocesado.json"


def list_available_model_paths(
    models_root: str = MODELS_ROOT,
    require_postprocessing: bool = True,
) -> list[str]:
    if not os.path.isdir(models_root):
        return []

    model_paths: list[str] = []
    for current_root, _, filenames in os.walk(models_root):
        for filename in filenames:
            if not filename.lower().endswith(".keras"):
                continue

            model_path = os.path.join(current_root, filename)
            if require_postprocessing and not os.path.exists(
                derive_postprocessing_path(model_path)
            ):
                continue
            model_paths.append(model_path)

    model_paths.sort(
        key=lambda path: (os.path.getmtime(path), path.lower()),
        reverse=True,
    )
    return model_paths


def find_default_model_path() -> str | None:
    available_models = list_available_model_paths(require_postprocessing=True)
    if available_models:
        return available_models[0]
    return None


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


def load_runtime_config(model_path: str) -> tuple[dict, str | None]:
    runtime_config = build_default_runtime_config()

    config_path = derive_postprocessing_path(model_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "No se encontro el JSON de postprocesado asociado al modelo.\n"
            f"Modelo: {model_path}\n"
            f"Esperado: {config_path}\n"
            "El detector exige ese JSON para no reutilizar por error una "
            "configuracion temporal incompatible con la del entrenamiento."
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
    runtime_config["mel_bins"] = saved_config.get("mel_bins", runtime_config["mel_bins"])
    runtime_config["padded_chunk_samples"] = saved_config.get(
        "padded_chunk_samples",
        runtime_config["padded_chunk_samples"],
    )
    runtime_config["labels"] = list(saved_config.get("labels", runtime_config["labels"]))
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
        use_frequency_normalization = bool(saved_config.get("use_frequency_normalization", True))
        runtime_config["spectrogram_normalization"] = (
            "frequency" if use_frequency_normalization else "none"
        )

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


def extract_features_from_array(audio_chunk: np.ndarray, runtime_config: dict) -> np.ndarray:
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

    raise ValueError(
        "spectral_frontend debe ser 'linear_stft' o 'log_mel'."
    )


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    if len(audio) > target_length:
        return audio[:target_length]
    return audio


def load_model_for_inference(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontro el modelo: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


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
        raise ValueError(
            "spectral_frontend debe ser 'linear_stft' o 'log_mel'."
        )

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
        channels = 2 if runtime_config["feature_representation"] == "harmonic_full" else 1
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
    raise ValueError(
        "spectral_frontend debe ser 'linear_stft' o 'log_mel'."
    )


def validate_model_against_runtime(model: tf.keras.Model, runtime_config: dict) -> None:
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


def list_input_devices() -> tuple[list[dict], int | None]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_input = None
    if sd.default.device is not None:
        default_input = sd.default.device[0]
        if default_input is not None and default_input < 0:
            default_input = None

    input_devices = []
    for index, device in enumerate(devices):
        max_input_channels = int(device["max_input_channels"])
        if max_input_channels <= 0:
            continue

        hostapi_index = int(device["hostapi"])
        hostapi_name = (
            hostapis[hostapi_index]["name"]
            if 0 <= hostapi_index < len(hostapis)
            else "desconocido"
        )
        input_devices.append(
            {
                "index": index,
                "name": str(device["name"]),
                "hostapi": hostapi_name,
                "max_input_channels": max_input_channels,
                "default_samplerate": float(device["default_samplerate"]),
            }
        )

    return input_devices, default_input


def print_input_devices(input_devices: list[dict], default_input: int | None) -> None:
    if not input_devices:
        print("No se han encontrado dispositivos de entrada.")
        return

    print("Dispositivos de entrada disponibles:\n")
    for device in input_devices:
        marker = " (default)" if device["index"] == default_input else ""
        print(
            "[{index}] {name} | hostapi={hostapi} | canales={channels} | "
            "sr_defecto={samplerate:.0f}{marker}".format(
                index=device["index"],
                name=device["name"],
                hostapi=device["hostapi"],
                channels=device["max_input_channels"],
                samplerate=device["default_samplerate"],
                marker=marker,
            )
        )


def print_device_channels(device_info: dict) -> None:
    max_channels = int(device_info["max_input_channels"])

    print(
        "\nCanales disponibles para el dispositivo "
        f"[{device_info['index']}] {device_info['name']}:\n"
    )
    for channel_index in range(max_channels):
        print(f"[{channel_index}] Canal {channel_index}")

    if max_channels == 6:
        print(
            "\nReferencia ReSpeaker XVF3800 USB (firmware 6ch, segun la wiki):\n"
            "  canal 0 -> Audio procesado (Conference)\n"
            "  canal 1 -> Audio procesado (ASR)\n"
            "  canal 2 -> Mic 0 raw\n"
            "  canal 3 -> Mic 1 raw\n"
            "  canal 4 -> Mic 2 raw\n"
            "  canal 5 -> Mic 3 raw"
        )
    elif max_channels == 2:
        print(
            "\nReferencia ReSpeaker XVF3800 USB (firmware 2ch, segun la wiki):\n"
            "  canal 0 -> Audio procesado (Conference)\n"
            "  canal 1 -> Audio procesado (ASR)"
        )


def prompt_for_device(input_devices: list[dict], default_input: int | None) -> int:
    valid_indices = {device["index"] for device in input_devices}
    if not valid_indices:
        raise RuntimeError("No hay dispositivos de entrada disponibles.")

    print_input_devices(input_devices, default_input)
    default_prompt = default_input if default_input in valid_indices else sorted(valid_indices)[0]

    while True:
        raw_value = input(
            f"\nSelecciona el indice del dispositivo [{default_prompt}]: "
        ).strip()
        if not raw_value:
            return default_prompt

        try:
            chosen_index = int(raw_value)
        except ValueError:
            print("Debes escribir un numero entero.")
            continue

        if chosen_index in valid_indices:
            return chosen_index

        print("Indice no valido. Elige uno de la lista.")


def prompt_for_channel(device_info: dict) -> int:
    max_channels = int(device_info["max_input_channels"])
    print_device_channels(device_info)

    while True:
        raw_value = input("\nSelecciona el canal del dispositivo [0]: ").strip()
        if not raw_value:
            return 0

        try:
            chosen_channel = int(raw_value)
        except ValueError:
            print("Debes escribir un numero entero.")
            continue

        if 0 <= chosen_channel < max_channels:
            return chosen_channel

        print(f"Canal no valido. Debe estar entre 0 y {max_channels - 1}.")


def resolve_device_index(
    args: argparse.Namespace,
    input_devices: list[dict],
    default_input: int | None,
) -> int:
    valid_indices = {device["index"] for device in input_devices}
    if not valid_indices:
        raise RuntimeError("No hay dispositivos de entrada disponibles.")

    if args.device_index is not None:
        if args.device_index not in valid_indices:
            raise ValueError(
                f"El dispositivo {args.device_index} no es una entrada valida."
            )
        return args.device_index

    if args.device_name:
        matches = [
            device
            for device in input_devices
            if args.device_name.lower() in device["name"].lower()
        ]
        if not matches:
            raise ValueError(
                f"No hay ninguna entrada que contenga '{args.device_name}'."
            )
        if len(matches) > 1:
            matched_indices = ", ".join(str(device["index"]) for device in matches)
            raise ValueError(
                "La busqueda por nombre es ambigua. Coincide con indices: "
                f"{matched_indices}."
            )
        return matches[0]["index"]

    if sys.stdin.isatty():
        return prompt_for_device(input_devices, default_input)

    if default_input in valid_indices:
        return default_input

    return sorted(valid_indices)[0]


def resolve_device_channel(
    args: argparse.Namespace,
    device_info: dict,
) -> int:
    max_channels = int(device_info["max_input_channels"])

    if args.device_channel is not None:
        if not 0 <= args.device_channel < max_channels:
            raise ValueError(
                f"El canal {args.device_channel} no existe para este dispositivo."
            )
        return args.device_channel

    if sys.stdin.isatty():
        return prompt_for_channel(device_info)

    return 0


def canonical_device_name(device_name: str) -> str:
    name = device_name.split("  (")[0].strip().lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def is_respeaker_device(device_info: dict) -> bool:
    return "respeaker" in canonical_device_name(device_info["name"])


def compute_stream_channels(
    requested_channels: int,
    device_info: dict,
    device_channel: int,
) -> int:
    minimum_channels = max(requested_channels, device_channel + 1)

    # Para arrays multicanal como ReSpeaker conviene abrir todo el bus USB y
    # despues seleccionar el canal, para no depender de como PortAudio
    # comprime o reindexa canales al abrir menos de los disponibles.
    if is_respeaker_device(device_info) and int(device_info["max_input_channels"]) > 2:
        return int(device_info["max_input_channels"])

    return minimum_channels


def choose_capture_samplerate(
    device_index: int,
    stream_channels: int,
    target_samplerate: int,
    requested_samplerate: float | None,
) -> int:
    if requested_samplerate is not None:
        samplerate = int(round(requested_samplerate))
        sd.check_input_settings(
            device=device_index,
            channels=stream_channels,
            samplerate=samplerate,
            dtype="float32",
        )
        return samplerate

    try:
        sd.check_input_settings(
            device=device_index,
            channels=stream_channels,
            samplerate=target_samplerate,
            dtype="float32",
        )
        return target_samplerate
    except Exception:
        device_info = sd.query_devices(device_index)
        fallback_samplerate = int(round(float(device_info["default_samplerate"])))
        sd.check_input_settings(
            device=device_index,
            channels=stream_channels,
            samplerate=fallback_samplerate,
            dtype="float32",
        )
        return fallback_samplerate


def can_open_input_stream(
    device_index: int,
    stream_channels: int,
    samplerate: int,
    blocksize: int,
) -> tuple[bool, str | None]:
    def noop_callback(indata, frames, time_info, status) -> None:
        del indata, frames, time_info, status

    try:
        with sd.InputStream(
            device=device_index,
            channels=stream_channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=noop_callback,
        ):
            sd.sleep(50)
        return True, None
    except Exception as exc:
        return False, str(exc)


def resolve_working_capture_device(
    selected_device_info: dict,
    input_devices: list[dict],
    stream_channels: int,
    requested_samplerate: float | None,
    target_samplerate: int,
    decision_step_s: float,
) -> tuple[dict, int]:
    selected_key = canonical_device_name(selected_device_info["name"])
    selected_hostapi = selected_device_info["hostapi"]

    alternative_devices = [
        device
        for device in input_devices
        if canonical_device_name(device["name"]) == selected_key
        and int(device["max_input_channels"]) >= stream_channels
    ]

    def hostapi_priority(device: dict) -> int:
        try:
            return PREFERRED_HOSTAPIS.index(device["hostapi"])
        except ValueError:
            return len(PREFERRED_HOSTAPIS)

    alternative_devices.sort(
        key=lambda device: (
            0 if device["index"] == selected_device_info["index"] else 1,
            hostapi_priority(device),
            -int(device["max_input_channels"]),
        )
    )

    errors = []
    for device_info in alternative_devices:
        try:
            capture_samplerate = choose_capture_samplerate(
                device_index=device_info["index"],
                stream_channels=stream_channels,
                target_samplerate=target_samplerate,
                requested_samplerate=requested_samplerate,
            )
        except Exception as exc:
            errors.append(
                f"[{device_info['index']}] {device_info['name']} @ {device_info['hostapi']}: {exc}"
            )
            continue

        capture_step_samples = int(round(decision_step_s * capture_samplerate))
        can_open, error_message = can_open_input_stream(
            device_index=device_info["index"],
            stream_channels=stream_channels,
            samplerate=capture_samplerate,
            blocksize=capture_step_samples,
        )
        if can_open:
            if device_info["index"] != selected_device_info["index"]:
                print(
                    "Aviso: la variante seleccionada en hostapi "
                    f"{selected_hostapi} no se deja abrir con sounddevice. "
                    f"Se usara [{device_info['index']}] {device_info['name']} "
                    f"en {device_info['hostapi']}."
                )
            return device_info, capture_samplerate

        errors.append(
            f"[{device_info['index']}] {device_info['name']} @ {device_info['hostapi']}: "
            f"{error_message}"
        )

    raise RuntimeError(
        "No se ha podido abrir ninguna variante compatible del dispositivo.\n"
        + "\n".join(errors)
    )


def select_input_channel(block: np.ndarray, device_channel: int) -> np.ndarray:
    if block.ndim == 1:
        if device_channel != 0:
            raise ValueError(
                f"Solo se ha abierto 1 canal, asi que no se puede leer el canal {device_channel}."
            )
        return block.astype(np.float32, copy=False)

    if device_channel >= block.shape[1]:
        raise ValueError(
            f"El bloque recibido tiene {block.shape[1]} canal(es), "
            f"pero se ha pedido el canal {device_channel}."
        )

    return block[:, device_channel].astype(np.float32, copy=False)


def resample_chunk(
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


def build_stream_callback(audio_queue: queue.Queue) -> callable:
    def audio_callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            print(
                "[audio] Cola llena; se descarta un bloque de entrada.",
                file=sys.stderr,
            )

    return audio_callback


def run_stream(
    model: tf.keras.Model,
    runtime_config: dict,
    device_index: int,
    capture_samplerate: int,
    stream_channels: int,
    device_channel: int,
    max_chunks: int | None,
) -> None:
    capture_chunk_samples = int(round(runtime_config["chunk_length_s"] * capture_samplerate))
    capture_step_samples = int(round(runtime_config["decision_step_s"] * capture_samplerate))

    if capture_chunk_samples <= 0 or capture_step_samples <= 0:
        raise ValueError("La configuracion temporal de captura no es valida.")

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    audio_buffer = np.empty(0, dtype=np.float32)
    chunk_index = 0

    stream_callback = build_stream_callback(audio_queue)

    with sd.InputStream(
        device=device_index,
        channels=stream_channels,
        samplerate=capture_samplerate,
        blocksize=capture_step_samples,
        dtype="float32",
        callback=stream_callback,
    ):
        print("\nCaptura iniciada. Pulsa Ctrl+C para detener.\n")

        while True:
            block = audio_queue.get()
            selected_block = select_input_channel(block, device_channel)
            audio_buffer = np.concatenate([audio_buffer, selected_block])

            while len(audio_buffer) >= capture_chunk_samples:
                raw_chunk = audio_buffer[:capture_chunk_samples].copy()
                audio_buffer = audio_buffer[capture_step_samples:]

                model_chunk = resample_chunk(
                    raw_chunk,
                    orig_sr=capture_samplerate,
                    target_sr=runtime_config["sample_rate"],
                    target_length=runtime_config["chunk_samples"],
                )
                features = extract_features_from_array(model_chunk, runtime_config)
                probability = predict_chunk_probability(model, features)

                is_positive = probability >= runtime_config["chunk_threshold"]
                label = (
                    runtime_config["positive_label"]
                    if is_positive
                    else runtime_config["negative_label"]
                )
                chunk_start_s = chunk_index * runtime_config["decision_step_s"]
                rms = float(np.sqrt(np.mean(np.square(model_chunk))))

                print(
                    "[{index:05d}] t={time:8.2f}s | p({positive})={prob:.3f} | "
                    "decision={label} | rms={rms:.4f}".format(
                        index=chunk_index + 1,
                        time=chunk_start_s,
                        positive=runtime_config["positive_label"],
                        prob=probability,
                        label=label,
                        rms=rms,
                    ),
                    flush=True,
                )

                chunk_index += 1
                if max_chunks is not None and chunk_index >= max_chunks:
                    return


def describe_runtime(
    model_path: str,
    config_path: str | None,
    device_info: dict,
    runtime_config: dict,
    capture_samplerate: int,
    stream_channels: int,
    device_channel: int,
    model: tf.keras.Model,
) -> None:
    print(f"Modelo: {model_path}")
    if config_path:
        print(f"Postprocesado: {config_path}")
    else:
        print("Postprocesado: no encontrado; se usan valores por defecto del script.")

    print(
        "Input del modelo: {input_shape} | Output: {output_shape}".format(
            input_shape=model.input_shape,
            output_shape=model.output_shape,
        )
    )
    print(
        "Dispositivo [{index}]: {name} | hostapi={hostapi}".format(
            index=device_info["index"],
            name=device_info["name"],
            hostapi=device_info["hostapi"],
        )
    )
    print(
        "Captura: {capture_sr} Hz, {stream_channels} canal(es) abiertos | "
        "Canal analizado: {device_channel} | Modelo: {model_sr} Hz".format(
            capture_sr=capture_samplerate,
            stream_channels=stream_channels,
            device_channel=device_channel,
            model_sr=runtime_config["sample_rate"],
        )
    )
    print(
        "Chunk: {chunk:.2f}s | Solape: {overlap:.2f}s | Paso: {step:.2f}s".format(
            chunk=runtime_config["chunk_length_s"],
            overlap=runtime_config["overlap_s"],
            step=runtime_config["decision_step_s"],
        )
    )
    print(
        "Frontend: {frontend} | Representacion: {representation} | "
        "Normalizacion: {normalization} | Umbral: {threshold:.2f}".format(
            frontend=runtime_config["spectral_frontend"],
            representation=runtime_config["feature_representation"],
            normalization=runtime_config["spectrogram_normalization"],
            threshold=runtime_config["chunk_threshold"],
        )
    )
    print(
        "Preprocesado: n_fft={n_fft} | hop={hop} | frames={frames} | "
        "linear_bins={linear_bins} | mel_bins={mel_bins} | padded_samples={padded}".format(
            n_fft=runtime_config["n_fft"],
            hop=runtime_config["hop_length"],
            frames=runtime_config["time_frames"],
            linear_bins=runtime_config["linear_freq_bins"],
            mel_bins=runtime_config["mel_bins"],
            padded=runtime_config["padded_chunk_samples"],
        )
    )
    if runtime_config["output_mode"] != EXPECTED_OUTPUT_MODE:
        print(
            "Aviso: el JSON indica output_mode="
            f"{runtime_config['output_mode']}. El script asumira una probabilidad "
            "escalar por chunk."
        )


def main() -> None:
    args = parse_args()

    resolved_model_path = args.model_path
    if not resolved_model_path:
        resolved_model_path = find_default_model_path()
        if resolved_model_path is None:
            raise FileNotFoundError(
                "No se encontro ningun modelo .keras con su JSON de "
                "postprocesado dentro de 'Modelos Atlas'."
            )

    model_path = os.path.abspath(resolved_model_path)
    runtime_config, config_path = load_runtime_config(model_path)

    input_devices, default_input = list_input_devices()
    if args.list_devices:
        print_input_devices(input_devices, default_input)
        return

    device_index = resolve_device_index(args, input_devices, default_input)
    device_lookup = {device["index"]: device for device in input_devices}
    device_info = device_lookup[device_index]
    device_channel = resolve_device_channel(args, device_info)

    if args.channels <= 0:
        raise ValueError("--channels debe ser mayor que cero.")
    stream_channels = compute_stream_channels(args.channels, device_info, device_channel)
    if stream_channels > device_info["max_input_channels"]:
        raise ValueError(
            "El dispositivo seleccionado solo admite "
            f"{device_info['max_input_channels']} canal(es) de entrada."
        )

    selected_device_info, capture_samplerate = resolve_working_capture_device(
        selected_device_info=device_info,
        input_devices=input_devices,
        stream_channels=stream_channels,
        requested_samplerate=args.capture_samplerate,
        target_samplerate=runtime_config["sample_rate"],
        decision_step_s=runtime_config["decision_step_s"],
    )
    device_index = selected_device_info["index"]
    device_info = selected_device_info

    model = load_model_for_inference(model_path)
    runtime_config = apply_model_shape_hints(model, runtime_config)
    if args.threshold is not None:
        runtime_config["chunk_threshold"] = float(args.threshold)
        runtime_config = finalize_runtime_config(runtime_config)
    validate_model_against_runtime(model, runtime_config)
    describe_runtime(
        model_path=model_path,
        config_path=config_path,
        device_info=device_info,
        runtime_config=runtime_config,
        capture_samplerate=capture_samplerate,
        stream_channels=stream_channels,
        device_channel=device_channel,
        model=model,
    )

    try:
        run_stream(
            model=model,
            runtime_config=runtime_config,
            device_index=device_index,
            capture_samplerate=capture_samplerate,
            stream_channels=stream_channels,
            device_channel=device_channel,
            max_chunks=args.max_chunks,
        )
    except KeyboardInterrupt:
        print("\nCaptura detenida por el usuario.")


if __name__ == "__main__":
    main()
