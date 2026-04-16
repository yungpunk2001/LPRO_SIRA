from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
METADATA_PATH = DATASET_DIR / "metadata" / "master_index.csv"
MODELS_DIR = SCRIPT_DIR / "modelos"

BUNDLE_PATH = MODELS_DIR / "clasificador_tradicional_bundle.joblib"
METADATA_BUNDLE_PATH = MODELS_DIR / "metadata_clasificador_trad.joblib"
SCALER_PATH = MODELS_DIR / "escalador_sirenas.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder_sirenas.joblib"
LEGACY_MODEL_PATH = MODELS_DIR / "modelo_sirenas_svm.joblib"

DEFAULT_SR = 16000
DEFAULT_CHUNK_SEC = 0.5
DEFAULT_OVERLAP_SEC = 0.125
DEFAULT_N_MFCC = 13
DEFAULT_TARGET_FALSE_ALARMS_PER_MIN = 1.0
FEATURE_VECTOR_SIZE = 50
DEFAULT_EQ_AUGMENTATION_PROB = 0.20
DEFAULT_EQ_ONE_FILTER_PROB = 0.70
DEFAULT_EQ_SHELF_GAIN_DB_MAX = 3.0
DEFAULT_EQ_BELL_GAIN_DB_MAX_SIREN_BAND = 2.0
DEFAULT_EQ_TOTAL_GAIN_DB_LIMIT = 4.0
DEFAULT_EQ_LOW_SHELF_CUTOFF_HZ_RANGE = (150.0, 500.0)
DEFAULT_EQ_BELL_CENTER_HZ_RANGE = (700.0, 2200.0)
DEFAULT_EQ_HIGH_SHELF_CUTOFF_HZ_RANGE = (2500.0, 6000.0)
DEFAULT_EQ_BELL_BANDWIDTH_OCTAVES_RANGE = (1.0, 1.8)
DEFAULT_EQ_SHELF_SHARPNESS_RANGE = (2.0, 3.0)

POSITIVE_LABEL_CANDIDATES = ("siren", "sirena")
BACKGROUND_LABEL_CANDIDATES = ("background", "fondo", "noise", "ruido", "normal")


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def model_name_to_slug(model_name: str) -> str:
    return str(model_name).lower().replace(" ", "_")


def build_model_bundle_path(
    model_name: str,
    output_dir: Path | str | None = None,
) -> Path:
    output_dir = resolve_output_dir(output_dir)
    return output_dir / f"clasificador_tradicional_{model_name_to_slug(model_name)}_bundle.joblib"


def resolve_output_dir(output_dir: Path | str | None = None) -> Path:
    if output_dir is None:
        return ensure_models_dir()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_positive_label(class_names: Sequence[str]) -> str:
    normalized = {str(name).lower(): str(name) for name in class_names}
    for candidate in POSITIVE_LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    if len(class_names) == 2:
        non_background = [
            str(name)
            for name in class_names
            if str(name).lower() not in BACKGROUND_LABEL_CANDIDATES
        ]
        if len(non_background) == 1:
            return non_background[0]

    raise ValueError(
        "No se pudo inferir la clase positiva. Revisa las etiquetas del dataset."
    )


def chunk_signal(
    y_full: np.ndarray,
    sr: int = DEFAULT_SR,
    chunk_length_s: float = DEFAULT_CHUNK_SEC,
    overlap_s: float = DEFAULT_OVERLAP_SEC,
) -> tuple[list[np.ndarray], list[float]]:
    chunk_samples = int(round(chunk_length_s * sr))
    step_samples = int(round((chunk_length_s - overlap_s) * sr))
    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("Los parametros de chunk y solape deben ser positivos.")

    y_full = np.asarray(y_full, dtype=np.float32).reshape(-1)
    if len(y_full) < chunk_samples:
        return [], []

    chunks: list[np.ndarray] = []
    times: list[float] = []
    for start in range(0, len(y_full) - chunk_samples + 1, step_samples):
        chunk = y_full[start : start + chunk_samples]
        if len(chunk) != chunk_samples:
            continue
        chunks.append(chunk)
        times.append(start / sr)
    return chunks, times


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    if len(audio) > target_length:
        return audio[:target_length]
    return audio


def add_shaped_noise(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, len(audio) + 64).astype(np.float32)
    kernel = np.exp(-np.linspace(0.0, 3.5, 65)).astype(np.float32)
    kernel /= np.sum(kernel)
    traffic_like_noise = np.convolve(noise, kernel, mode="valid")[: len(audio)]
    traffic_like_noise /= np.max(np.abs(traffic_like_noise)) + 1e-6

    signal_rms = np.sqrt(np.mean(np.square(audio)) + 1e-8)
    noise_rms = signal_rms / (10.0 ** (rng.uniform(12.0, 24.0) / 20.0))
    return (audio + traffic_like_noise * noise_rms).astype(np.float32)


def add_reverb(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ir_length = int(rng.integers(32, 128))
    impulse = rng.normal(0.0, 1.0, ir_length).astype(np.float32)
    impulse *= np.exp(-np.linspace(0.0, rng.uniform(2.0, 4.5), ir_length)).astype(
        np.float32
    )
    impulse[0] += 1.0

    reverberated = np.convolve(audio, impulse, mode="full")[: len(audio)]
    reverberated = reverberated.astype(np.float32)
    peak = np.max(np.abs(reverberated)) + 1e-6
    reference_peak = max(np.max(np.abs(audio)), 1e-3)
    return ((reverberated / peak) * reference_peak).astype(np.float32)


def apply_compression(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    drive = rng.uniform(1.0, 1.6)
    return np.tanh(drive * audio).astype(np.float32)


def build_low_shelf_curve_db(
    freqs_hz: np.ndarray,
    cutoff_hz: float,
    gain_db: float,
    sharpness: float,
) -> np.ndarray:
    safe_freqs = np.maximum(freqs_hz, 1.0)
    return gain_db / (1.0 + np.power(safe_freqs / max(cutoff_hz, 1.0), sharpness))


def build_high_shelf_curve_db(
    freqs_hz: np.ndarray,
    cutoff_hz: float,
    gain_db: float,
    sharpness: float,
) -> np.ndarray:
    safe_freqs = np.maximum(freqs_hz, 1.0)
    low_component = 1.0 / (
        1.0 + np.power(safe_freqs / max(cutoff_hz, 1.0), sharpness)
    )
    return gain_db * (1.0 - low_component)


def build_bell_curve_db(
    freqs_hz: np.ndarray,
    center_hz: float,
    gain_db: float,
    bandwidth_octaves: float,
) -> np.ndarray:
    safe_freqs = np.maximum(freqs_hz, 1.0)
    log_distance = (
        np.log2(safe_freqs) - np.log2(max(center_hz, 1.0))
    ) / max(bandwidth_octaves, 1e-3)
    return gain_db * np.exp(-0.5 * np.square(log_distance))


def apply_random_spectral_eq(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    eq_one_filter_prob: float = DEFAULT_EQ_ONE_FILTER_PROB,
    eq_shelf_gain_db_max: float = DEFAULT_EQ_SHELF_GAIN_DB_MAX,
    eq_bell_gain_db_max_siren_band: float = DEFAULT_EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
    eq_total_gain_db_limit: float = DEFAULT_EQ_TOTAL_GAIN_DB_LIMIT,
    eq_low_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_LOW_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_center_hz_range: Sequence[float] = DEFAULT_EQ_BELL_CENTER_HZ_RANGE,
    eq_high_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_HIGH_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_bandwidth_octaves_range: Sequence[float] = DEFAULT_EQ_BELL_BANDWIDTH_OCTAVES_RANGE,
    eq_shelf_sharpness_range: Sequence[float] = DEFAULT_EQ_SHELF_SHARPNESS_RANGE,
) -> np.ndarray:
    freqs_hz = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total_curve_db = np.zeros_like(freqs_hz, dtype=np.float32)

    num_filters = 1 if rng.random() < eq_one_filter_prob else 2
    filter_types = rng.choice(
        np.array(["low_shelf", "bell", "high_shelf"], dtype=object),
        size=num_filters,
        replace=False,
    )

    for filter_type in np.atleast_1d(filter_types):
        if filter_type == "low_shelf":
            cutoff_hz = rng.uniform(*eq_low_shelf_cutoff_hz_range)
            gain_db = rng.uniform(-eq_shelf_gain_db_max, eq_shelf_gain_db_max)
            sharpness = rng.uniform(*eq_shelf_sharpness_range)
            curve_db = build_low_shelf_curve_db(
                freqs_hz,
                cutoff_hz,
                gain_db,
                sharpness,
            )
        elif filter_type == "high_shelf":
            cutoff_hz = rng.uniform(*eq_high_shelf_cutoff_hz_range)
            gain_db = rng.uniform(-eq_shelf_gain_db_max, eq_shelf_gain_db_max)
            sharpness = rng.uniform(*eq_shelf_sharpness_range)
            curve_db = build_high_shelf_curve_db(
                freqs_hz,
                cutoff_hz,
                gain_db,
                sharpness,
            )
        else:
            center_hz = rng.uniform(*eq_bell_center_hz_range)
            gain_db = rng.uniform(
                -eq_bell_gain_db_max_siren_band,
                eq_bell_gain_db_max_siren_band,
            )
            bandwidth_octaves = rng.uniform(*eq_bell_bandwidth_octaves_range)
            curve_db = build_bell_curve_db(
                freqs_hz,
                center_hz,
                gain_db,
                bandwidth_octaves,
            )

        total_curve_db += curve_db.astype(np.float32)

    total_curve_db = np.clip(
        total_curve_db,
        -eq_total_gain_db_limit,
        eq_total_gain_db_limit,
    )

    spectrum = np.fft.rfft(audio.astype(np.float32))
    eq_gain = np.power(10.0, total_curve_db / 20.0).astype(np.float32)
    equalized = np.fft.irfft(spectrum * eq_gain, n=len(audio)).astype(np.float32)

    original_rms = np.sqrt(np.mean(np.square(audio)) + 1e-8)
    equalized_rms = np.sqrt(np.mean(np.square(equalized)) + 1e-8)
    equalized *= original_rms / max(equalized_rms, 1e-8)
    return equalized.astype(np.float32)


def compute_effective_eq_apply_probability(
    augmentation_apply_prob: float,
    eq_augmentation_prob: float = DEFAULT_EQ_AUGMENTATION_PROB,
) -> float:
    if augmentation_apply_prob <= 0.0:
        return 0.0
    return min(1.0, eq_augmentation_prob / augmentation_apply_prob)


def augment_audio_chunk(
    audio_chunk: np.ndarray,
    sr: int = DEFAULT_SR,
    rng: np.random.Generator | None = None,
    use_pitch_shift: bool = True,
    use_spectral_eq_augmentation: bool = True,
    spectral_eq_apply_probability: float = DEFAULT_EQ_AUGMENTATION_PROB,
    eq_one_filter_prob: float = DEFAULT_EQ_ONE_FILTER_PROB,
    eq_shelf_gain_db_max: float = DEFAULT_EQ_SHELF_GAIN_DB_MAX,
    eq_bell_gain_db_max_siren_band: float = DEFAULT_EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
    eq_total_gain_db_limit: float = DEFAULT_EQ_TOTAL_GAIN_DB_LIMIT,
    eq_low_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_LOW_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_center_hz_range: Sequence[float] = DEFAULT_EQ_BELL_CENTER_HZ_RANGE,
    eq_high_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_HIGH_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_bandwidth_octaves_range: Sequence[float] = DEFAULT_EQ_BELL_BANDWIDTH_OCTAVES_RANGE,
    eq_shelf_sharpness_range: Sequence[float] = DEFAULT_EQ_SHELF_SHARPNESS_RANGE,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    augmented = np.copy(audio_chunk).astype(np.float32)

    # Orden replicado desde margin_3:
    # 1. Ganancia
    # 2. Time stretch
    # 3. Pitch shift
    # 4. Compresion ligera
    # 5. Reverb
    # 6. Ruido
    # 7. EQ suave
    if rng.random() < 0.8:
        augmented *= rng.uniform(0.75, 1.25)
    if rng.random() < 0.20:
        augmented = librosa.effects.time_stretch(augmented, rate=rng.uniform(0.97, 1.03))
    if use_pitch_shift and rng.random() < 0.20:
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=sr,
            n_steps=rng.uniform(-0.35, 0.35),
        )
    if rng.random() < 0.20:
        augmented = apply_compression(augmented, rng)
    if rng.random() < 0.20:
        augmented = add_reverb(augmented, rng)
    if rng.random() < 0.35:
        augmented = add_shaped_noise(augmented, rng)
    if use_spectral_eq_augmentation and rng.random() < spectral_eq_apply_probability:
        augmented = apply_random_spectral_eq(
            augmented,
            sr=sr,
            rng=rng,
            eq_one_filter_prob=eq_one_filter_prob,
            eq_shelf_gain_db_max=eq_shelf_gain_db_max,
            eq_bell_gain_db_max_siren_band=eq_bell_gain_db_max_siren_band,
            eq_total_gain_db_limit=eq_total_gain_db_limit,
            eq_low_shelf_cutoff_hz_range=eq_low_shelf_cutoff_hz_range,
            eq_bell_center_hz_range=eq_bell_center_hz_range,
            eq_high_shelf_cutoff_hz_range=eq_high_shelf_cutoff_hz_range,
            eq_bell_bandwidth_octaves_range=eq_bell_bandwidth_octaves_range,
            eq_shelf_sharpness_range=eq_shelf_sharpness_range,
        )

    augmented = pad_or_trim(augmented, len(audio_chunk)).astype(np.float32)
    peak = np.max(np.abs(augmented))
    if peak > 1.0:
        augmented /= peak
    return augmented


def count_valid_chunks_in_audio(
    audio_path: Path | str,
    sr: int = DEFAULT_SR,
    chunk_length_s: float = DEFAULT_CHUNK_SEC,
    overlap_s: float = DEFAULT_OVERLAP_SEC,
) -> int:
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        chunk_samples = int(round(chunk_length_s * sr))
        step_samples = int(round((chunk_length_s - overlap_s) * sr))
        if len(y) < chunk_samples or step_samples <= 0:
            return 0
        return len(range(0, len(y) - chunk_samples + 1, step_samples))
    except Exception as exc:
        print(f"Error contando chunks en {audio_path}: {exc}")
        return 0


def add_chunk_count_column(
    df: pd.DataFrame,
    base_path: Path | str = DATASET_DIR,
    sr: int = DEFAULT_SR,
    chunk_length_s: float = DEFAULT_CHUNK_SEC,
    overlap_s: float = DEFAULT_OVERLAP_SEC,
) -> pd.DataFrame:
    df_local = df.copy()
    if "num_chunks" in df_local.columns:
        return df_local

    base_path = Path(base_path)
    df_local["num_chunks"] = df_local["path"].apply(
        lambda relative_path: count_valid_chunks_in_audio(
            base_path / relative_path,
            sr=sr,
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
        )
    )
    return df_local


def compute_class_sample_weights_from_chunks(
    df: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, int]]:
    if "target" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'target'.")
    if "num_chunks" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'num_chunks'.")

    class_chunk_counts = (
        df.groupby("target")["num_chunks"].sum().astype(int).to_dict()
    )
    total_chunks = int(sum(class_chunk_counts.values()))
    num_classes = len(class_chunk_counts)

    if total_chunks == 0 or num_classes == 0:
        raise RuntimeError(
            "No se han podido contar chunks validos para calcular class weights."
        )

    weights = {}
    for class_id, count in sorted(class_chunk_counts.items()):
        if count > 0:
            weights[int(class_id)] = float(total_chunks / (num_classes * count))

    return weights, {int(key): int(value) for key, value in class_chunk_counts.items()}


def build_sample_weight_vector(
    y: np.ndarray,
    class_weights: dict[int, float] | None,
) -> np.ndarray | None:
    if not class_weights:
        return None
    y = np.asarray(y)
    return np.asarray(
        [class_weights.get(int(class_id), 1.0) for class_id in y],
        dtype=np.float32,
    )


def get_path_source(path_value) -> str:
    if pd.isna(path_value):
        return "missing"

    normalized_path = str(path_value).replace("\\", "/")
    path_parts = [part for part in normalized_path.split("/") if part]

    if len(path_parts) >= 3:
        return path_parts[2]
    if len(path_parts) >= 1:
        return path_parts[-1]
    return "missing"


def enrich_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()

    if "source" not in df_local.columns:
        df_local["source"] = df_local["path"].apply(get_path_source)

    if "domain" not in df_local.columns:
        df_local["domain"] = df_local["source"]

    return df_local


def resolve_stratify_columns(
    df: pd.DataFrame,
    requested_columns: Sequence[str],
    fallback_columns: Sequence[str] = ("label",),
) -> tuple[str, ...]:
    available_columns = [column for column in requested_columns if column in df.columns]
    missing_columns = [column for column in requested_columns if column not in df.columns]

    if missing_columns:
        print(
            "Aviso: faltan columnas de estratificacion en metadata: "
            f"{', '.join(missing_columns)}"
        )

    if available_columns:
        return tuple(available_columns)

    fallback_available_columns = [
        column for column in fallback_columns if column in df.columns
    ]
    if fallback_available_columns:
        print(
            "Aviso: se usaran columnas alternativas para estratificar: "
            f"{', '.join(fallback_available_columns)}"
        )
        return tuple(fallback_available_columns)

    raise RuntimeError(
        "No hay columnas validas para estratificar los splits. "
        "Revisa la metadata o la configuracion."
    )


def make_stratum_keys(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series(["global"] * len(df), index=df.index)
    return df.loc[:, columns].fillna("missing").astype(str).agg("|".join, axis=1)


def split_assignment_cost(
    current_counts: dict[str, int],
    current_rows: int,
    target_counts: dict[str, float],
    target_rows: float,
) -> float:
    row_error = ((current_rows - target_rows) / max(1.0, target_rows)) ** 2
    stratum_error = 0.0
    for key, target_value in target_counts.items():
        current_value = current_counts.get(key, 0)
        stratum_error += ((current_value - target_value) / max(1.0, target_value)) ** 2
    return float(row_error + stratum_error)


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    stratify_columns: Sequence[str],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    df_local = df.copy()
    df_local["_stratum_key"] = make_stratum_keys(df_local, list(stratify_columns))

    target_test_rows = len(df_local) * float(test_size)
    target_train_rows = len(df_local) - target_test_rows

    total_stratum_counts = df_local["_stratum_key"].value_counts().to_dict()
    target_test_counts = {
        key: value * float(test_size) for key, value in total_stratum_counts.items()
    }
    target_train_counts = {
        key: value - target_test_counts[key] for key, value in total_stratum_counts.items()
    }

    rng = np.random.default_rng(random_state)
    group_summaries = []
    for group_id, group_df in df_local.groupby(group_col):
        group_summaries.append(
            {
                "group_id": group_id,
                "indices": group_df.index.to_numpy(),
                "size": len(group_df),
                "counts": group_df["_stratum_key"].value_counts().to_dict(),
                "tie_breaker": float(rng.random()),
            }
        )

    group_summaries.sort(key=lambda item: (-item["size"], item["tie_breaker"]))

    train_indices: list[int] = []
    test_indices: list[int] = []
    train_rows = 0
    test_rows = 0
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}

    for summary in group_summaries:
        candidate_train_rows = train_rows + summary["size"]
        candidate_test_rows = test_rows + summary["size"]

        candidate_train_counts = train_counts.copy()
        candidate_test_counts = test_counts.copy()
        for key, value in summary["counts"].items():
            candidate_train_counts[key] = candidate_train_counts.get(key, 0) + value
            candidate_test_counts[key] = candidate_test_counts.get(key, 0) + value

        train_cost = split_assignment_cost(
            candidate_train_counts,
            candidate_train_rows,
            target_train_counts,
            target_train_rows,
        ) + split_assignment_cost(
            test_counts,
            test_rows,
            target_test_counts,
            target_test_rows,
        )
        test_cost = split_assignment_cost(
            train_counts,
            train_rows,
            target_train_counts,
            target_train_rows,
        ) + split_assignment_cost(
            candidate_test_counts,
            candidate_test_rows,
            target_test_counts,
            target_test_rows,
        )

        if test_cost < train_cost:
            test_indices.extend(summary["indices"].tolist())
            test_rows = candidate_test_rows
            test_counts = candidate_test_counts
        else:
            train_indices.extend(summary["indices"].tolist())
            train_rows = candidate_train_rows
            train_counts = candidate_train_counts

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError(
            "El split agrupado/estratificado ha dejado un subconjunto vacio."
        )

    return np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)


def extract_feature_vector(
    y_chunk: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
) -> np.ndarray:
    y_chunk = np.asarray(y_chunk, dtype=np.float32).reshape(-1)
    if y_chunk.size == 0:
        raise ValueError("El fragmento de audio esta vacio.")

    spectrum = np.abs(librosa.stft(y_chunk))
    mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=n_mfcc)
    centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y_chunk, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=spectrum, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_chunk)
    rms = librosa.feature.rms(y=y_chunk)

    features_vector = np.hstack(
        [
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(centroid, axis=1),
            np.std(centroid, axis=1),
            np.mean(bandwidth, axis=1),
            np.std(bandwidth, axis=1),
            np.mean(rolloff, axis=1),
            np.std(rolloff, axis=1),
            np.mean(contrast, axis=1),
            np.std(contrast, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1),
            np.mean(rms, axis=1),
            np.std(rms, axis=1),
        ]
    ).astype(np.float32)

    if features_vector.shape[0] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"Se esperaban {FEATURE_VECTOR_SIZE} features y se obtuvieron "
            f"{features_vector.shape[0]}."
        )
    return features_vector


def extract_features_from_file(
    audio_path: Path | str,
    sr: int = DEFAULT_SR,
    chunk_length_s: float = DEFAULT_CHUNK_SEC,
    overlap_s: float = DEFAULT_OVERLAP_SEC,
    augment: bool = False,
    augmentation_apply_prob: float = 0.5,
    augmentation_extra_copies: int = 1,
    rng: np.random.Generator | None = None,
    use_pitch_shift_augmentation: bool = True,
    use_spectral_eq_augmentation: bool = True,
    spectral_eq_apply_probability: float = DEFAULT_EQ_AUGMENTATION_PROB,
    eq_one_filter_prob: float = DEFAULT_EQ_ONE_FILTER_PROB,
    eq_shelf_gain_db_max: float = DEFAULT_EQ_SHELF_GAIN_DB_MAX,
    eq_bell_gain_db_max_siren_band: float = DEFAULT_EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
    eq_total_gain_db_limit: float = DEFAULT_EQ_TOTAL_GAIN_DB_LIMIT,
    eq_low_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_LOW_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_center_hz_range: Sequence[float] = DEFAULT_EQ_BELL_CENTER_HZ_RANGE,
    eq_high_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_HIGH_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_bandwidth_octaves_range: Sequence[float] = DEFAULT_EQ_BELL_BANDWIDTH_OCTAVES_RANGE,
    eq_shelf_sharpness_range: Sequence[float] = DEFAULT_EQ_SHELF_SHARPNESS_RANGE,
) -> np.ndarray | None:
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        chunks, _ = chunk_signal(
            y,
            sr=sr,
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
        )
        if not chunks:
            return np.empty((0, FEATURE_VECTOR_SIZE), dtype=np.float32)

        rng = rng if rng is not None else np.random.default_rng()
        features_rows: list[np.ndarray] = []
        copies = max(0, int(augmentation_extra_copies))

        for chunk in chunks:
            features_rows.append(extract_feature_vector(chunk, sr=sr))

            if augment and copies > 0:
                for _ in range(copies):
                    if rng.random() < augmentation_apply_prob:
                        augmented_chunk = augment_audio_chunk(
                            chunk,
                            sr=sr,
                            rng=rng,
                            use_pitch_shift=use_pitch_shift_augmentation,
                            use_spectral_eq_augmentation=use_spectral_eq_augmentation,
                            spectral_eq_apply_probability=spectral_eq_apply_probability,
                            eq_one_filter_prob=eq_one_filter_prob,
                            eq_shelf_gain_db_max=eq_shelf_gain_db_max,
                            eq_bell_gain_db_max_siren_band=eq_bell_gain_db_max_siren_band,
                            eq_total_gain_db_limit=eq_total_gain_db_limit,
                            eq_low_shelf_cutoff_hz_range=eq_low_shelf_cutoff_hz_range,
                            eq_bell_center_hz_range=eq_bell_center_hz_range,
                            eq_high_shelf_cutoff_hz_range=eq_high_shelf_cutoff_hz_range,
                            eq_bell_bandwidth_octaves_range=eq_bell_bandwidth_octaves_range,
                            eq_shelf_sharpness_range=eq_shelf_sharpness_range,
                        )
                        features_rows.append(
                            extract_feature_vector(augmented_chunk, sr=sr)
                        )

        return np.vstack(features_rows)
    except Exception as exc:
        print(f"Error procesando {audio_path}: {exc}")
        return None


def build_dataset_in_memory(
    df: pd.DataFrame,
    base_path: Path | str = DATASET_DIR,
    sr: int = DEFAULT_SR,
    chunk_length_s: float = DEFAULT_CHUNK_SEC,
    overlap_s: float = DEFAULT_OVERLAP_SEC,
    augment: bool = False,
    augmentation_apply_prob: float = 0.5,
    augmentation_extra_copies: int = 1,
    random_seed: int | None = None,
    use_pitch_shift_augmentation: bool = True,
    use_spectral_eq_augmentation: bool = True,
    spectral_eq_apply_probability: float = DEFAULT_EQ_AUGMENTATION_PROB,
    eq_one_filter_prob: float = DEFAULT_EQ_ONE_FILTER_PROB,
    eq_shelf_gain_db_max: float = DEFAULT_EQ_SHELF_GAIN_DB_MAX,
    eq_bell_gain_db_max_siren_band: float = DEFAULT_EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
    eq_total_gain_db_limit: float = DEFAULT_EQ_TOTAL_GAIN_DB_LIMIT,
    eq_low_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_LOW_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_center_hz_range: Sequence[float] = DEFAULT_EQ_BELL_CENTER_HZ_RANGE,
    eq_high_shelf_cutoff_hz_range: Sequence[float] = DEFAULT_EQ_HIGH_SHELF_CUTOFF_HZ_RANGE,
    eq_bell_bandwidth_octaves_range: Sequence[float] = DEFAULT_EQ_BELL_BANDWIDTH_OCTAVES_RANGE,
    eq_shelf_sharpness_range: Sequence[float] = DEFAULT_EQ_SHELF_SHARPNESS_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    base_path = Path(base_path)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rng = np.random.default_rng(random_seed) if augment else None

    for row in df.itertuples(index=False):
        audio_path = base_path / row.path
        fragments = extract_features_from_file(
            audio_path,
            sr=sr,
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
            augment=augment,
            augmentation_apply_prob=augmentation_apply_prob,
            augmentation_extra_copies=augmentation_extra_copies,
            rng=rng,
            use_pitch_shift_augmentation=use_pitch_shift_augmentation,
            use_spectral_eq_augmentation=use_spectral_eq_augmentation,
            spectral_eq_apply_probability=spectral_eq_apply_probability,
            eq_one_filter_prob=eq_one_filter_prob,
            eq_shelf_gain_db_max=eq_shelf_gain_db_max,
            eq_bell_gain_db_max_siren_band=eq_bell_gain_db_max_siren_band,
            eq_total_gain_db_limit=eq_total_gain_db_limit,
            eq_low_shelf_cutoff_hz_range=eq_low_shelf_cutoff_hz_range,
            eq_bell_center_hz_range=eq_bell_center_hz_range,
            eq_high_shelf_cutoff_hz_range=eq_high_shelf_cutoff_hz_range,
            eq_bell_bandwidth_octaves_range=eq_bell_bandwidth_octaves_range,
            eq_shelf_sharpness_range=eq_shelf_sharpness_range,
        )
        if fragments is None or len(fragments) == 0:
            continue

        x_list.append(fragments)
        y_list.append(np.full(len(fragments), row.target, dtype=np.int32))

    if not x_list:
        return (
            np.empty((0, FEATURE_VECTOR_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return np.vstack(x_list), np.hstack(y_list)


def feature_names() -> list[str]:
    names: list[str] = []
    for index in range(1, DEFAULT_N_MFCC + 1):
        names.extend([f"mfcc{index}_mean", f"mfcc{index}_std"])

    names.extend(
        [
            "centroid_mean",
            "centroid_std",
            "bandwidth_mean",
            "bandwidth_std",
            "rolloff_mean",
            "rolloff_std",
        ]
    )

    for index in range(1, 8):
        names.extend([f"contrast_b{index}_mean", f"contrast_b{index}_std"])

    names.extend(["zcr_mean", "zcr_std", "rms_mean", "rms_std"])
    return names


def resolve_probability_index(classes: Iterable, positive_target) -> int:
    classes_list = list(classes)
    if positive_target in classes_list:
        return classes_list.index(positive_target)

    normalized_lookup = {
        str(class_value).lower(): index
        for index, class_value in enumerate(classes_list)
    }
    normalized_target = str(positive_target).lower()
    if normalized_target in normalized_lookup:
        return normalized_lookup[normalized_target]

    raise ValueError(
        f"La clase positiva {positive_target!r} no aparece en classes_={classes_list!r}."
    )


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    y_score_positive: np.ndarray,
    positive_class_encoded: int,
    chunk_step_s: float,
) -> dict[str, float | int | list[list[int]]]:
    y_true_binary = (np.asarray(y_true) == positive_class_encoded).astype(np.int32)
    y_pred_binary = np.asarray(y_pred_binary).astype(np.int32)
    y_score_positive = np.asarray(y_score_positive, dtype=np.float32)

    matrix = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    accuracy = (tn + tp) / max(1, tn + fp + fn + tp)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    f2 = fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0)
    auc_pr = (
        average_precision_score(y_true_binary, y_score_positive)
        if len(np.unique(y_true_binary)) > 1
        else float("nan")
    )

    negative_chunks = max(1, int(np.sum(y_true_binary == 0)))
    false_alarms_per_min = (fp / negative_chunks) * (60.0 / chunk_step_s)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "auc_pr": float(auc_pr),
        "accuracy": float(accuracy),
        "false_alarms_per_min": float(false_alarms_per_min),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


def select_best_threshold(
    y_true: np.ndarray,
    positive_probs: np.ndarray,
    positive_class_encoded: int,
    thresholds: np.ndarray | None = None,
    target_false_alarms_per_min: float = DEFAULT_TARGET_FALSE_ALARMS_PER_MIN,
    chunk_step_s: float = DEFAULT_CHUNK_SEC - DEFAULT_OVERLAP_SEC,
) -> tuple[dict[str, float | bool], pd.DataFrame]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)

    threshold_rows = []
    for threshold in thresholds:
        y_pred_binary = (np.asarray(positive_probs) >= threshold).astype(np.int32)
        metrics = compute_detection_metrics(
            y_true,
            y_pred_binary,
            positive_probs,
            positive_class_encoded=positive_class_encoded,
            chunk_step_s=chunk_step_s,
        )
        threshold_rows.append({"threshold": float(threshold), **metrics})

    threshold_df = pd.DataFrame(threshold_rows)
    allowed = threshold_df[
        threshold_df["false_alarms_per_min"] <= target_false_alarms_per_min
    ]

    if not allowed.empty:
        best_row = allowed.sort_values(
            by=["f2", "recall", "precision", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        constraint_satisfied = True
    else:
        best_row = threshold_df.sort_values(
            by=["false_alarms_per_min", "f2", "recall", "precision"],
            ascending=[True, False, False, False],
        ).iloc[0]
        constraint_satisfied = False

    threshold_info = dict(best_row.to_dict())
    threshold_info["constraint_satisfied"] = constraint_satisfied
    threshold_info["target_false_alarms_per_min"] = float(target_false_alarms_per_min)
    return threshold_info, threshold_df


def optimize_binary_threshold(
    y_true: np.ndarray,
    positive_probs: np.ndarray,
    positive_class_encoded: int,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    threshold_info, _ = select_best_threshold(
        y_true,
        positive_probs,
        positive_class_encoded=positive_class_encoded,
        thresholds=thresholds,
    )
    return {
        "threshold": float(threshold_info["threshold"]),
        "f1": float(threshold_info["f1"]),
        "binary_accuracy": float(
            (int(threshold_info["tp"]) + int(threshold_info["tn"]))
            / max(
                1,
                int(threshold_info["tp"])
                + int(threshold_info["tn"])
                + int(threshold_info["fp"])
                + int(threshold_info["fn"]),
            )
        ),
    }


def prepare_loaded_inference_bundle(bundle: dict, bundle_path: Path | str | None = None) -> dict:
    metadata = bundle.get("metadata", {})
    model = bundle["model"]

    positive_class_encoded = metadata.get("positive_class_encoded")
    positive_label = metadata.get("positive_label")
    if positive_class_encoded is not None:
        positive_probability_index = resolve_probability_index(
            model.classes_,
            positive_class_encoded,
        )
    elif positive_label is not None:
        positive_probability_index = resolve_probability_index(
            model.classes_,
            positive_label,
        )
    else:
        raise ValueError(
            "El bundle no incluye informacion suficiente para localizar la clase positiva."
        )

    metadata.setdefault("sample_rate", DEFAULT_SR)
    metadata.setdefault("chunk_seconds", DEFAULT_CHUNK_SEC)
    metadata.setdefault("overlap_seconds", DEFAULT_OVERLAP_SEC)
    if bundle_path is not None:
        metadata.setdefault("bundle_path", str(Path(bundle_path)))

    bundle["metadata"] = metadata
    bundle["positive_probability_index"] = positive_probability_index
    bundle["runtime_threshold"] = float(metadata.get("recommended_threshold", 0.5))
    return bundle


def save_training_artifacts(
    model,
    scaler,
    label_encoder,
    winner_name: str,
    positive_label: str,
    selection_metric: str,
    validation_accuracy: float,
    test_accuracy: float,
    threshold_info: dict[str, float | bool],
    all_models: dict[str, object] | None = None,
    model_scores: dict[str, dict] | None = None,
    sample_rate: int = DEFAULT_SR,
    chunk_seconds: float = DEFAULT_CHUNK_SEC,
    overlap_seconds: float = DEFAULT_OVERLAP_SEC,
    threshold_selection_metric: str = "f2",
    extra_metadata: dict | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    output_dir = resolve_output_dir(output_dir)

    positive_class_encoded = int(label_encoder.transform([positive_label])[0])
    saved_model_paths: dict[str, str] = {}
    saved_bundle_paths: dict[str, str] = {}

    if all_models:
        for model_name, trained_model in all_models.items():
            model_slug = model_name_to_slug(model_name)
            model_path = output_dir / f"modelo_sirenas_{model_slug}.joblib"
            joblib.dump(trained_model, model_path)
            saved_model_paths[model_name] = str(model_path)
            saved_bundle_paths[model_name] = str(
                build_model_bundle_path(model_name, output_dir=output_dir)
            )
    else:
        winner_slug = model_name_to_slug(winner_name)
        winner_model_path = output_dir / f"modelo_sirenas_{winner_slug}.joblib"
        joblib.dump(model, winner_model_path)
        saved_model_paths[winner_name] = str(winner_model_path)
        saved_bundle_paths[winner_name] = str(
            build_model_bundle_path(winner_name, output_dir=output_dir)
        )

    winner_model_path = Path(saved_model_paths[winner_name])
    bundle_path = output_dir / BUNDLE_PATH.name
    metadata_path = output_dir / METADATA_BUNDLE_PATH.name
    scaler_path = output_dir / SCALER_PATH.name
    label_encoder_path = output_dir / LABEL_ENCODER_PATH.name

    def build_bundle_metadata(
        model_name: str,
        model_threshold_info: dict[str, float | bool],
        model_validation_accuracy: float,
        model_test_accuracy: float,
        bundle_role: str,
        bundle_target_path: Path,
    ) -> dict:
        metadata = {
            "winner_name": winner_name,
            "model_name": model_name,
            "bundle_role": bundle_role,
            "selection_metric": selection_metric,
            "validation_accuracy": float(model_validation_accuracy),
            "test_accuracy": float(model_test_accuracy),
            "positive_label": positive_label,
            "positive_class_encoded": positive_class_encoded,
            "class_names": [str(name) for name in label_encoder.classes_],
            "recommended_threshold": float(model_threshold_info["threshold"]),
            "threshold_f1": float(model_threshold_info.get("f1", 0.0)),
            "threshold_f2": float(model_threshold_info.get("f2", 0.0)),
            "threshold_precision": float(model_threshold_info.get("precision", 0.0)),
            "threshold_recall": float(model_threshold_info.get("recall", 0.0)),
            "threshold_auc_pr": float(model_threshold_info.get("auc_pr", float("nan"))),
            "threshold_false_alarms_per_min": float(
                model_threshold_info.get("false_alarms_per_min", 0.0)
            ),
            "threshold_constraint_satisfied": bool(
                model_threshold_info.get("constraint_satisfied", False)
            ),
            "target_false_alarms_per_min": float(
                model_threshold_info.get(
                    "target_false_alarms_per_min",
                    DEFAULT_TARGET_FALSE_ALARMS_PER_MIN,
                )
            ),
            "sample_rate": int(sample_rate),
            "chunk_seconds": float(chunk_seconds),
            "overlap_seconds": float(overlap_seconds),
            "decision_step_seconds": float(chunk_seconds - overlap_seconds),
            "feature_names": feature_names(),
            "saved_models": saved_model_paths,
            "saved_bundles": saved_bundle_paths,
            "default_bundle_path": str(bundle_path),
            "threshold_selection_metric": threshold_selection_metric,
            "model_scores": model_scores or {},
            "output_dir": str(output_dir),
            "bundle_path": str(bundle_target_path),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    if all_models:
        for model_name, trained_model in all_models.items():
            model_result = (model_scores or {}).get(model_name, {})
            model_threshold_info = model_result.get("final_threshold_info", threshold_info)
            validation_metrics_refit = model_result.get("validation_metrics_refit", {})
            test_metrics = model_result.get("test_metrics", {})
            model_validation_accuracy = float(
                validation_metrics_refit.get(
                    "accuracy",
                    validation_accuracy if model_name == winner_name else 0.0,
                )
            )
            model_test_accuracy = float(
                test_metrics.get(
                    "accuracy",
                    test_accuracy if model_name == winner_name else 0.0,
                )
            )
            model_bundle_path = Path(saved_bundle_paths[model_name])
            model_bundle = {
                "model": trained_model,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "metadata": build_bundle_metadata(
                    model_name=model_name,
                    model_threshold_info=model_threshold_info,
                    model_validation_accuracy=model_validation_accuracy,
                    model_test_accuracy=model_test_accuracy,
                    bundle_role="model_specific",
                    bundle_target_path=model_bundle_path,
                ),
            }
            joblib.dump(model_bundle, model_bundle_path)
    else:
        model_bundle_path = Path(saved_bundle_paths[winner_name])
        winner_only_bundle = {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "metadata": build_bundle_metadata(
                model_name=winner_name,
                model_threshold_info=threshold_info,
                model_validation_accuracy=validation_accuracy,
                model_test_accuracy=test_accuracy,
                bundle_role="model_specific",
                bundle_target_path=model_bundle_path,
            ),
        }
        joblib.dump(winner_only_bundle, model_bundle_path)

    winner_bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "metadata": build_bundle_metadata(
            model_name=winner_name,
            model_threshold_info=threshold_info,
            model_validation_accuracy=validation_accuracy,
            model_test_accuracy=test_accuracy,
            bundle_role="winner_default",
            bundle_target_path=bundle_path,
        ),
    }

    joblib.dump(winner_bundle, bundle_path)
    metadata = winner_bundle["metadata"]
    joblib.dump(metadata, metadata_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)

    return {
        "bundle_path": bundle_path,
        "model_bundle_paths": {name: Path(path) for name, path in saved_bundle_paths.items()},
        "metadata_path": metadata_path,
        "winner_model_path": winner_model_path,
        "scaler_path": scaler_path,
        "label_encoder_path": label_encoder_path,
    }


def list_available_inference_bundles(
    models_dir: Path | str = MODELS_DIR,
) -> list[dict]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    bundle_paths = sorted(
        models_dir.glob("clasificador_tradicional_*_bundle.joblib")
    )
    bundles: list[dict] = []

    for bundle_path in bundle_paths:
        try:
            bundle = prepare_loaded_inference_bundle(
                joblib.load(bundle_path),
                bundle_path=bundle_path,
            )
        except Exception as exc:
            print(f"Aviso: no se pudo cargar el bundle {bundle_path.name}: {exc}")
            continue

        metadata = bundle.get("metadata", {})
        model_name = str(
            metadata.get("model_name")
            or metadata.get("winner_name")
            or bundle_path.stem
        )
        bundles.append(
            {
                "model_name": model_name,
                "bundle_path": bundle_path,
                "bundle": bundle,
                "recommended_threshold": float(bundle["runtime_threshold"]),
                "winner_name": str(metadata.get("winner_name", model_name)),
                "bundle_role": str(metadata.get("bundle_role", "model_specific")),
            }
        )

    if bundles:
        return bundles

    if BUNDLE_PATH.exists():
        try:
            bundle = prepare_loaded_inference_bundle(joblib.load(BUNDLE_PATH), bundle_path=BUNDLE_PATH)
            metadata = bundle.get("metadata", {})
            model_name = str(metadata.get("model_name") or metadata.get("winner_name") or "Modelo por defecto")
            return [
                {
                    "model_name": model_name,
                    "bundle_path": BUNDLE_PATH,
                    "bundle": bundle,
                    "recommended_threshold": float(bundle["runtime_threshold"]),
                    "winner_name": str(metadata.get("winner_name", model_name)),
                    "bundle_role": str(metadata.get("bundle_role", "winner_default")),
                }
            ]
        except Exception as exc:
            print(f"Aviso: no se pudo cargar el bundle por defecto: {exc}")

    if LEGACY_MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(LEGACY_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        positive_probability_index = resolve_probability_index(model.classes_, 1)
        return [
            {
                "model_name": "SVM legado",
                "bundle_path": LEGACY_MODEL_PATH,
                "bundle": {
                    "model": model,
                    "scaler": scaler,
                    "label_encoder": None,
                    "metadata": {
                        "winner_name": "SVM legado",
                        "model_name": "SVM legado",
                        "positive_label": "siren",
                        "positive_class_encoded": 1,
                        "recommended_threshold": 0.75,
                        "sample_rate": DEFAULT_SR,
                        "chunk_seconds": DEFAULT_CHUNK_SEC,
                        "overlap_seconds": DEFAULT_OVERLAP_SEC,
                        "bundle_path": str(LEGACY_MODEL_PATH),
                    },
                    "positive_probability_index": positive_probability_index,
                    "runtime_threshold": 0.75,
                },
                "recommended_threshold": 0.75,
                "winner_name": "SVM legado",
                "bundle_role": "legacy",
            }
        ]

    return []


def prompt_user_to_select_inference_bundle(
    models_dir: Path | str = MODELS_DIR,
) -> dict:
    available_bundles = list_available_inference_bundles(models_dir=models_dir)
    if not available_bundles:
        raise FileNotFoundError(
            "No se encontro ningun bundle de inferencia en la carpeta de modelos."
        )

    print("\nModelos disponibles para inferencia:")
    for index, bundle_info in enumerate(available_bundles, start=1):
        winner_suffix = ""
        if bundle_info["model_name"] == bundle_info["winner_name"]:
            winner_suffix = " [ganador]"
        print(
            f"{index}. {bundle_info['model_name']}{winner_suffix} | "
            f"umbral={bundle_info['recommended_threshold']:.2f} | "
            f"archivo={Path(bundle_info['bundle_path']).name}"
        )

    while True:
        user_input = input(
            f"Selecciona un modelo para cargar [1-{len(available_bundles)}]: "
        ).strip()
        if user_input.isdigit():
            selected_index = int(user_input)
            if 1 <= selected_index <= len(available_bundles):
                return available_bundles[selected_index - 1]["bundle"]
        print("Seleccion no valida. Introduce un numero de la lista.")


def load_inference_bundle(bundle_path: Path | str | None = None) -> dict:
    if bundle_path is not None:
        return prepare_loaded_inference_bundle(
            joblib.load(bundle_path),
            bundle_path=bundle_path,
        )

    available_bundles = list_available_inference_bundles()
    if available_bundles:
        return available_bundles[0]["bundle"]

    raise FileNotFoundError(
        "No se encontro ningun bundle de inferencia en la carpeta de modelos."
    )


def predict_positive_probability(bundle: dict, x_scaled: np.ndarray) -> float:
    model = bundle["model"]
    probability_index = bundle["positive_probability_index"]
    probabilities = model.predict_proba(x_scaled)
    return float(probabilities[0, probability_index])
