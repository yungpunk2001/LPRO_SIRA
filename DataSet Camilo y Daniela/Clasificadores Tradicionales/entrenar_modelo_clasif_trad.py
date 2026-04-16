import inspect
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from clasif_trad_utils import (
    DATASET_DIR,
    METADATA_PATH,
    MODELS_DIR,
    add_chunk_count_column,
    build_model_bundle_path,
    build_dataset_in_memory,
    build_sample_weight_vector,
    compute_class_sample_weights_from_chunks,
    compute_detection_metrics,
    compute_effective_eq_apply_probability,
    enrich_metadata_columns,
    feature_names,
    grouped_stratified_split,
    resolve_positive_label,
    resolve_probability_index,
    resolve_stratify_columns,
    save_training_artifacts,
    select_best_threshold,
)

TRAINING_CONFIG_ENV_VAR = "SIREN_TRAD_TRAINING_CONFIG_PATH"


def load_runtime_config_overrides():
    config_path = os.environ.get(TRAINING_CONFIG_ENV_VAR)
    if not config_path:
        return {}, None

    with open(config_path, "r", encoding="utf-8") as file_handle:
        overrides = json.load(file_handle)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"El fichero indicado en {TRAINING_CONFIG_ENV_VAR} debe contener un objeto JSON."
        )

    return overrides, config_path


RUNTIME_CONFIG_OVERRIDES, RUNTIME_CONFIG_PATH = load_runtime_config_overrides()


def get_config_value(name, default):
    return RUNTIME_CONFIG_OVERRIDES.get(name, default)


def get_float_range_value(name, default):
    raw_value = get_config_value(name, list(default))
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{name} debe ser una lista o tupla de dos valores.")
    return (float(raw_value[0]), float(raw_value[1]))


def build_default_split_manifest_basename() -> str:
    """Construye un basename reproducible y propio para el split tradicional."""
    fingerprint_payload = {
        "sample_rate": int(get_config_value("SAMPLE_RATE", 16000)),
        "chunk_length_s": float(get_config_value("CHUNK_LENGTH_S", 0.5)),
        "use_overlap": bool(get_config_value("USE_OVERLAP", True)),
        "overlap_s": float(get_config_value("OVERLAP_S", 0.125)),
        "random_seed": int(get_config_value("RANDOM_SEED", 42)),
        "split_train_fraction": float(get_config_value("SPLIT_TRAIN_FRACTION", 0.70)),
        "split_validation_fraction": float(
            get_config_value("SPLIT_VALIDATION_FRACTION", 0.15)
        ),
        "split_test_fraction": float(get_config_value("SPLIT_TEST_FRACTION", 0.15)),
        "split_stratify_columns": list(
            get_config_value("SPLIT_STRATIFY_COLUMNS", ["label", "domain"])
        ),
        "split_weight_column": str(get_config_value("SPLIT_WEIGHT_COLUMN", "num_chunks")),
        "split_row_cost_weight": float(get_config_value("SPLIT_ROW_COST_WEIGHT", 0.15)),
        "apply_train_background_subsampling": bool(
            get_config_value("APPLY_TRAIN_BACKGROUND_SUBSAMPLING", True)
        ),
        "train_background_to_siren_chunk_ratio": float(
            get_config_value("TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO", 1.0)
        ),
        "train_background_min_groups_per_bucket": int(
            get_config_value("TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET", 1)
        ),
        "train_background_default_bucket_weight": float(
            get_config_value("TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT", 1.0)
        ),
        "train_background_hard_negative_weight": float(
            get_config_value("TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT", 2.0)
        ),
        "train_background_reduced_bucket_weight": float(
            get_config_value("TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT", 0.35)
        ),
        "train_background_hard_negative_buckets": list(
            get_config_value(
                "TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS",
                [
                    "UrbanSound8K_Clasificado/car_horn",
                    "UrbanSound8K_Clasificado/children_playing",
                    "UrbanSound8K_Clasificado/drilling",
                    "UrbanSound8K_Clasificado/jackhammer",
                    "UrbanSound8K_Clasificado/street_music",
                ],
            )
        ),
        "train_background_reduced_buckets": list(
            get_config_value(
                "TRAIN_BACKGROUND_REDUCED_BUCKETS",
                [
                    "UrbanSound8K_Clasificado/air_conditioner",
                    "UrbanSound8K_Clasificado/dog_bark",
                    "UrbanSound8K_Clasificado/gun_shot",
                ],
            )
        ),
        "manifest_version": int(get_config_value("SPLIT_MANIFEST_VERSION", 3)),
    }
    payload_bytes = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload_bytes).hexdigest()[:12]
    return f"split_manifest_clasif_trad_v3_{digest}"


RANDOM_SEED = int(get_config_value("RANDOM_SEED", 42))
SAMPLE_RATE = int(get_config_value("SAMPLE_RATE", 16000))
CHUNK_LENGTH_S = float(get_config_value("CHUNK_LENGTH_S", 0.5))
USE_OVERLAP = bool(get_config_value("USE_OVERLAP", True))
CONFIGURED_OVERLAP_S = float(get_config_value("OVERLAP_S", 0.125))
OVERLAP_S = CONFIGURED_OVERLAP_S if USE_OVERLAP else 0.0
CHUNK_STEP_S = CHUNK_LENGTH_S - OVERLAP_S
if OVERLAP_S < 0:
    raise ValueError("OVERLAP_S no puede ser negativo.")
if CHUNK_STEP_S <= 0:
    raise ValueError("OVERLAP_S debe ser menor que CHUNK_LENGTH_S.")

RF_N_ESTIMATORS = int(get_config_value("RF_N_ESTIMATORS", 200))
SVM_C = float(get_config_value("SVM_C", 1.0))
SVM_GAMMA = get_config_value("SVM_GAMMA", "scale")
KNN_NEIGHBORS = int(get_config_value("KNN_NEIGHBORS", 5))

TARGET_FALSE_ALARMS_PER_MIN = float(
    get_config_value("TARGET_FALSE_ALARMS_PER_MIN", 1.0)
)
THRESHOLD_GRID = np.array(
    get_config_value("THRESHOLD_GRID", np.linspace(0.10, 0.95, 18).tolist()),
    dtype=np.float32,
)
USE_CLASS_WEIGHTS = bool(get_config_value("USE_CLASS_WEIGHTS", True))
USE_DATA_AUGMENTATION = bool(get_config_value("USE_DATA_AUGMENTATION", True))
USE_PITCH_SHIFT_AUGMENTATION = bool(
    get_config_value("USE_PITCH_SHIFT_AUGMENTATION", True)
)
AUGMENTATION_APPLY_PROB = float(get_config_value("AUGMENTATION_APPLY_PROB", 0.5))
AUGMENTATION_EXTRA_COPIES = int(get_config_value("AUGMENTATION_EXTRA_COPIES", 1))
USE_SPECTRAL_EQ_AUGMENTATION = bool(
    get_config_value("USE_SPECTRAL_EQ_AUGMENTATION", True)
)
EQ_AUGMENTATION_PROB = float(get_config_value("EQ_AUGMENTATION_PROB", 0.20))
EQ_ONE_FILTER_PROB = float(get_config_value("EQ_ONE_FILTER_PROB", 0.70))
EQ_SHELF_GAIN_DB_MAX = float(get_config_value("EQ_SHELF_GAIN_DB_MAX", 3.0))
EQ_BELL_GAIN_DB_MAX_SIREN_BAND = float(
    get_config_value("EQ_BELL_GAIN_DB_MAX_SIREN_BAND", 2.0)
)
EQ_TOTAL_GAIN_DB_LIMIT = float(get_config_value("EQ_TOTAL_GAIN_DB_LIMIT", 4.0))
EQ_LOW_SHELF_CUTOFF_HZ_RANGE = get_float_range_value(
    "EQ_LOW_SHELF_CUTOFF_HZ_RANGE",
    (150.0, 500.0),
)
EQ_BELL_CENTER_HZ_RANGE = get_float_range_value(
    "EQ_BELL_CENTER_HZ_RANGE",
    (700.0, 2200.0),
)
EQ_HIGH_SHELF_CUTOFF_HZ_RANGE = get_float_range_value(
    "EQ_HIGH_SHELF_CUTOFF_HZ_RANGE",
    (2500.0, 6000.0),
)
EQ_BELL_BANDWIDTH_OCTAVES_RANGE = get_float_range_value(
    "EQ_BELL_BANDWIDTH_OCTAVES_RANGE",
    (1.0, 1.8),
)
EQ_SHELF_SHARPNESS_RANGE = get_float_range_value(
    "EQ_SHELF_SHARPNESS_RANGE",
    (2.0, 3.0),
)
AUGMENTATION_RANDOM_SEED = int(
    get_config_value("AUGMENTATION_RANDOM_SEED", RANDOM_SEED + 1000)
)
SPLIT_STRATIFY_COLUMNS = tuple(
    get_config_value("SPLIT_STRATIFY_COLUMNS", ["label", "domain"])
)
SPLIT_MANIFEST_VERSION = int(get_config_value("SPLIT_MANIFEST_VERSION", 3))
SPLIT_TRAIN_FRACTION = float(get_config_value("SPLIT_TRAIN_FRACTION", 0.70))
SPLIT_VALIDATION_FRACTION = float(get_config_value("SPLIT_VALIDATION_FRACTION", 0.15))
SPLIT_TEST_FRACTION = float(get_config_value("SPLIT_TEST_FRACTION", 0.15))
SPLIT_WEIGHT_COLUMN = str(get_config_value("SPLIT_WEIGHT_COLUMN", "num_chunks"))
SPLIT_ROW_COST_WEIGHT = float(get_config_value("SPLIT_ROW_COST_WEIGHT", 0.15))
REUSE_SPLIT_MANIFEST = bool(get_config_value("REUSE_SPLIT_MANIFEST", True))
SAVE_SPLIT_MANIFEST = bool(get_config_value("SAVE_SPLIT_MANIFEST", True))
SPLIT_MANIFEST_BASENAME = str(
    get_config_value(
        "SPLIT_MANIFEST_BASENAME",
        build_default_split_manifest_basename(),
    )
)
APPLY_TRAIN_BACKGROUND_SUBSAMPLING = bool(
    get_config_value("APPLY_TRAIN_BACKGROUND_SUBSAMPLING", True)
)
TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO = float(
    get_config_value("TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO", 1.0)
)
TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET = int(
    get_config_value("TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET", 1)
)
TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT", 1.0)
)
TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT", 2.0)
)
TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT = float(
    get_config_value("TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT", 0.35)
)
TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS = tuple(
    get_config_value(
        "TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS",
        [
            "UrbanSound8K_Clasificado/car_horn",
            "UrbanSound8K_Clasificado/children_playing",
            "UrbanSound8K_Clasificado/drilling",
            "UrbanSound8K_Clasificado/jackhammer",
            "UrbanSound8K_Clasificado/street_music",
        ],
    )
)
TRAIN_BACKGROUND_REDUCED_BUCKETS = tuple(
    get_config_value(
        "TRAIN_BACKGROUND_REDUCED_BUCKETS",
        [
            "UrbanSound8K_Clasificado/air_conditioner",
            "UrbanSound8K_Clasificado/dog_bark",
            "UrbanSound8K_Clasificado/gun_shot",
        ],
    )
)
SAVE_EXPERIMENT_REPORT = bool(get_config_value("SAVE_EXPERIMENT_REPORT", True))
SAVE_POSTPROCESSING_CONFIG = bool(get_config_value("SAVE_POSTPROCESSING_CONFIG", True))
SHOW_RF_PLOT = bool(get_config_value("SHOW_RF_PLOT", True))
if not 0.0 <= AUGMENTATION_APPLY_PROB <= 1.0:
    raise ValueError("AUGMENTATION_APPLY_PROB debe estar entre 0 y 1.")
if AUGMENTATION_EXTRA_COPIES < 0:
    raise ValueError("AUGMENTATION_EXTRA_COPIES no puede ser negativo.")
if not 0.0 <= EQ_AUGMENTATION_PROB <= 1.0:
    raise ValueError("EQ_AUGMENTATION_PROB debe estar entre 0 y 1.")
if not 0.0 <= EQ_ONE_FILTER_PROB <= 1.0:
    raise ValueError("EQ_ONE_FILTER_PROB debe estar entre 0 y 1.")
if not np.isclose(
    SPLIT_TRAIN_FRACTION + SPLIT_VALIDATION_FRACTION + SPLIT_TEST_FRACTION,
    1.0,
    atol=1e-6,
):
    raise ValueError("Las fracciones de train/validation/test deben sumar 1.0.")
if TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO <= 0.0:
    raise ValueError(
        "TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO debe ser positivo."
    )
if TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET < 0:
    raise ValueError(
        "TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET no puede ser negativo."
    )

EFFECTIVE_EQ_APPLY_PROB = compute_effective_eq_apply_probability(
    AUGMENTATION_APPLY_PROB,
    EQ_AUGMENTATION_PROB,
)

SELECTION_METRIC = "validation_f2_with_false_alarm_constraint"
THRESHOLD_SELECTION_METRIC = "f2_under_false_alarms_per_min"

RUN_NAME_PREFIX = str(get_config_value("RUN_NAME_PREFIX", "clasif_trad_run"))
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_BASENAME = f"{RUN_NAME_PREFIX}_{RUN_TIMESTAMP}"
RUN_OUTPUT_DIR = Path(
    str(get_config_value("RUN_OUTPUT_DIR", MODELS_DIR / RUN_BASENAME))
)
EXPERIMENT_REPORT_PATH = RUN_OUTPUT_DIR / f"{RUN_BASENAME}_reporte.json"
POSTPROCESSING_PATH = RUN_OUTPUT_DIR / f"{RUN_BASENAME}_postprocesado.json"
SPLIT_MANIFEST_PATH = DATASET_DIR / "metadata" / f"{SPLIT_MANIFEST_BASENAME}.csv"
SPLIT_MANIFEST_INFO_PATH = (
    DATASET_DIR / "metadata" / f"{SPLIT_MANIFEST_BASENAME}_info.json"
)


def validate_required_paths() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "No se encontro el indice maestro del dataset en "
            f"{METADATA_PATH}."
        )
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"No se encontro la carpeta dataset en {DATASET_DIR}.")


def candidate_models() -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_SEED,
            C=SVM_C,
            gamma=SVM_GAMMA,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            n_jobs=-1,
        ),
    }


def print_metrics_block(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(
        "Precision: {precision:.4f} | Recall: {recall:.4f} | "
        "F1: {f1:.4f} | F2: {f2:.4f} | AUC-PR: {auc_pr:.4f} | "
        "Accuracy: {accuracy:.4f} | Falsas alarmas/min: {false_alarms_per_min:.2f}".format(
            **metrics
        )
    )
    print(
        "Matriz de confusion [[TN, FP], [FN, TP]] = "
        f"{metrics['confusion_matrix']}"
    )


def build_metrics_report_block(title: str, metrics: dict) -> str:
    lines = [
        title,
        (
            "Precision: {precision:.4f} | Recall: {recall:.4f} | "
            "F1: {f1:.4f} | F2: {f2:.4f} | AUC-PR: {auc_pr:.4f} | "
            "Accuracy: {accuracy:.4f} | Falsas alarmas/min: {false_alarms_per_min:.2f}"
        ).format(**metrics),
        "Matriz de confusion [[TN, FP], [FN, TP]]:",
        str(np.array(metrics["confusion_matrix"])),
    ]
    return "\n".join(lines)


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(key): make_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def compute_class_weights_from_targets(
    y: np.ndarray,
) -> tuple[dict[int, float], dict[int, int]]:
    classes, counts = np.unique(np.asarray(y, dtype=np.int32), return_counts=True)
    total_chunks = int(np.sum(counts))
    num_classes = len(classes)

    if total_chunks == 0 or num_classes == 0:
        raise RuntimeError("No hay chunks validos para calcular class weights.")

    class_chunk_counts = {
        int(class_id): int(count)
        for class_id, count in zip(classes.tolist(), counts.tolist())
    }
    class_weights = {
        int(class_id): float(total_chunks / (num_classes * count))
        for class_id, count in class_chunk_counts.items()
        if count > 0
    }
    return class_weights, class_chunk_counts


def fit_model_with_optional_weights(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[object, bool]:
    fit_kwargs = {}
    used_sample_weight = False

    if sample_weight is not None:
        try:
            fit_signature = inspect.signature(model.fit)
            if "sample_weight" in fit_signature.parameters:
                fit_kwargs["sample_weight"] = sample_weight
                used_sample_weight = True
        except (TypeError, ValueError):
            pass

    model.fit(x_train, y_train, **fit_kwargs)
    return model, used_sample_weight


def build_model_selection_key(result: dict) -> tuple:
    metrics = result["validation_metrics"]
    threshold_info = result["threshold_info"]
    constraint_satisfied = bool(threshold_info["constraint_satisfied"])

    if constraint_satisfied:
        return (
            1,
            float(metrics["f2"]),
            float(metrics["recall"]),
            float(metrics["precision"]),
            float(metrics["auc_pr"]),
            -float(metrics["false_alarms_per_min"]),
        )

    return (
        0,
        -float(metrics["false_alarms_per_min"]),
        float(metrics["f2"]),
        float(metrics["recall"]),
        float(metrics["precision"]),
        float(metrics["auc_pr"]),
    )


def get_background_bucket_weight(bucket_name: str | None) -> float:
    bucket = "" if bucket_name is None else str(bucket_name)
    if bucket in TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS:
        return TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT
    if bucket in TRAIN_BACKGROUND_REDUCED_BUCKETS:
        return TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT
    return TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT


def select_training_background_subset(
    train_df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
    target_bg_to_siren_ratio: float = TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
) -> pd.DataFrame:
    """
    Selecciona un subconjunto reproducible de backgrounds solo para train.

    Reglas:
    - Siren se conserva completo.
    - Validation/test no se tocan nunca.
    - Background se submuestrea por grupos seguros y no por archivos sueltos.
    - La prioridad de muestreo se define por buckets reproducibles y no por
      orden fijo de fuentes.
    - Se garantiza una representacion minima por bucket cuando el presupuesto
      global de chunks lo permite.
    """
    curated_df = train_df.copy()
    label_series = curated_df["label"].astype(str).str.strip().str.lower()
    bg_mask = label_series == "background"
    siren_mask = label_series.isin({"siren", "sirena"})

    curated_df["train_keep"] = siren_mask
    curated_df["background_sampling_weight"] = np.nan
    if bg_mask.any():
        curated_df.loc[bg_mask, "background_sampling_weight"] = curated_df.loc[
            bg_mask, "background_sampling_bucket"
        ].apply(get_background_bucket_weight)

    if not APPLY_TRAIN_BACKGROUND_SUBSAMPLING:
        curated_df.loc[bg_mask, "train_keep"] = True
        return curated_df

    siren_chunks = int(curated_df.loc[siren_mask, "num_chunks"].sum())
    total_background_chunks = int(curated_df.loc[bg_mask, "num_chunks"].sum())
    target_background_chunks = int(
        np.ceil(max(1.0, siren_chunks * target_bg_to_siren_ratio))
    )

    if (
        total_background_chunks <= target_background_chunks
        or not bg_mask.any()
        or siren_chunks <= 0
    ):
        curated_df.loc[bg_mask, "train_keep"] = True
        return curated_df

    valid_background_df = curated_df.loc[bg_mask & (curated_df["num_chunks"] > 0)].copy()
    if valid_background_df.empty:
        curated_df.loc[bg_mask, "train_keep"] = False
        return curated_df

    rng = np.random.default_rng(random_state)
    group_records = []

    for safe_group_id, group_df in valid_background_df.groupby("safe_group_id"):
        bucket_series = group_df["background_sampling_bucket"].dropna().astype(str)
        bucket_value = (
            bucket_series.mode().iloc[0]
            if not bucket_series.empty
            else str(group_df["source"].iloc[0])
        )
        group_records.append(
            {
                "safe_group_id": safe_group_id,
                "bucket": bucket_value,
                "num_chunks": int(group_df["num_chunks"].sum()),
                "bucket_weight": float(get_background_bucket_weight(bucket_value)),
                "random_key": float(rng.random()),
            }
        )

    groups_df = pd.DataFrame(group_records)
    if groups_df.empty:
        curated_df.loc[bg_mask, "train_keep"] = False
        return curated_df

    bucket_available_chunks = groups_df.groupby("bucket")["num_chunks"].sum().astype(float)
    bucket_weights = groups_df.groupby("bucket")["bucket_weight"].first().astype(float)
    weighted_available_chunks = bucket_available_chunks * bucket_weights
    bucket_target_chunks = (
        target_background_chunks * weighted_available_chunks / weighted_available_chunks.sum()
    )

    selected_group_ids: set[str] = set()
    selected_chunks = 0
    selected_chunks_by_bucket = {bucket_name: 0 for bucket_name in bucket_target_chunks.index}

    ordered_buckets = sorted(
        bucket_target_chunks.index.tolist(),
        key=lambda bucket_name: (
            -float(bucket_target_chunks[bucket_name]),
            -float(bucket_weights[bucket_name]),
            bucket_name,
        ),
    )

    def register_group_selection(group_row) -> None:
        nonlocal selected_chunks
        selected_group_ids.add(group_row.safe_group_id)
        selected_chunks += int(group_row.num_chunks)
        selected_chunks_by_bucket[group_row.bucket] = (
            selected_chunks_by_bucket.get(group_row.bucket, 0) + int(group_row.num_chunks)
        )

    minimum_bucket_plan = []
    if TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET > 0:
        for bucket_name in ordered_buckets:
            bucket_groups = groups_df.loc[groups_df["bucket"] == bucket_name].sort_values(
                by=["num_chunks", "random_key"],
                ascending=[True, True],
            )
            minimum_group_count = min(
                TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
                len(bucket_groups),
            )
            if minimum_group_count <= 0 or float(bucket_target_chunks[bucket_name]) <= 0.0:
                continue

            mandatory_groups = bucket_groups.head(minimum_group_count).copy()
            minimum_bucket_plan.append(
                {
                    "bucket": bucket_name,
                    "mandatory_chunks": int(mandatory_groups["num_chunks"].sum()),
                    "mandatory_groups": mandatory_groups,
                    "bucket_target": float(bucket_target_chunks[bucket_name]),
                    "bucket_weight": float(bucket_weights[bucket_name]),
                }
            )

        minimum_bucket_plan.sort(
            key=lambda item: (
                -item["bucket_target"],
                -item["bucket_weight"],
                item["mandatory_chunks"],
                item["bucket"],
            )
        )

        for plan_item in minimum_bucket_plan:
            if (
                selected_chunks + plan_item["mandatory_chunks"] > target_background_chunks
                and selected_group_ids
            ):
                continue

            for group_row in plan_item["mandatory_groups"].itertuples(index=False):
                if group_row.safe_group_id in selected_group_ids:
                    continue
                register_group_selection(group_row)

    for bucket_name in ordered_buckets:
        if selected_chunks >= target_background_chunks and selected_group_ids:
            break

        bucket_target = float(bucket_target_chunks[bucket_name])
        bucket_selected_chunks = selected_chunks_by_bucket.get(bucket_name, 0)
        bucket_groups = groups_df.loc[groups_df["bucket"] == bucket_name].sort_values(
            by=["random_key", "num_chunks"],
            ascending=[True, True],
        )

        for group_row in bucket_groups.itertuples(index=False):
            if group_row.safe_group_id in selected_group_ids:
                continue
            if (
                bucket_selected_chunks >= bucket_target and bucket_selected_chunks > 0
            ) or (
                selected_chunks >= target_background_chunks and selected_group_ids
            ):
                break

            register_group_selection(group_row)
            bucket_selected_chunks = selected_chunks_by_bucket.get(bucket_name, 0)

    if selected_chunks < target_background_chunks:
        remaining_groups = groups_df.loc[
            ~groups_df["safe_group_id"].isin(selected_group_ids)
        ].sort_values(
            by=["bucket_weight", "random_key", "num_chunks"],
            ascending=[False, True, True],
        )

        for group_row in remaining_groups.itertuples(index=False):
            selected_group_ids.add(group_row.safe_group_id)
            selected_chunks += int(group_row.num_chunks)
            if selected_chunks >= target_background_chunks:
                break

    curated_df.loc[bg_mask, "train_keep"] = curated_df.loc[bg_mask, "safe_group_id"].isin(
        selected_group_ids
    )

    return curated_df


def build_split_manifest_settings(stratify_columns: tuple[str, ...]) -> dict:
    """Resume la configuracion que hace valido un manifiesto persistido."""
    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "sample_rate": SAMPLE_RATE,
        "chunk_length_s": CHUNK_LENGTH_S,
        "overlap_s": OVERLAP_S,
        "random_seed": RANDOM_SEED,
        "split_train_fraction": SPLIT_TRAIN_FRACTION,
        "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
        "split_test_fraction": SPLIT_TEST_FRACTION,
        "split_weight_column": SPLIT_WEIGHT_COLUMN,
        "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
        "split_stratify_columns": list(stratify_columns),
        "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
        "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
        "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
        "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
        "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
        "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
        "train_background_hard_negative_buckets": list(
            TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
        ),
        "train_background_reduced_buckets": list(TRAIN_BACKGROUND_REDUCED_BUCKETS),
    }


def try_load_split_manifest(
    df_master: pd.DataFrame,
    manifest_settings: dict,
) -> pd.DataFrame | None:
    """Carga el manifiesto si existe y sigue siendo compatible con el dataset."""
    if not REUSE_SPLIT_MANIFEST:
        return None

    if not SPLIT_MANIFEST_PATH.exists() or not SPLIT_MANIFEST_INFO_PATH.exists():
        return None

    try:
        with open(SPLIT_MANIFEST_INFO_PATH, "r", encoding="utf-8") as file_handle:
            stored_settings = json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Aviso: no se ha podido leer la informacion del manifiesto ({exc}).")
        return None

    if stored_settings != manifest_settings:
        print(
            "Aviso: el manifiesto persistido no coincide con la configuracion actual. "
            "Se regenerara."
        )
        return None

    try:
        manifest_df = pd.read_csv(SPLIT_MANIFEST_PATH)
    except Exception as exc:
        print(
            f"Aviso: no se ha podido leer el manifiesto de splits ({exc}). Se regenerara."
        )
        return None

    required_columns = {"path", "split", "train_keep"}
    if not required_columns.issubset(set(manifest_df.columns)):
        print("Aviso: faltan columnas obligatorias en el manifiesto. Se regenerara.")
        return None

    current_paths = set(df_master["path"].astype(str))
    manifest_paths = set(manifest_df["path"].astype(str))
    if current_paths != manifest_paths:
        print("Aviso: el manifiesto no corresponde al dataset actual. Se regenerara.")
        return None

    manifest_subset = manifest_df.loc[:, ["path", "split", "train_keep"]].copy()
    manifest_subset["split"] = manifest_subset["split"].astype(str)
    manifest_subset["train_keep"] = (
        manifest_subset["train_keep"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    if manifest_subset["train_keep"].isna().any():
        print(
            "Aviso: el manifiesto contiene valores no validos en `train_keep`. "
            "Se regenerara."
        )
        return None

    manifest_subset["train_keep"] = manifest_subset["train_keep"].astype(bool)
    return manifest_subset


def save_split_manifest(manifest_df: pd.DataFrame, manifest_settings: dict) -> None:
    """Persistencia estable del split y del submuestreo de train."""
    if not SAVE_SPLIT_MANIFEST:
        return

    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(SPLIT_MANIFEST_PATH, index=False)

    with open(SPLIT_MANIFEST_INFO_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(manifest_settings, file_handle, indent=2)


def build_split_manifest(
    df_master: pd.DataFrame,
    stratify_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Genera el manifiesto completo de split y curacion de train."""
    train_idx, temp_idx = grouped_stratified_split(
        df_master,
        group_col="safe_group_id",
        test_size=(1.0 - SPLIT_TRAIN_FRACTION),
        stratify_columns=stratify_columns,
        random_state=RANDOM_SEED,
        weight_col=SPLIT_WEIGHT_COLUMN,
        row_cost_weight=SPLIT_ROW_COST_WEIGHT,
    )

    temp_df = df_master.iloc[temp_idx].copy()
    relative_test_fraction = SPLIT_TEST_FRACTION / (
        SPLIT_VALIDATION_FRACTION + SPLIT_TEST_FRACTION
    )
    validation_idx, test_idx = grouped_stratified_split(
        temp_df,
        group_col="safe_group_id",
        test_size=relative_test_fraction,
        stratify_columns=stratify_columns,
        random_state=RANDOM_SEED,
        weight_col=SPLIT_WEIGHT_COLUMN,
        row_cost_weight=SPLIT_ROW_COST_WEIGHT,
    )

    split_labels = pd.Series("unassigned", index=df_master.index, dtype="object")
    split_labels.iloc[train_idx] = "train"
    split_labels.iloc[temp_df.iloc[validation_idx].index] = "validation"
    split_labels.iloc[temp_df.iloc[test_idx].index] = "test"

    manifest_df = df_master.loc[
        :,
        [
            "path",
            "label",
            "source",
            "domain",
            "safe_group_id",
            "background_sampling_bucket",
            "num_chunks",
        ],
    ].copy()
    manifest_df["split"] = split_labels.values
    manifest_df["train_keep"] = False

    train_curated_df = select_training_background_subset(
        df_master.loc[split_labels == "train"].copy(),
        random_state=RANDOM_SEED,
        target_bg_to_siren_ratio=TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
    )
    manifest_df.loc[train_curated_df.index, "train_keep"] = (
        train_curated_df["train_keep"].astype(bool)
    )
    return manifest_df


def summarize_training_selection(
    full_train_df: pd.DataFrame,
    selected_train_df: pd.DataFrame,
) -> dict:
    """Resume en formato JSON-friendly que se ha quedado dentro de train."""
    full_bg_df = full_train_df.loc[
        full_train_df["label"].astype(str).str.strip().str.lower() == "background"
    ]
    selected_bg_df = selected_train_df.loc[
        selected_train_df["label"].astype(str).str.strip().str.lower() == "background"
    ]

    siren_selected_mask = selected_train_df["label"].astype(str).str.strip().str.lower().isin(
        {"siren", "sirena"}
    )
    summary = {
        "train_split_audio_count": int(len(full_train_df)),
        "train_selected_audio_count": int(len(selected_train_df)),
        "train_split_chunk_count": int(full_train_df["num_chunks"].sum()),
        "train_selected_chunk_count": int(selected_train_df["num_chunks"].sum()),
        "siren_audio_count": int(siren_selected_mask.sum()),
        "siren_chunk_count": int(
            selected_train_df.loc[siren_selected_mask, "num_chunks"].sum()
        ),
        "background_audio_count_before": int(len(full_bg_df)),
        "background_audio_count_after": int(len(selected_bg_df)),
        "background_chunk_count_before": int(full_bg_df["num_chunks"].sum()),
        "background_chunk_count_after": int(selected_bg_df["num_chunks"].sum()),
        "background_bucket_count_before": int(
            full_bg_df["background_sampling_bucket"].nunique(dropna=True)
        ),
        "background_bucket_count_after": int(
            selected_bg_df["background_sampling_bucket"].nunique(dropna=True)
        ),
        "background_chunk_ratio_after": float(
            selected_bg_df["num_chunks"].sum()
            / max(1.0, selected_train_df.loc[siren_selected_mask, "num_chunks"].sum())
        ),
        "background_chunks_by_bucket_before": full_bg_df.groupby(
            "background_sampling_bucket"
        )["num_chunks"].sum().sort_values(ascending=False).astype(int).to_dict(),
        "background_chunks_by_bucket_after": selected_bg_df.groupby(
            "background_sampling_bucket"
        )["num_chunks"].sum().sort_values(ascending=False).astype(int).to_dict(),
    }
    return summary


def print_split_diagnostics(
    split_name: str,
    df: pd.DataFrame,
    stratify_columns: tuple[str, ...] = SPLIT_STRATIFY_COLUMNS,
) -> None:
    """
    Imprime diagnosticos del split tanto por audio como por chunk para detectar
    desajustes entre train, validation y test.
    """
    print(f"\nDiagnostico del split: {split_name}")
    print(f"Audios: {len(df)} | Chunks validos: {int(df['num_chunks'].sum())}")
    print("Distribucion por audio (label):")
    print(df["label"].value_counts())
    print("Distribucion por chunk (label):")
    label_chunk_counts = df.groupby("label")["num_chunks"].sum()
    print(label_chunk_counts)
    if {"background", "siren"}.issubset(set(label_chunk_counts.index)):
        ratio = float(
            label_chunk_counts["background"] / max(1.0, label_chunk_counts["siren"])
        )
        print(f"Ratio background/siren por chunk: {ratio:.2f}")

    available_stratify_columns = [
        column for column in stratify_columns if column in df.columns
    ]
    if available_stratify_columns:
        print(f"Distribucion por audio ({', '.join(available_stratify_columns)}):")
        print(df.groupby(available_stratify_columns).size())
        print(f"Distribucion por chunk ({', '.join(available_stratify_columns)}):")
        print(df.groupby(available_stratify_columns)["num_chunks"].sum())
    else:
        print("No hay columnas adicionales disponibles para diagnostico estratificado.")

    if "source" in df.columns:
        print("Top fuentes por audio:")
        print(df["source"].value_counts().head(10))
        print("Top fuentes por chunk:")
        print(df.groupby("source")["num_chunks"].sum().sort_values(ascending=False).head(10))

    if "background_sampling_bucket" in df.columns:
        background_df = df.loc[
            df["label"].astype(str).str.strip().str.lower() == "background"
        ]
        if not background_df.empty:
            print("Top buckets background por chunk:")
            print(
                background_df.groupby("background_sampling_bucket")["num_chunks"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )


def evaluate_candidates(
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_val_scaled: np.ndarray,
    y_val: np.ndarray,
    positive_class_encoded: int,
    class_weights: dict[int, float] | None,
) -> tuple[str, dict[str, dict], dict[str, object]]:
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    sample_weight = build_sample_weight_vector(y_train, class_weights)

    print("\n" + "=" * 56)
    print(" TORNEO DE CLASIFICADORES (SELECCION ORIENTADA A DETECCION) ")
    print("=" * 56)

    for model_name, model in candidate_models().items():
        print(f"\nEntrenando {model_name}...")
        model, used_sample_weight = fit_model_with_optional_weights(
            model,
            x_train_scaled,
            y_train,
            sample_weight,
        )

        probability_index = resolve_probability_index(
            model.classes_,
            positive_class_encoded,
        )
        val_positive_probs = model.predict_proba(x_val_scaled)[:, probability_index]
        threshold_info, threshold_table = select_best_threshold(
            y_val,
            val_positive_probs,
            positive_class_encoded=positive_class_encoded,
            thresholds=THRESHOLD_GRID,
            target_false_alarms_per_min=TARGET_FALSE_ALARMS_PER_MIN,
            chunk_step_s=CHUNK_STEP_S,
        )
        val_pred_binary = (
            val_positive_probs >= float(threshold_info["threshold"])
        ).astype(np.int32)
        validation_metrics = compute_detection_metrics(
            y_val,
            val_pred_binary,
            val_positive_probs,
            positive_class_encoded=positive_class_encoded,
            chunk_step_s=CHUNK_STEP_S,
        )

        fitted_models[model_name] = model
        results[model_name] = {
            "validation_metrics": validation_metrics,
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "threshold_info": threshold_info,
            "threshold_table": threshold_table.to_dict(orient="records"),
            "used_sample_weight": used_sample_weight,
        }

        print_metrics_block(f"Validacion de {model_name}", validation_metrics)
        print(
            f"Umbral seleccionado: {threshold_info['threshold']:.2f} | "
            f"Restriccion de falsas alarmas cumplida: {threshold_info['constraint_satisfied']}"
        )
        if sample_weight is not None and not used_sample_weight:
            print(
                "Aviso: este modelo no acepta sample_weight; se entrena sin balanceo explicito."
            )

    winner_name = max(results, key=lambda name: build_model_selection_key(results[name]))
    return winner_name, results, fitted_models


def train_all_final_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    positive_class_encoded: int,
    class_weights: dict[int, float] | None,
) -> tuple[dict[str, object], StandardScaler, dict[str, dict]]:
    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.hstack([y_train, y_val])

    scaler = StandardScaler()
    x_train_val_scaled = scaler.fit_transform(x_train_val)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    final_models: dict[str, object] = {}
    final_results: dict[str, dict] = {}

    sample_weight = build_sample_weight_vector(y_train_val, class_weights)

    for model_name, base_model in candidate_models().items():
        final_model = clone(base_model)
        final_model, used_sample_weight = fit_model_with_optional_weights(
            final_model,
            x_train_val_scaled,
            y_train_val,
            sample_weight,
        )

        probability_index = resolve_probability_index(
            final_model.classes_,
            positive_class_encoded,
        )
        val_positive_probs = final_model.predict_proba(x_val_scaled)[:, probability_index]
        recalibrated_threshold_info, recalibrated_threshold_table = select_best_threshold(
            y_val,
            val_positive_probs,
            positive_class_encoded=positive_class_encoded,
            thresholds=THRESHOLD_GRID,
            target_false_alarms_per_min=TARGET_FALSE_ALARMS_PER_MIN,
            chunk_step_s=CHUNK_STEP_S,
        )
        val_pred_binary = (
            val_positive_probs >= float(recalibrated_threshold_info["threshold"])
        ).astype(np.int32)
        validation_metrics_refit = compute_detection_metrics(
            y_val,
            val_pred_binary,
            val_positive_probs,
            positive_class_encoded=positive_class_encoded,
            chunk_step_s=CHUNK_STEP_S,
        )

        test_positive_probs = final_model.predict_proba(x_test_scaled)[:, probability_index]
        threshold = float(recalibrated_threshold_info["threshold"])
        test_pred_binary = (test_positive_probs >= threshold).astype(np.int32)
        test_metrics = compute_detection_metrics(
            y_test,
            test_pred_binary,
            test_positive_probs,
            positive_class_encoded=positive_class_encoded,
            chunk_step_s=CHUNK_STEP_S,
        )

        final_models[model_name] = final_model
        final_results[model_name] = {
            "validation_metrics_refit": validation_metrics_refit,
            "recalibrated_threshold_info": recalibrated_threshold_info,
            "recalibrated_threshold_table": recalibrated_threshold_table.to_dict(
                orient="records"
            ),
            "test_metrics": test_metrics,
            "test_accuracy": float(test_metrics["accuracy"]),
            "threshold": threshold,
            "used_sample_weight": used_sample_weight,
        }

    return final_models, scaler, final_results


def plot_rf_feature_importance(fitted_rf_model: RandomForestClassifier) -> None:
    importances = fitted_rf_model.feature_importances_
    df_importances = pd.DataFrame(
        {
            "Caracteristica": feature_names(),
            "Importancia": importances,
        }
    ).sort_values(by="Importancia", ascending=False)

    top_10 = df_importances.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_10["Caracteristica"][::-1],
        top_10["Importancia"][::-1],
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Top 10 features del Random Forest")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def save_experiment_outputs(report_payload: dict, winner_postprocessing: dict) -> None:
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_EXPERIMENT_REPORT:
        with open(EXPERIMENT_REPORT_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(make_jsonable(report_payload), file_handle, indent=2)
        print(f"Reporte del experimento guardado en: {EXPERIMENT_REPORT_PATH}")

    if SAVE_POSTPROCESSING_CONFIG:
        with open(POSTPROCESSING_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(make_jsonable(winner_postprocessing), file_handle, indent=2)
        print(f"Configuracion de postprocesado guardada en: {POSTPROCESSING_PATH}")


def main() -> None:
    validate_required_paths()
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cargando metadata desde: {METADATA_PATH}")
    if RUNTIME_CONFIG_PATH is not None:
        print(f"Overrides de configuracion cargados desde: {RUNTIME_CONFIG_PATH}")
        print(
            "Claves sobrescritas: "
            f"{', '.join(sorted(RUNTIME_CONFIG_OVERRIDES.keys()))}"
        )

    df_master = pd.read_csv(METADATA_PATH)
    if df_master["path"].duplicated().any():
        duplicated_paths = (
            df_master.loc[df_master["path"].duplicated(), "path"].astype(str).tolist()[:10]
        )
        raise RuntimeError(
            "El metadata contiene paths duplicados. Esto rompe el manifiesto de splits. "
            f"Ejemplos: {duplicated_paths}"
        )

    df_master = enrich_metadata_columns(df_master)
    effective_split_stratify_columns = resolve_stratify_columns(
        df_master,
        SPLIT_STRATIFY_COLUMNS,
        fallback_columns=("label", "source"),
    )

    label_encoder = LabelEncoder()
    df_master["target"] = label_encoder.fit_transform(df_master["label"])
    positive_label = resolve_positive_label(label_encoder.classes_)
    positive_class_encoded = int(label_encoder.transform([positive_label])[0])

    print("Contando chunks validos de todo el dataset. Esto puede tardar unos minutos...")
    df_master = add_chunk_count_column(
        df_master,
        base_path=DATASET_DIR,
        sr=SAMPLE_RATE,
        chunk_length_s=CHUNK_LENGTH_S,
        overlap_s=OVERLAP_S,
    )

    print(
        f"Dataset completo -> audios: {len(df_master)} | "
        f"chunks validos totales: {int(df_master['num_chunks'].sum())}"
    )
    print("Distribucion por chunk en el dataset completo:")
    print(df_master.groupby("label")["num_chunks"].sum().to_string())

    manifest_settings = build_split_manifest_settings(effective_split_stratify_columns)
    split_manifest = try_load_split_manifest(df_master, manifest_settings)
    reused_split_manifest = split_manifest is not None

    if split_manifest is None:
        print(
            "Generando un nuevo manifiesto de splits agrupados y estratificados por: "
            f"{', '.join(effective_split_stratify_columns)}"
        )
        split_manifest = build_split_manifest(df_master, effective_split_stratify_columns)
        save_split_manifest(split_manifest, manifest_settings)
        if SAVE_SPLIT_MANIFEST:
            print(f"Manifiesto de splits guardado en: {SPLIT_MANIFEST_PATH}")
            print(f"Informacion del manifiesto guardada en: {SPLIT_MANIFEST_INFO_PATH}")
    else:
        print(f"Usando manifiesto de splits existente: {SPLIT_MANIFEST_PATH}")

    df_master = df_master.merge(
        split_manifest.loc[:, ["path", "split", "train_keep"]],
        on="path",
        how="left",
        validate="one_to_one",
    )

    if df_master["split"].isna().any():
        raise RuntimeError(
            "Hay audios sin split asignado tras cargar el manifiesto. "
            "Revisa el metadata y el fichero de manifiesto."
        )

    train_split_df = df_master.loc[df_master["split"] == "train"].copy()
    val_df = df_master.loc[df_master["split"] == "validation"].copy()
    test_df = df_master.loc[df_master["split"] == "test"].copy()
    train_df = train_split_df.loc[
        train_split_df["label"].astype(str).str.strip().str.lower().isin({"siren", "sirena"})
        | train_split_df["train_keep"].astype(bool)
    ].copy()
    train_selection_summary = summarize_training_selection(train_split_df, train_df)

    print(
        "\nSplits agrupados y estratificados por: "
        f"{', '.join(effective_split_stratify_columns)}"
    )
    print(
        f"Archivos originales -> Train: {len(train_split_df)} | "
        f"Validation: {len(val_df)} | Test: {len(test_df)}"
    )
    print(
        f"Chunks validos -> Train: {int(train_split_df['num_chunks'].sum())} | "
        f"Validation: {int(val_df['num_chunks'].sum())} | "
        f"Test: {int(test_df['num_chunks'].sum())}"
    )
    print(
        "Configuracion temporal -> "
        f"chunk={CHUNK_LENGTH_S:.3f}s | "
        f"use_overlap={USE_OVERLAP} | "
        f"overlap_configurado={CONFIGURED_OVERLAP_S:.3f}s | "
        f"overlap_efectivo={OVERLAP_S:.3f}s | "
        f"step={CHUNK_STEP_S:.3f}s"
    )
    print(
        "Curacion de train -> audios: {train_audio_count} -> {selected_audio_count} | "
        "chunks: {train_chunk_count} -> {selected_chunk_count}".format(
            train_audio_count=train_selection_summary["train_split_audio_count"],
            selected_audio_count=train_selection_summary["train_selected_audio_count"],
            train_chunk_count=train_selection_summary["train_split_chunk_count"],
            selected_chunk_count=train_selection_summary["train_selected_chunk_count"],
        )
    )
    print(
        "Background train -> audios: {before_audio} -> {after_audio} | "
        "chunks: {before_chunk} -> {after_chunk} | buckets: {before_bucket} -> {after_bucket} | "
        "ratio bg/sirena final: {ratio:.2f}".format(
            before_audio=train_selection_summary["background_audio_count_before"],
            after_audio=train_selection_summary["background_audio_count_after"],
            before_chunk=train_selection_summary["background_chunk_count_before"],
            after_chunk=train_selection_summary["background_chunk_count_after"],
            before_bucket=train_selection_summary["background_bucket_count_before"],
            after_bucket=train_selection_summary["background_bucket_count_after"],
            ratio=train_selection_summary["background_chunk_ratio_after"],
        )
    )
    print("Distribucion de clases en train (split completo antes de curacion):")
    print(train_split_df["label"].value_counts())
    print_split_diagnostics(
        "Train split completo",
        train_split_df,
        effective_split_stratify_columns,
    )
    print("Distribucion de clases en train (subset usado por el modelo):")
    print(train_df["label"].value_counts())
    print_split_diagnostics("Train usado por el modelo", train_df, effective_split_stratify_columns)
    print_split_diagnostics("Validation", val_df, effective_split_stratify_columns)
    print_split_diagnostics("Test", test_df, effective_split_stratify_columns)

    class_weights_metadata = None
    chunk_counts_metadata = None
    if USE_CLASS_WEIGHTS:
        class_weights_metadata, chunk_counts_metadata = compute_class_sample_weights_from_chunks(
            train_df
        )
        print(f"Chunks por clase en train (metadata): {chunk_counts_metadata}")
        print(f"Pesos por clase en train (metadata): {class_weights_metadata}")
    else:
        print("Class weights desactivados.")

    if USE_DATA_AUGMENTATION:
        print(
            "Data augmentation activada en train -> "
            f"prob={AUGMENTATION_APPLY_PROB:.2f} | "
            f"copias_extra={AUGMENTATION_EXTRA_COPIES} | "
            f"seed={AUGMENTATION_RANDOM_SEED} | "
            f"pitch_shift={USE_PITCH_SHIFT_AUGMENTATION} | "
            f"eq={USE_SPECTRAL_EQ_AUGMENTATION} | "
            f"eq_objetivo={EQ_AUGMENTATION_PROB:.2f} | "
            f"eq_condicional={EFFECTIVE_EQ_APPLY_PROB:.2f}"
        )
    else:
        print("Data augmentation desactivada.")

    print("\nExtrayendo features acusticas...")
    print("Procesando train set...")
    x_train, y_train = build_dataset_in_memory(
        train_df,
        base_path=DATASET_DIR,
        sr=SAMPLE_RATE,
        chunk_length_s=CHUNK_LENGTH_S,
        overlap_s=OVERLAP_S,
        augment=USE_DATA_AUGMENTATION,
        augmentation_apply_prob=AUGMENTATION_APPLY_PROB,
        augmentation_extra_copies=AUGMENTATION_EXTRA_COPIES,
        random_seed=AUGMENTATION_RANDOM_SEED,
        use_pitch_shift_augmentation=USE_PITCH_SHIFT_AUGMENTATION,
        use_spectral_eq_augmentation=USE_SPECTRAL_EQ_AUGMENTATION,
        spectral_eq_apply_probability=EFFECTIVE_EQ_APPLY_PROB,
        eq_one_filter_prob=EQ_ONE_FILTER_PROB,
        eq_shelf_gain_db_max=EQ_SHELF_GAIN_DB_MAX,
        eq_bell_gain_db_max_siren_band=EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
        eq_total_gain_db_limit=EQ_TOTAL_GAIN_DB_LIMIT,
        eq_low_shelf_cutoff_hz_range=EQ_LOW_SHELF_CUTOFF_HZ_RANGE,
        eq_bell_center_hz_range=EQ_BELL_CENTER_HZ_RANGE,
        eq_high_shelf_cutoff_hz_range=EQ_HIGH_SHELF_CUTOFF_HZ_RANGE,
        eq_bell_bandwidth_octaves_range=EQ_BELL_BANDWIDTH_OCTAVES_RANGE,
        eq_shelf_sharpness_range=EQ_SHELF_SHARPNESS_RANGE,
    )
    print("Procesando validation set...")
    x_val, y_val = build_dataset_in_memory(
        val_df,
        base_path=DATASET_DIR,
        sr=SAMPLE_RATE,
        chunk_length_s=CHUNK_LENGTH_S,
        overlap_s=OVERLAP_S,
    )
    print("Procesando test set...")
    x_test, y_test = build_dataset_in_memory(
        test_df,
        base_path=DATASET_DIR,
        sr=SAMPLE_RATE,
        chunk_length_s=CHUNK_LENGTH_S,
        overlap_s=OVERLAP_S,
    )

    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise RuntimeError(
            "Alguna particion quedo vacia tras la extraccion de features. "
            "Revisa rutas y audios del dataset."
        )

    print(
        "\nDimensiones finales -> "
        f"train={x_train.shape} | val={x_val.shape} | test={x_test.shape}"
    )

    exact_train_class_weights = None
    exact_train_chunk_counts = None
    if USE_CLASS_WEIGHTS:
        exact_train_class_weights, exact_train_chunk_counts = compute_class_weights_from_targets(
            y_train
        )
        print(f"Chunks por clase en train (extraidos): {exact_train_chunk_counts}")
        print(f"Pesos por clase en train (extraidos): {exact_train_class_weights}")

    selection_scaler = StandardScaler()
    x_train_scaled = selection_scaler.fit_transform(x_train)
    x_val_scaled = selection_scaler.transform(x_val)

    winner_name, results, fitted_models = evaluate_candidates(
        x_train_scaled,
        y_train,
        x_val_scaled,
        y_val,
        positive_class_encoded=positive_class_encoded,
        class_weights=exact_train_class_weights,
    )

    winner_validation_metrics = results[winner_name]["validation_metrics"]
    winner_selection_threshold_info = results[winner_name]["threshold_info"]

    print("\n" + "=" * 56)
    print(" RESULTADOS DE SELECCION ")
    print("=" * 56)
    print(f"Ganador por validacion orientada a deteccion: {winner_name}")
    print_metrics_block("Resumen de validacion del ganador", winner_validation_metrics)
    print(
        f"Umbral recomendado para '{positive_label}': "
        f"{winner_selection_threshold_info['threshold']:.2f}"
    )

    x_train_val = np.vstack([x_train, x_val])
    y_train_val = np.hstack([y_train, y_val])
    exact_train_val_class_weights = None
    exact_train_val_chunk_counts = None
    if USE_CLASS_WEIGHTS:
        exact_train_val_class_weights, exact_train_val_chunk_counts = (
            compute_class_weights_from_targets(y_train_val)
        )
        print(
            f"Chunks por clase en train+val (extraidos): {exact_train_val_chunk_counts}"
        )
        print(
            f"Pesos por clase en train+val (extraidos): {exact_train_val_class_weights}"
        )

    final_models, final_scaler, final_results = train_all_final_models(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        positive_class_encoded=positive_class_encoded,
        class_weights=exact_train_val_class_weights,
    )

    print("\n" + "=" * 56)
    print(" EVALUACION FINAL EN TEST ")
    print("=" * 56)
    for model_name, model_result in final_results.items():
        print_metrics_block(
            f"Validacion recalibrada de {model_name}",
            model_result["validation_metrics_refit"],
        )
        print(
            f"Umbral recalibrado para {model_name}: {model_result['threshold']:.2f}"
        )
        print_metrics_block(
            f"Test de {model_name}",
            model_result["test_metrics"],
        )
        print(f"Umbral usado en test: {model_result['threshold']:.2f}\n")

    final_model = final_models[winner_name]
    winner_threshold_info = final_results[winner_name]["recalibrated_threshold_info"]
    winner_test_metrics = final_results[winner_name]["test_metrics"]

    model_scores = {}
    for model_name in candidate_models().keys():
        model_scores[model_name] = {
            "validation_metrics": results[model_name]["validation_metrics"],
            "validation_threshold": float(results[model_name]["threshold_info"]["threshold"]),
            "validation_threshold_info": results[model_name]["threshold_info"],
            "validation_metrics_refit": final_results[model_name]["validation_metrics_refit"],
            "final_threshold": float(
                final_results[model_name]["recalibrated_threshold_info"]["threshold"]
            ),
            "final_threshold_info": final_results[model_name]["recalibrated_threshold_info"],
            "test_metrics": final_results[model_name]["test_metrics"],
            "used_sample_weight_in_validation_fit": bool(
                results[model_name]["used_sample_weight"]
            ),
            "used_sample_weight_in_final_fit": bool(
                final_results[model_name]["used_sample_weight"]
            ),
        }

    report_payload = {
        "run_name": RUN_BASENAME,
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "runtime_config_path": RUNTIME_CONFIG_PATH,
        "runtime_config_overrides": RUNTIME_CONFIG_OVERRIDES,
        "selection_metric": SELECTION_METRIC,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "target_false_alarms_per_min": TARGET_FALSE_ALARMS_PER_MIN,
        "sample_rate": SAMPLE_RATE,
        "chunk_length_s": CHUNK_LENGTH_S,
        "use_overlap": USE_OVERLAP,
        "configured_overlap_s": CONFIGURED_OVERLAP_S,
        "overlap_s": OVERLAP_S,
        "decision_step_s": CHUNK_STEP_S,
        "split_manifest_path": str(SPLIT_MANIFEST_PATH),
        "split_manifest_info_path": str(SPLIT_MANIFEST_INFO_PATH),
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "split_manifest_reused": reused_split_manifest,
        "split_train_fraction": SPLIT_TRAIN_FRACTION,
        "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
        "split_test_fraction": SPLIT_TEST_FRACTION,
        "split_stratify_columns": list(effective_split_stratify_columns),
        "split_weight_column": SPLIT_WEIGHT_COLUMN,
        "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
        "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
        "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
        "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
        "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
        "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
        "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
        "train_background_hard_negative_buckets": list(
            TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
        ),
        "train_background_reduced_buckets": list(TRAIN_BACKGROUND_REDUCED_BUCKETS),
        "train_selection_summary": train_selection_summary,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_data_augmentation": USE_DATA_AUGMENTATION,
        "use_pitch_shift_augmentation": USE_PITCH_SHIFT_AUGMENTATION,
        "augmentation_apply_prob": AUGMENTATION_APPLY_PROB,
        "augmentation_extra_copies": AUGMENTATION_EXTRA_COPIES,
        "augmentation_random_seed": AUGMENTATION_RANDOM_SEED,
        "use_spectral_eq_augmentation": USE_SPECTRAL_EQ_AUGMENTATION,
        "eq_augmentation_probability": EQ_AUGMENTATION_PROB,
        "eq_effective_conditional_probability": EFFECTIVE_EQ_APPLY_PROB,
        "eq_one_filter_probability": EQ_ONE_FILTER_PROB,
        "eq_shelf_gain_db_max": EQ_SHELF_GAIN_DB_MAX,
        "eq_bell_gain_db_max_siren_band": EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
        "eq_total_gain_db_limit": EQ_TOTAL_GAIN_DB_LIMIT,
        "eq_low_shelf_cutoff_hz_range": list(EQ_LOW_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_center_hz_range": list(EQ_BELL_CENTER_HZ_RANGE),
        "eq_high_shelf_cutoff_hz_range": list(EQ_HIGH_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_bandwidth_octaves_range": list(EQ_BELL_BANDWIDTH_OCTAVES_RANGE),
        "eq_shelf_sharpness_range": list(EQ_SHELF_SHARPNESS_RANGE),
        "feature_names": feature_names(),
        "positive_label": positive_label,
        "labels": [str(label) for label in label_encoder.classes_],
        "train_chunk_counts_metadata": chunk_counts_metadata or {},
        "train_chunk_counts_extracted": exact_train_chunk_counts or {},
        "train_val_chunk_counts_extracted": exact_train_val_chunk_counts or {},
        "class_weights_metadata": class_weights_metadata or {},
        "class_weights_extracted": exact_train_class_weights or {},
        "class_weights_train_val_extracted": exact_train_val_class_weights or {},
        "per_model": model_scores,
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "threshold_calibration_stage": "validation_after_train_val_refit",
    }

    postprocessing_payload = {
        "bundle_path": str(RUN_OUTPUT_DIR / "clasificador_tradicional_bundle.joblib"),
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "winner_name": winner_name,
        "recommended_chunk_threshold": float(winner_threshold_info["threshold"]),
        "target_false_alarms_per_min": TARGET_FALSE_ALARMS_PER_MIN,
        "chunk_length_s": CHUNK_LENGTH_S,
        "use_overlap": USE_OVERLAP,
        "configured_overlap_s": CONFIGURED_OVERLAP_S,
        "overlap_s": OVERLAP_S,
        "decision_step_s": CHUNK_STEP_S,
        "sample_rate": SAMPLE_RATE,
        "split_manifest_path": str(SPLIT_MANIFEST_PATH),
        "split_manifest_info_path": str(SPLIT_MANIFEST_INFO_PATH),
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "split_manifest_reused": reused_split_manifest,
        "split_train_fraction": SPLIT_TRAIN_FRACTION,
        "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
        "split_test_fraction": SPLIT_TEST_FRACTION,
        "split_stratify_columns": list(effective_split_stratify_columns),
        "split_weight_column": SPLIT_WEIGHT_COLUMN,
        "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
        "labels": [str(label) for label in label_encoder.classes_],
        "positive_label": positive_label,
        "use_data_augmentation": USE_DATA_AUGMENTATION,
        "use_pitch_shift_augmentation": USE_PITCH_SHIFT_AUGMENTATION,
        "augmentation_apply_prob": AUGMENTATION_APPLY_PROB,
        "augmentation_extra_copies": AUGMENTATION_EXTRA_COPIES,
        "use_spectral_eq_augmentation": USE_SPECTRAL_EQ_AUGMENTATION,
        "eq_augmentation_probability": EQ_AUGMENTATION_PROB,
        "eq_effective_conditional_probability": EFFECTIVE_EQ_APPLY_PROB,
        "eq_one_filter_probability": EQ_ONE_FILTER_PROB,
        "eq_shelf_gain_db_max": EQ_SHELF_GAIN_DB_MAX,
        "eq_bell_gain_db_max_siren_band": EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
        "eq_total_gain_db_limit": EQ_TOTAL_GAIN_DB_LIMIT,
        "eq_low_shelf_cutoff_hz_range": list(EQ_LOW_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_center_hz_range": list(EQ_BELL_CENTER_HZ_RANGE),
        "eq_high_shelf_cutoff_hz_range": list(EQ_HIGH_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_bandwidth_octaves_range": list(EQ_BELL_BANDWIDTH_OCTAVES_RANGE),
        "eq_shelf_sharpness_range": list(EQ_SHELF_SHARPNESS_RANGE),
        "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
        "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
        "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
        "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
        "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
        "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
        "train_background_hard_negative_buckets": list(
            TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
        ),
        "train_background_reduced_buckets": list(TRAIN_BACKGROUND_REDUCED_BUCKETS),
        "train_selection_summary": train_selection_summary,
        "saved_models": {
            name: str(RUN_OUTPUT_DIR / f"modelo_sirenas_{name.lower().replace(' ', '_')}.joblib")
            for name in candidate_models().keys()
        },
        "saved_bundles": {
            name: str(build_model_bundle_path(name, output_dir=RUN_OUTPUT_DIR))
            for name in candidate_models().keys()
        },
        "runtime_config_path": RUNTIME_CONFIG_PATH,
        "runtime_config_overrides": RUNTIME_CONFIG_OVERRIDES,
        "validation_metrics": winner_validation_metrics,
        "validation_metrics_refit": final_results[winner_name]["validation_metrics_refit"],
        "selection_threshold_info": winner_selection_threshold_info,
        "final_threshold_info": winner_threshold_info,
        "test_metrics": winner_test_metrics,
        "selection_metric": SELECTION_METRIC,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "threshold_calibration_stage": "validation_after_train_val_refit",
    }

    save_experiment_outputs(report_payload, postprocessing_payload)

    extra_metadata = {
        "run_name": RUN_BASENAME,
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "runtime_config_path": RUNTIME_CONFIG_PATH,
        "runtime_config_overrides": RUNTIME_CONFIG_OVERRIDES,
        "use_overlap": USE_OVERLAP,
        "configured_overlap_s": CONFIGURED_OVERLAP_S,
        "split_manifest_path": str(SPLIT_MANIFEST_PATH),
        "split_manifest_info_path": str(SPLIT_MANIFEST_INFO_PATH),
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "split_manifest_reused": reused_split_manifest,
        "split_train_fraction": SPLIT_TRAIN_FRACTION,
        "split_validation_fraction": SPLIT_VALIDATION_FRACTION,
        "split_test_fraction": SPLIT_TEST_FRACTION,
        "split_stratify_columns": list(effective_split_stratify_columns),
        "split_weight_column": SPLIT_WEIGHT_COLUMN,
        "split_row_cost_weight": SPLIT_ROW_COST_WEIGHT,
        "apply_train_background_subsampling": APPLY_TRAIN_BACKGROUND_SUBSAMPLING,
        "train_background_to_siren_chunk_ratio": TRAIN_BACKGROUND_TO_SIREN_CHUNK_RATIO,
        "train_background_min_groups_per_bucket": TRAIN_BACKGROUND_MIN_GROUPS_PER_BUCKET,
        "train_background_default_bucket_weight": TRAIN_BACKGROUND_DEFAULT_BUCKET_WEIGHT,
        "train_background_hard_negative_weight": TRAIN_BACKGROUND_HARD_NEGATIVE_WEIGHT,
        "train_background_reduced_bucket_weight": TRAIN_BACKGROUND_REDUCED_BUCKET_WEIGHT,
        "train_background_hard_negative_buckets": list(
            TRAIN_BACKGROUND_HARD_NEGATIVE_BUCKETS
        ),
        "train_background_reduced_buckets": list(TRAIN_BACKGROUND_REDUCED_BUCKETS),
        "train_selection_summary": train_selection_summary,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_data_augmentation": USE_DATA_AUGMENTATION,
        "use_pitch_shift_augmentation": USE_PITCH_SHIFT_AUGMENTATION,
        "augmentation_apply_prob": AUGMENTATION_APPLY_PROB,
        "augmentation_extra_copies": AUGMENTATION_EXTRA_COPIES,
        "augmentation_random_seed": AUGMENTATION_RANDOM_SEED,
        "use_spectral_eq_augmentation": USE_SPECTRAL_EQ_AUGMENTATION,
        "eq_augmentation_probability": EQ_AUGMENTATION_PROB,
        "eq_effective_conditional_probability": EFFECTIVE_EQ_APPLY_PROB,
        "eq_one_filter_probability": EQ_ONE_FILTER_PROB,
        "eq_shelf_gain_db_max": EQ_SHELF_GAIN_DB_MAX,
        "eq_bell_gain_db_max_siren_band": EQ_BELL_GAIN_DB_MAX_SIREN_BAND,
        "eq_total_gain_db_limit": EQ_TOTAL_GAIN_DB_LIMIT,
        "eq_low_shelf_cutoff_hz_range": list(EQ_LOW_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_center_hz_range": list(EQ_BELL_CENTER_HZ_RANGE),
        "eq_high_shelf_cutoff_hz_range": list(EQ_HIGH_SHELF_CUTOFF_HZ_RANGE),
        "eq_bell_bandwidth_octaves_range": list(EQ_BELL_BANDWIDTH_OCTAVES_RANGE),
        "eq_shelf_sharpness_range": list(EQ_SHELF_SHARPNESS_RANGE),
        "train_chunk_counts_metadata": chunk_counts_metadata or {},
        "train_chunk_counts_extracted": exact_train_chunk_counts or {},
        "train_val_chunk_counts_extracted": exact_train_val_chunk_counts or {},
        "class_weights_metadata": class_weights_metadata or {},
        "class_weights_extracted": exact_train_class_weights or {},
        "class_weights_train_val_extracted": exact_train_val_class_weights or {},
        "experiment_report_path": str(EXPERIMENT_REPORT_PATH),
        "postprocessing_path": str(POSTPROCESSING_PATH),
        "threshold_calibration_stage": "validation_after_train_val_refit",
    }

    saved_paths = save_training_artifacts(
        model=final_model,
        scaler=final_scaler,
        label_encoder=label_encoder,
        winner_name=winner_name,
        positive_label=positive_label,
        selection_metric=SELECTION_METRIC,
        validation_accuracy=float(winner_validation_metrics["accuracy"]),
        test_accuracy=float(winner_test_metrics["accuracy"]),
        threshold_info=winner_threshold_info,
        all_models=final_models,
        model_scores=model_scores,
        sample_rate=SAMPLE_RATE,
        chunk_seconds=CHUNK_LENGTH_S,
        overlap_seconds=OVERLAP_S,
        threshold_selection_metric=THRESHOLD_SELECTION_METRIC,
        extra_metadata=extra_metadata,
        output_dir=RUN_OUTPUT_DIR,
    )

    print("\nArtefactos guardados:")
    for path_name, path_value in saved_paths.items():
        if isinstance(path_value, dict):
            print(f"- {path_name}:")
            for sub_name, sub_path in path_value.items():
                print(f"  - {sub_name}: {Path(sub_path)}")
        else:
            print(f"- {path_name}: {Path(path_value)}")

    rf_model = final_models.get("Random Forest")
    if rf_model is not None and SHOW_RF_PLOT:
        print("\nMostrando top 10 de features del Random Forest.")
        plot_rf_feature_importance(rf_model)


if __name__ == "__main__":
    main()
