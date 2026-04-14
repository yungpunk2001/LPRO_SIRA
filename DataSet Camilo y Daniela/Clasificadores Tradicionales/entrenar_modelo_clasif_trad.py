import inspect
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

EFFECTIVE_EQ_APPLY_PROB = compute_effective_eq_apply_probability(
    AUGMENTATION_APPLY_PROB,
    EQ_AUGMENTATION_PROB,
)

SELECTION_METRIC = "validation_f2_with_false_alarm_constraint"
THRESHOLD_SELECTION_METRIC = "f2_under_false_alarms_per_min"

RUN_NAME_PREFIX = str(get_config_value("RUN_NAME_PREFIX", "clasif_trad_run"))
RUN_OUTPUT_DIR = Path(str(get_config_value("RUN_OUTPUT_DIR", MODELS_DIR)))
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_BASENAME = f"{RUN_NAME_PREFIX}_{RUN_TIMESTAMP}"
EXPERIMENT_REPORT_PATH = RUN_OUTPUT_DIR / f"{RUN_BASENAME}_reporte.json"
POSTPROCESSING_PATH = RUN_OUTPUT_DIR / f"{RUN_BASENAME}_postprocesado.json"


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


def safe_group_stratified_split(
    df_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    effective_split_stratify_columns = resolve_stratify_columns(
        df_master,
        SPLIT_STRATIFY_COLUMNS,
        fallback_columns=("label", "source"),
    )

    train_idx, temp_idx = grouped_stratified_split(
        df_master,
        group_col="grupo_seguro",
        test_size=0.30,
        stratify_columns=effective_split_stratify_columns,
        random_state=RANDOM_SEED,
    )
    train_df = df_master.iloc[train_idx].reset_index(drop=True)
    temp_df = df_master.iloc[temp_idx].reset_index(drop=True)

    val_idx, test_idx = grouped_stratified_split(
        temp_df,
        group_col="grupo_seguro",
        test_size=0.50,
        stratify_columns=effective_split_stratify_columns,
        random_state=RANDOM_SEED,
    )
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df, effective_split_stratify_columns


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
    df_master = enrich_metadata_columns(df_master)

    label_encoder = LabelEncoder()
    df_master["target"] = label_encoder.fit_transform(df_master["label"])
    positive_label = resolve_positive_label(label_encoder.classes_)
    positive_class_encoded = int(label_encoder.transform([positive_label])[0])
    df_master["grupo_seguro"] = df_master["siren_id"].fillna(df_master["group_id"])
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

    train_df, val_df, test_df, effective_split_stratify_columns = safe_group_stratified_split(
        df_master
    )
    print(
        "\nSplits agrupados y estratificados por: "
        f"{', '.join(effective_split_stratify_columns)}"
    )
    print(
        f"Archivos originales -> Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | Test: {len(test_df)}"
    )
    print(
        f"Chunks validos -> Train: {int(train_df['num_chunks'].sum())} | "
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
        "split_stratify_columns": list(effective_split_stratify_columns),
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
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "use_overlap": USE_OVERLAP,
        "configured_overlap_s": CONFIGURED_OVERLAP_S,
        "split_stratify_columns": list(effective_split_stratify_columns),
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
