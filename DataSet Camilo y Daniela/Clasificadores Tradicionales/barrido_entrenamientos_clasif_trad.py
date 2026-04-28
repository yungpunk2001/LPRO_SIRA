import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Barrido iterativo para `entrenar_modelo_clasif_trad.py`.
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT_PATH = SCRIPT_DIR / "entrenar_modelo_clasif_trad.py"
LOCAL_VENV_PYTHON = SCRIPT_DIR.parent / ".venv" / "Scripts" / "python.exe"


# ---------------------------------------------------------------------------
# Configuracion general del barrido
# ---------------------------------------------------------------------------
PYTHON_COMMAND = [
    os.environ.get("PYTHON_BIN")
    or (str(LOCAL_VENV_PYTHON) if LOCAL_VENV_PYTHON.exists() else sys.executable or "python3")
]
CONTINUE_ON_ERROR = True
STREAM_TRAINING_LOGS = True
MAX_EXPERIMENTS = None

# El barrido debe decidir con la validacion limpia previa al refit. Las
# metricas `validation_refit_*` se conservan solo como referencia diagnostica.
PRIMARY_RANK_METRIC = "validation_event_recall"
FALSE_ALARM_RANK_METRIC = "validation_false_alarm_episodes_per_min"
SECONDARY_RANK_METRIC = "validation_macro_event_coverage"
TERTIARY_RANK_METRIC = "validation_f2"
QUATERNARY_RANK_METRIC = "validation_auc_pr"

AUTO_ITERATIVE_REFINEMENT = True
MAX_SWEEP_ROUNDS = 2
REFINEMENT_TOP_K = 4
REFINEMENT_MAX_CHILDREN_PER_PARENT = 6
ROUND_REPORT_TOP_N = 8


# ---------------------------------------------------------------------------
# Constantes compartidas para refinamiento.
# ---------------------------------------------------------------------------
CORE_CHUNK_LENGTHS_S = [0.5, 1.0]
AUGMENTATION_PROB_CANDIDATES = [0.35, 0.65]
EQ_AUGMENTATION_PROB_CANDIDATES = [0.10, 0.35]
AUGMENTATION_EXTRA_COPIES_CANDIDATES = [1, 2]
TARGET_FALSE_ALARMS_CANDIDATES = [0.5, 1.5]

DEFAULT_RF_N_ESTIMATORS = 200
DEFAULT_SVM_C = 1.0
DEFAULT_SVM_GAMMA = "scale"
DEFAULT_KNN_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# Overrides fijos comunes a todos los experimentos.
# ---------------------------------------------------------------------------
FIXED_OVERRIDES = {
    "RANDOM_SEED": 42,
    "SAVE_EXPERIMENT_REPORT": True,
    "SAVE_POSTPROCESSING_CONFIG": True,
    "SHOW_RF_PLOT": False,
    "TARGET_FALSE_ALARMS_PER_MIN": 1.0,
    "AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT": True,
    "AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES": [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    "AUTO_EVENT_RECALL_RETENTION": 0.95,
    "CHUNK_LENGTH_S": 0.5,
    "USE_OVERLAP": False,
    "OVERLAP_S": 0.0,
    "USE_SPECTRAL_EQ_AUGMENTATION": True,
    "EQ_AUGMENTATION_PROB": 0.20,
}


# ---------------------------------------------------------------------------
# Bateria inicial controlada de experimentos.
# ---------------------------------------------------------------------------
EXPERIMENT_BATTERY = [
    {
        "EXPERIMENT_LABEL": "baseline_limpio",
        "USE_OVERLAP": False,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "solo_chunk_1p0s",
        "CHUNK_LENGTH_S": 1.0,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "solo_class_weights",
        "USE_OVERLAP": False,
        "USE_CLASS_WEIGHTS": True,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "augmentacion_sin_pitch",
        "USE_OVERLAP": False,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": False,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 1,
    },
    {
        "EXPERIMENT_LABEL": "augmentacion_con_pitch",
        "USE_OVERLAP": False,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": True,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 1,
    },
    {
        "EXPERIMENT_LABEL": "augmentacion_doble_copia",
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": False,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 2,
    },
    {
        "EXPERIMENT_LABEL": "class_weights_y_chunk_1p0s",
        "CHUNK_LENGTH_S": 1.0,
        "USE_CLASS_WEIGHTS": True,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "combinacion_prometedora",
        "USE_CLASS_WEIGHTS": True,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": False,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 1,
    },
]


def seconds_tag(seconds):
    text = f"{float(seconds):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "s"


def prob_tag(value):
    return str(value).replace(".", "p")


def normalize_optional_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "si", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def safe_float(value, default):
    if value is None:
        return float(default)
    return float(value)


def safe_int(value, default):
    if value is None:
        return int(default)
    return int(value)


def float_close(left, right, tolerance=1e-9):
    return abs(float(left) - float(right)) <= tolerance


def build_experiment_grid():
    experiments = []
    for index, experiment in enumerate(EXPERIMENT_BATTERY, start=1):
        experiment_copy = dict(experiment)
        label = experiment_copy.get("EXPERIMENT_LABEL", f"seed_{index:03d}")
        experiment_copy.setdefault("EXPERIMENT_TIER", "seed")
        experiment_copy.setdefault("BASE_EXPERIMENT_ID", label)
        experiment_copy.setdefault("BASE_EXPERIMENT_LABEL", label)
        experiment_copy.setdefault(
            "EXPERIMENT_NOTES",
            "Configuracion base de la primera ronda del barrido tradicional.",
        )
        experiments.append(experiment_copy)
    return experiments


def build_experiment_signature_from_overrides(overrides):
    use_overlap = normalize_optional_bool(overrides.get("USE_OVERLAP"), False)
    use_data_augmentation = normalize_optional_bool(
        overrides.get("USE_DATA_AUGMENTATION"),
        False,
    )
    use_pitch_shift = normalize_optional_bool(
        overrides.get("USE_PITCH_SHIFT_AUGMENTATION"),
        True,
    )
    use_spectral_eq = normalize_optional_bool(
        overrides.get("USE_SPECTRAL_EQ_AUGMENTATION"),
        True,
    )

    return (
        round(
            safe_float(
                overrides.get("CHUNK_LENGTH_S"),
                FIXED_OVERRIDES["CHUNK_LENGTH_S"],
            ),
            4,
        ),
        use_overlap,
        (
            round(
                safe_float(
                    overrides.get("OVERLAP_S"),
                    FIXED_OVERRIDES["OVERLAP_S"],
                ),
                4,
            )
            if use_overlap
            else None
        ),
        normalize_optional_bool(overrides.get("USE_CLASS_WEIGHTS"), False),
        use_data_augmentation,
        (
            round(safe_float(overrides.get("AUGMENTATION_APPLY_PROB"), 0.5), 4)
            if use_data_augmentation
            else None
        ),
        (
            safe_int(overrides.get("AUGMENTATION_EXTRA_COPIES"), 1)
            if use_data_augmentation
            else None
        ),
        use_pitch_shift if use_data_augmentation else None,
        use_spectral_eq if use_data_augmentation else None,
        (
            round(
                safe_float(
                    overrides.get("EQ_AUGMENTATION_PROB"),
                    FIXED_OVERRIDES["EQ_AUGMENTATION_PROB"],
                ),
                4,
            )
            if use_data_augmentation and use_spectral_eq
            else None
        ),
        round(
            safe_float(
                overrides.get("TARGET_FALSE_ALARMS_PER_MIN"),
                FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
            ),
            4,
        ),
        round(
            safe_float(
                overrides.get(
                    "TARGET_FALSE_ALARM_EPISODES_PER_MIN",
                    overrides.get(
                        "TARGET_FALSE_ALARMS_PER_MIN",
                        FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
                    ),
                ),
                FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
            ),
            4,
        ),
        normalize_optional_bool(
            overrides.get("AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"),
            FIXED_OVERRIDES["AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"],
        ),
        round(
            safe_float(
                overrides.get("AUTO_EVENT_RECALL_RETENTION"),
                FIXED_OVERRIDES["AUTO_EVENT_RECALL_RETENTION"],
            ),
            4,
        ),
        safe_int(overrides.get("RF_N_ESTIMATORS"), DEFAULT_RF_N_ESTIMATORS),
        round(safe_float(overrides.get("SVM_C"), DEFAULT_SVM_C), 6),
        str(overrides.get("SVM_GAMMA", DEFAULT_SVM_GAMMA)),
        safe_int(overrides.get("KNN_NEIGHBORS"), DEFAULT_KNN_NEIGHBORS),
    )


def derive_overrides_from_row(row):
    use_overlap = normalize_optional_bool(row.get("config_use_overlap"), False)
    use_data_augmentation = normalize_optional_bool(
        row.get("config_use_data_augmentation"),
        False,
    )
    use_spectral_eq = normalize_optional_bool(
        row.get("config_use_spectral_eq_augmentation"),
        True,
    )

    overrides = {
        "CHUNK_LENGTH_S": safe_float(
            row.get("config_chunk_length_s"),
            FIXED_OVERRIDES["CHUNK_LENGTH_S"],
        ),
        "USE_OVERLAP": use_overlap,
        "OVERLAP_S": safe_float(
            row.get("config_overlap_s"),
            FIXED_OVERRIDES["OVERLAP_S"],
        ),
        "USE_CLASS_WEIGHTS": normalize_optional_bool(
            row.get("config_use_class_weights"),
            False,
        ),
        "USE_DATA_AUGMENTATION": use_data_augmentation,
        "TARGET_FALSE_ALARMS_PER_MIN": safe_float(
            row.get("config_target_false_alarms_per_min"),
            FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
        ),
        "TARGET_FALSE_ALARM_EPISODES_PER_MIN": safe_float(
            row.get("config_target_false_alarm_episodes_per_min"),
            row.get("config_target_false_alarms_per_min")
            or FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
        ),
        "AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT": normalize_optional_bool(
            row.get("config_auto_calibrate_false_alarm_episode_limit"),
            FIXED_OVERRIDES["AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"],
        ),
        "AUTO_EVENT_RECALL_RETENTION": safe_float(
            row.get("config_auto_event_recall_retention"),
            FIXED_OVERRIDES["AUTO_EVENT_RECALL_RETENTION"],
        ),
        "AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES": (
            json.loads(row["config_auto_false_alarm_episode_limit_candidates"])
            if row.get("config_auto_false_alarm_episode_limit_candidates")
            else FIXED_OVERRIDES["AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES"]
        ),
        "RF_N_ESTIMATORS": safe_int(
            row.get("config_rf_n_estimators"),
            DEFAULT_RF_N_ESTIMATORS,
        ),
        "SVM_C": safe_float(row.get("config_svm_c"), DEFAULT_SVM_C),
        "SVM_GAMMA": row.get("config_svm_gamma") or DEFAULT_SVM_GAMMA,
        "KNN_NEIGHBORS": safe_int(
            row.get("config_knn_neighbors"),
            DEFAULT_KNN_NEIGHBORS,
        ),
    }

    if use_data_augmentation:
        overrides["USE_PITCH_SHIFT_AUGMENTATION"] = normalize_optional_bool(
            row.get("config_use_pitch_shift_augmentation"),
            True,
        )
        overrides["AUGMENTATION_APPLY_PROB"] = safe_float(
            row.get("config_augmentation_apply_prob"),
            0.5,
        )
        overrides["AUGMENTATION_EXTRA_COPIES"] = safe_int(
            row.get("config_augmentation_extra_copies"),
            1,
        )
        overrides["USE_SPECTRAL_EQ_AUGMENTATION"] = use_spectral_eq
        if use_spectral_eq:
            overrides["EQ_AUGMENTATION_PROB"] = safe_float(
                row.get("config_eq_augmentation_prob"),
                FIXED_OVERRIDES["EQ_AUGMENTATION_PROB"],
            )

    return overrides


def build_experiment_signature_from_row(row):
    return build_experiment_signature_from_overrides(derive_overrides_from_row(row))


def build_round_dir_name(round_number):
    return f"ronda_{round_number:02d}"


def format_row_compact(row, rank=None):
    rank_prefix = f"{rank}. " if rank is not None else ""
    return (
        f"{rank_prefix}{row['experiment_id']} ({row['experiment_label']}) | "
        f"round={row.get('sweep_round')} | "
        f"tier={row.get('experiment_tier')} | "
        f"base={row.get('base_experiment_id')} | "
        f"winner={row.get('winner_name')} | "
        f"chunk={row.get('config_chunk_length_s')} s | "
        f"feasible={normalize_optional_bool(row.get('selected_threshold_constraint_satisfied'), False)} | "
        f"overlap={row.get('config_use_overlap')} | "
        f"weights={row.get('config_use_class_weights')} | "
        f"augment={row.get('config_use_data_augmentation')} | "
        f"val_event_recall={safe_metric(row, 'validation_event_recall', default=0.0):.4f} | "
        f"val_macro_cov={safe_metric(row, 'validation_macro_event_coverage', default=0.0):.4f} | "
        f"val_f2={safe_metric(row, 'validation_f2', default=0.0):.4f} | "
        f"val_auc_pr={safe_metric(row, 'validation_auc_pr', default=0.0):.4f} | "
        f"fa_epi/min={safe_metric(row, 'validation_false_alarm_episodes_per_min', default=0.0):.2f} | "
        f"fa_chunk/min={safe_metric(row, 'validation_false_alarms_per_min', default=0.0):.2f} | "
        f"fa_limit={safe_metric(row, 'selected_false_alarm_episode_limit', default=0.0):.2f}"
    )


def safe_metric(row, key, default=-1.0):
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def build_summary_row(
    experiment_id,
    overrides,
    postprocess_data,
    status,
    return_code,
    log_path,
    sweep_round,
    round_dir,
):
    validation_metrics = postprocess_data.get("validation_metrics") or {}
    validation_metrics_refit = postprocess_data.get("validation_metrics_refit") or {}
    test_metrics = postprocess_data.get("test_metrics") or {}
    selection_threshold_info = postprocess_data.get("selection_threshold_info") or {}
    final_threshold_info = postprocess_data.get("final_threshold_info") or {}

    return {
        "experiment_id": experiment_id,
        "experiment_label": overrides.get("EXPERIMENT_LABEL"),
        "sweep_round": sweep_round,
        "round_dir": str(round_dir),
        "experiment_tier": overrides.get("EXPERIMENT_TIER"),
        "base_experiment_id": overrides.get("BASE_EXPERIMENT_ID"),
        "base_experiment_label": overrides.get("BASE_EXPERIMENT_LABEL"),
        "parent_experiment_id": overrides.get("PARENT_EXPERIMENT_ID"),
        "parent_experiment_label": overrides.get("PARENT_EXPERIMENT_LABEL"),
        "parent_sweep_round": overrides.get("PARENT_SWEEP_ROUND"),
        "experiment_notes": overrides.get("EXPERIMENT_NOTES"),
        "status": status,
        "return_code": return_code,
        "winner_name": postprocess_data.get("winner_name"),
        "config_chunk_length_s": postprocess_data.get(
            "chunk_length_s",
            overrides.get("CHUNK_LENGTH_S"),
        ),
        "config_use_overlap": postprocess_data.get(
            "use_overlap",
            overrides.get("USE_OVERLAP"),
        ),
        "config_overlap_s": postprocess_data.get(
            "overlap_s",
            overrides.get("OVERLAP_S"),
        ),
        "config_use_class_weights": postprocess_data.get(
            "use_class_weights",
            overrides.get("USE_CLASS_WEIGHTS"),
        ),
        "config_use_data_augmentation": postprocess_data.get(
            "use_data_augmentation",
            overrides.get("USE_DATA_AUGMENTATION"),
        ),
        "config_use_pitch_shift_augmentation": postprocess_data.get(
            "use_pitch_shift_augmentation",
            overrides.get("USE_PITCH_SHIFT_AUGMENTATION"),
        ),
        "config_use_spectral_eq_augmentation": postprocess_data.get(
            "use_spectral_eq_augmentation",
            overrides.get("USE_SPECTRAL_EQ_AUGMENTATION"),
        ),
        "config_eq_augmentation_prob": postprocess_data.get(
            "eq_augmentation_probability",
            overrides.get("EQ_AUGMENTATION_PROB"),
        ),
        "config_augmentation_apply_prob": postprocess_data.get(
            "augmentation_apply_prob",
            overrides.get("AUGMENTATION_APPLY_PROB"),
        ),
        "config_augmentation_extra_copies": postprocess_data.get(
            "augmentation_extra_copies",
            overrides.get("AUGMENTATION_EXTRA_COPIES"),
        ),
        "config_target_false_alarms_per_min": postprocess_data.get(
            "target_false_alarms_per_min",
            overrides.get(
                "TARGET_FALSE_ALARMS_PER_MIN",
                FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
            ),
        ),
        "config_target_false_alarm_episodes_per_min": postprocess_data.get(
            "target_false_alarm_episodes_per_min",
            overrides.get(
                "TARGET_FALSE_ALARM_EPISODES_PER_MIN",
                overrides.get(
                    "TARGET_FALSE_ALARMS_PER_MIN",
                    FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
                ),
            ),
        ),
        "config_auto_calibrate_false_alarm_episode_limit": postprocess_data.get(
            "auto_calibrate_false_alarm_episode_limit",
            overrides.get("AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"),
        ),
        "config_auto_false_alarm_episode_limit_candidates": json.dumps(
            postprocess_data.get(
                "auto_false_alarm_episode_limit_candidates",
                overrides.get("AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES"),
            )
        ),
        "config_auto_event_recall_retention": postprocess_data.get(
            "auto_event_recall_retention",
            overrides.get("AUTO_EVENT_RECALL_RETENTION"),
        ),
        "config_rf_n_estimators": overrides.get(
            "RF_N_ESTIMATORS",
            DEFAULT_RF_N_ESTIMATORS,
        ),
        "config_svm_c": overrides.get("SVM_C", DEFAULT_SVM_C),
        "config_svm_gamma": overrides.get("SVM_GAMMA", DEFAULT_SVM_GAMMA),
        "config_knn_neighbors": overrides.get(
            "KNN_NEIGHBORS",
            DEFAULT_KNN_NEIGHBORS,
        ),
        "recommended_chunk_threshold": postprocess_data.get(
            "recommended_chunk_threshold"
        ),
        "selected_false_alarm_episode_limit": selection_threshold_info.get(
            "selected_false_alarm_episode_limit",
            postprocess_data.get(
                "selected_false_alarm_episode_limit",
                postprocess_data.get("target_false_alarm_episodes_per_min"),
            ),
        ),
        "final_selected_false_alarm_episode_limit": final_threshold_info.get(
            "selected_false_alarm_episode_limit",
            postprocess_data.get(
                "selected_false_alarm_episode_limit",
                postprocess_data.get("target_false_alarm_episodes_per_min"),
            ),
        ),
        "selected_threshold_constraint_satisfied": selection_threshold_info.get(
            "constraint_satisfied"
        ),
        "final_threshold_constraint_satisfied": final_threshold_info.get(
            "constraint_satisfied"
        ),
        "threshold_selection_metric": postprocess_data.get("threshold_selection_metric"),
        "bundle_path": postprocess_data.get("bundle_path"),
        "saved_bundles": postprocess_data.get("saved_bundles"),
        "run_output_dir": postprocess_data.get("run_output_dir"),
        "postprocess_json_path": postprocess_data.get("postprocess_json_path"),
        "log_path": str(log_path),
        "validation_precision": validation_metrics.get("precision"),
        "validation_event_recall": validation_metrics.get("event_recall"),
        "validation_macro_event_coverage": validation_metrics.get("macro_event_coverage"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1"),
        "validation_f2": validation_metrics.get("f2"),
        "validation_auc_pr": validation_metrics.get("auc_pr"),
        "validation_false_alarm_episodes_per_min": validation_metrics.get(
            "false_alarm_episodes_per_min"
        ),
        "validation_false_alarms_per_min": validation_metrics.get(
            "false_alarms_per_min"
        ),
        "validation_refit_precision": validation_metrics_refit.get("precision"),
        "validation_refit_event_recall": validation_metrics_refit.get("event_recall"),
        "validation_refit_macro_event_coverage": validation_metrics_refit.get(
            "macro_event_coverage"
        ),
        "validation_refit_recall": validation_metrics_refit.get("recall"),
        "validation_refit_f1": validation_metrics_refit.get("f1"),
        "validation_refit_f2": validation_metrics_refit.get("f2"),
        "validation_refit_auc_pr": validation_metrics_refit.get("auc_pr"),
        "validation_refit_false_alarm_episodes_per_min": validation_metrics_refit.get(
            "false_alarm_episodes_per_min"
        ),
        "validation_refit_false_alarms_per_min": validation_metrics_refit.get(
            "false_alarms_per_min"
        ),
        "test_precision": test_metrics.get("precision"),
        "test_event_recall": test_metrics.get("event_recall"),
        "test_macro_event_coverage": test_metrics.get("macro_event_coverage"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_f2": test_metrics.get("f2"),
        "test_auc_pr": test_metrics.get("auc_pr"),
        "test_false_alarm_episodes_per_min": test_metrics.get("false_alarm_episodes_per_min"),
        "test_false_alarms_per_min": test_metrics.get("false_alarms_per_min"),
    }


def stream_process_output(process, log_path):
    with open(log_path, "w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            if STREAM_TRAINING_LOGS:
                print(line, end="")


def find_new_postprocess_json(output_dir, run_name_prefix, previous_matches):
    current_matches = set(
        path.resolve()
        for path in output_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
    )
    new_matches = sorted(current_matches - previous_matches)
    if not new_matches:
        return None
    return Path(new_matches[-1])


def rank_rows(rows):
    successful_rows = [row for row in rows if row["status"] == "ok"]

    def build_rank_key(row):
        constraint_satisfied = normalize_optional_bool(
            row.get("selected_threshold_constraint_satisfied"),
            False,
        )
        if constraint_satisfied:
            return (
                1,
                safe_metric(row, PRIMARY_RANK_METRIC),
                safe_metric(row, SECONDARY_RANK_METRIC),
                safe_metric(row, TERTIARY_RANK_METRIC),
                safe_metric(row, "validation_recall"),
                safe_metric(row, QUATERNARY_RANK_METRIC),
                -safe_metric(row, FALSE_ALARM_RANK_METRIC, default=1e9),
            )

        return (
            0,
            safe_metric(row, PRIMARY_RANK_METRIC),
            -safe_metric(row, FALSE_ALARM_RANK_METRIC, default=1e9),
            safe_metric(row, SECONDARY_RANK_METRIC),
            safe_metric(row, TERTIARY_RANK_METRIC),
            safe_metric(row, "validation_recall"),
            safe_metric(row, QUATERNARY_RANK_METRIC),
            -safe_metric(
                row,
                "validation_false_alarms_per_min",
                default=1e9,
            ),
        )

    successful_rows.sort(
        key=build_rank_key,
        reverse=True,
    )
    return successful_rows


def save_summary_files(output_dir, rows, base_name="resumen_experimentos"):
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as json_handle:
        json.dump(rows, json_handle, indent=2)

    return csv_path, json_path


def get_best_result_for_group(ranked_rows, predicate):
    for row in ranked_rows:
        if predicate(row):
            return row
    return None


def select_refinement_parents(ranked_rows, max_parents):
    selected_rows = []
    seen_signatures = set()
    covered_base_experiment_ids = set()

    def try_add_row(row):
        signature = build_experiment_signature_from_row(row)
        if signature in seen_signatures:
            return False
        selected_rows.append(row)
        seen_signatures.add(signature)
        base_experiment_id = row.get("base_experiment_id")
        if base_experiment_id:
            covered_base_experiment_ids.add(base_experiment_id)
        return True

    if ranked_rows:
        try_add_row(ranked_rows[0])

    for row in ranked_rows:
        if len(selected_rows) >= max_parents:
            break
        base_experiment_id = row.get("base_experiment_id")
        if base_experiment_id and base_experiment_id not in covered_base_experiment_ids:
            try_add_row(row)

    for row in ranked_rows:
        if len(selected_rows) >= max_parents:
            break
        try_add_row(row)

    return selected_rows[:max_parents]


def build_refinement_mutations(parent_row):
    parent_overrides = derive_overrides_from_row(parent_row)
    current_chunk_length_s = float(parent_overrides["CHUNK_LENGTH_S"])
    current_use_class_weights = bool(parent_overrides["USE_CLASS_WEIGHTS"])
    current_use_augmentation = bool(parent_overrides["USE_DATA_AUGMENTATION"])
    current_augmentation_prob = float(
        parent_overrides.get("AUGMENTATION_APPLY_PROB", 0.5)
    )
    current_augmentation_copies = int(
        parent_overrides.get("AUGMENTATION_EXTRA_COPIES", 1)
    )
    current_use_pitch_shift = bool(
        parent_overrides.get("USE_PITCH_SHIFT_AUGMENTATION", True)
    )
    current_use_spectral_eq = bool(
        parent_overrides.get("USE_SPECTRAL_EQ_AUGMENTATION", True)
    )
    current_eq_probability = float(
        parent_overrides.get(
            "EQ_AUGMENTATION_PROB",
            FIXED_OVERRIDES["EQ_AUGMENTATION_PROB"],
        )
    )
    current_target_false_alarms = float(
        parent_overrides.get(
            "TARGET_FALSE_ALARM_EPISODES_PER_MIN",
            parent_overrides.get(
                "TARGET_FALSE_ALARMS_PER_MIN",
                FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
            ),
        )
    )

    target_chunk_length_s = (
        CORE_CHUNK_LENGTHS_S[1]
        if current_chunk_length_s < 0.75
        else CORE_CHUNK_LENGTHS_S[0]
    )
    mutations = [
        (
            f"chunk_{seconds_tag(target_chunk_length_s)}",
            {"CHUNK_LENGTH_S": target_chunk_length_s},
            "Cambia la duracion del chunk manteniendo el resto constante.",
        ),
        (
            "weights_off" if current_use_class_weights else "weights_on",
            {"USE_CLASS_WEIGHTS": not current_use_class_weights},
            "Invierte el uso de class weights para revisar si realmente ayudan.",
        ),
    ]

    if current_use_augmentation:
        for probability in AUGMENTATION_PROB_CANDIDATES:
            if not float_close(current_augmentation_prob, probability):
                mutations.append(
                    (
                        f"aug_p{prob_tag(probability)}",
                        {"AUGMENTATION_APPLY_PROB": probability},
                        "Ajusta la probabilidad de augmentacion.",
                    )
                )

        for copies in AUGMENTATION_EXTRA_COPIES_CANDIDATES:
            if current_augmentation_copies != copies:
                mutations.append(
                    (
                        f"aug_copies_{copies}",
                        {"AUGMENTATION_EXTRA_COPIES": copies},
                        "Ajusta el numero de copias augmentadas por chunk.",
                    )
                )

        mutations.append(
            (
                "pitch_off" if current_use_pitch_shift else "pitch_on",
                {"USE_PITCH_SHIFT_AUGMENTATION": not current_use_pitch_shift},
                "Invierte el pitch shift dentro de la augmentacion.",
            )
        )

        if current_use_spectral_eq:
            mutations.append(
                (
                    "eq_off",
                    {"USE_SPECTRAL_EQ_AUGMENTATION": False},
                    "Desactiva la EQ para aislar su efecto dentro de la augmentacion.",
                )
            )
            for probability in EQ_AUGMENTATION_PROB_CANDIDATES:
                if not float_close(current_eq_probability, probability):
                    mutations.append(
                        (
                            f"eq_p{prob_tag(probability)}",
                            {"EQ_AUGMENTATION_PROB": probability},
                            "Ajusta la frecuencia de la EQ espectral.",
                        )
                    )
        else:
            mutations.append(
                (
                    "eq_on_p10",
                    {
                        "USE_SPECTRAL_EQ_AUGMENTATION": True,
                        "EQ_AUGMENTATION_PROB": EQ_AUGMENTATION_PROB_CANDIDATES[0],
                    },
                    "Reactiva la EQ con una probabilidad suave.",
                )
            )
    else:
        mutations.extend(
            [
                (
                    "augment_p35",
                    {
                        "USE_DATA_AUGMENTATION": True,
                        "AUGMENTATION_APPLY_PROB": AUGMENTATION_PROB_CANDIDATES[0],
                        "AUGMENTATION_EXTRA_COPIES": 1,
                        "USE_PITCH_SHIFT_AUGMENTATION": False,
                        "USE_SPECTRAL_EQ_AUGMENTATION": True,
                        "EQ_AUGMENTATION_PROB": EQ_AUGMENTATION_PROB_CANDIDATES[0],
                    },
                    "Activa una augmentacion suave como primer salto de robustez.",
                ),
                (
                    "augment_p65",
                    {
                        "USE_DATA_AUGMENTATION": True,
                        "AUGMENTATION_APPLY_PROB": AUGMENTATION_PROB_CANDIDATES[1],
                        "AUGMENTATION_EXTRA_COPIES": 1,
                        "USE_PITCH_SHIFT_AUGMENTATION": True,
                        "USE_SPECTRAL_EQ_AUGMENTATION": True,
                        "EQ_AUGMENTATION_PROB": FIXED_OVERRIDES["EQ_AUGMENTATION_PROB"],
                    },
                    "Activa una augmentacion mas intensa para medir robustez.",
                ),
            ]
        )

    for target_false_alarms in TARGET_FALSE_ALARMS_CANDIDATES:
        if not float_close(current_target_false_alarms, target_false_alarms):
            mutations.append(
                (
                    f"fa_{prob_tag(target_false_alarms)}",
                    {
                        "TARGET_FALSE_ALARMS_PER_MIN": target_false_alarms,
                        "TARGET_FALSE_ALARM_EPISODES_PER_MIN": target_false_alarms,
                    },
                    "Cambia la restriccion de falsas alarmas por minuto usada al calibrar.",
                )
            )

    return mutations


def make_refinement_experiment(parent_row, round_number, label_suffix, extra_overrides, notes):
    base_overrides = derive_overrides_from_row(parent_row)
    merged_overrides = dict(base_overrides)
    merged_overrides.update(extra_overrides)

    chunk_length_s = float(
        merged_overrides.get("CHUNK_LENGTH_S", FIXED_OVERRIDES["CHUNK_LENGTH_S"])
    )
    overlap_s = float(
        merged_overrides.get("OVERLAP_S", FIXED_OVERRIDES["OVERLAP_S"])
    )
    if not normalize_optional_bool(merged_overrides.get("USE_OVERLAP"), False):
        overlap_s = FIXED_OVERRIDES["OVERLAP_S"]

    parent_round = int(parent_row.get("sweep_round") or max(1, round_number - 1))
    base_experiment_id = parent_row.get("base_experiment_id") or "derived"
    experiment_label = (
        f"{base_experiment_id}_r{round_number:02d}_"
        f"from_r{parent_round:02d}_{parent_row['experiment_id']}_{label_suffix}"
    )

    overrides = {
        "EXPERIMENT_LABEL": experiment_label,
        "EXPERIMENT_TIER": f"refine_r{round_number:02d}",
        "BASE_EXPERIMENT_ID": parent_row.get("base_experiment_id"),
        "BASE_EXPERIMENT_LABEL": parent_row.get("base_experiment_label"),
        "PARENT_EXPERIMENT_ID": parent_row.get("experiment_id"),
        "PARENT_EXPERIMENT_LABEL": parent_row.get("experiment_label"),
        "PARENT_SWEEP_ROUND": parent_row.get("sweep_round"),
        "EXPERIMENT_NOTES": (
            f"Refinamiento automatico desde {parent_row['experiment_label']} "
            f"(ronda {parent_round}). {notes}"
        ),
    }
    overrides.update(merged_overrides)
    overrides["CHUNK_LENGTH_S"] = chunk_length_s
    overrides["OVERLAP_S"] = overlap_s
    return overrides


def build_refinement_experiment_grid(parent_rows, seen_signatures, round_number):
    refinement_experiments = []
    round_signatures = set()

    for parent_row in parent_rows:
        children_added = 0
        for label_suffix, extra_overrides, notes in build_refinement_mutations(parent_row):
            experiment = make_refinement_experiment(
                parent_row=parent_row,
                round_number=round_number,
                label_suffix=label_suffix,
                extra_overrides=extra_overrides,
                notes=notes,
            )
            signature = build_experiment_signature_from_overrides(experiment)
            if signature in seen_signatures or signature in round_signatures:
                continue

            refinement_experiments.append(experiment)
            round_signatures.add(signature)
            children_added += 1

            if children_added >= REFINEMENT_MAX_CHILDREN_PER_PARENT:
                break

    return refinement_experiments


def save_round_report(round_dir, round_number, round_rows, ranked_rows, selected_parents, next_round_grid):
    report_path = round_dir / "reporte_ronda.md"
    successful_rows = [row for row in round_rows if row["status"] == "ok"]
    failed_rows = [row for row in round_rows if row["status"] != "ok"]
    report_lines = [
        f"# Reporte ronda {round_number:02d}",
        "",
        f"- Experimentos lanzados: {len(round_rows)}",
        f"- Experimentos exitosos: {len(successful_rows)}",
        f"- Experimentos con error o sin JSON: {len(failed_rows)}",
    ]

    if ranked_rows:
        best_row = ranked_rows[0]
        report_lines.extend(
            [
                f"- Mejor experimento: {format_row_compact(best_row)}",
                "",
                f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))}",
            ]
        )
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

        report_lines.extend(["", "## Mejores por grupo"])
        group_definitions = [
            ("Semillas iniciales", lambda row: row["experiment_tier"] == "seed"),
            (
                "Refinamiento actual",
                lambda row: str(row["experiment_tier"]).startswith("refine_"),
            ),
        ]
        for title, predicate in group_definitions:
            best_group_row = get_best_result_for_group(ranked_rows, predicate)
            if best_group_row is not None:
                report_lines.append(f"- {title}: {format_row_compact(best_group_row)}")
    else:
        report_lines.extend(["", "No hay experimentos exitosos en esta ronda."])

    if selected_parents:
        report_lines.extend(["", "## Padres elegidos para la siguiente ronda"])
        for rank, row in enumerate(selected_parents, start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    if next_round_grid is not None:
        report_lines.extend(
            [
                "",
                "## Siguiente ronda automatica",
                f"- Configuraciones nuevas generadas: {len(next_round_grid)}",
            ]
        )
        for experiment in next_round_grid[:ROUND_REPORT_TOP_N]:
            report_lines.append(
                f"- {experiment['EXPERIMENT_LABEL']} | "
                f"chunk={experiment.get('CHUNK_LENGTH_S')} s | "
                f"overlap={experiment.get('USE_OVERLAP')} | "
                f"weights={experiment.get('USE_CLASS_WEIGHTS')} | "
                f"augment={experiment.get('USE_DATA_AUGMENTATION')}"
            )

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def save_global_report(run_group_dir, all_rows, round_results):
    report_path = run_group_dir / "reporte_global.md"
    ranked_rows = rank_rows(all_rows)
    report_lines = [
        "# Reporte global del barrido iterativo",
        "",
        f"- Rondas ejecutadas: {len(round_results)}",
        f"- Experimentos totales: {len(all_rows)}",
        f"- Experimentos exitosos: {len(ranked_rows)}",
    ]

    report_lines.extend(["", "## Resumen por ronda"])
    for round_result in round_results:
        round_rows = round_result["rows"]
        round_ranked_rows = round_result["ranked_rows"]
        report_lines.append(
            f"- Ronda {round_result['round_number']:02d}: "
            f"{len(round_rows)} experimentos, "
            f"{len(round_ranked_rows)} exitosos, "
            f"dir={round_result['round_dir']}"
        )
        if round_ranked_rows:
            report_lines.append(f"  Mejor: {format_row_compact(round_ranked_rows[0])}")

    if ranked_rows:
        report_lines.extend(["", f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))} global"])
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def execute_round(experiment_grid, round_number, run_group_dir):
    if MAX_EXPERIMENTS is not None:
        experiment_grid = experiment_grid[:MAX_EXPERIMENTS]

    round_dir = run_group_dir / build_round_dir_name(round_number)
    configs_dir = round_dir / "configs"
    logs_dir = round_dir / "logs"
    artifacts_dir = round_dir / "artefactos"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"Ronda {round_number:02d}")
    print(f"Script de entrenamiento: {TRAINING_SCRIPT_PATH}")
    print(f"Directorio de la ronda: {round_dir}")
    print(f"Numero de experimentos a ejecutar: {len(experiment_grid)}")

    all_rows = []

    for experiment_number, experiment_overrides in enumerate(experiment_grid, start=1):
        experiment_id = f"exp_{experiment_number:03d}"
        experiment_label = experiment_overrides.get("EXPERIMENT_LABEL", experiment_id)
        run_name_prefix = f"{experiment_id}_{experiment_label}"

        applied_overrides = dict(FIXED_OVERRIDES)
        applied_overrides.update(experiment_overrides)

        experiment_output_dir = artifacts_dir / run_name_prefix
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        applied_overrides["RUN_OUTPUT_DIR"] = str(experiment_output_dir)
        applied_overrides["RUN_NAME_PREFIX"] = run_name_prefix

        config_path = configs_dir / f"{experiment_id}.json"
        log_path = logs_dir / f"{experiment_id}.log"

        with open(config_path, "w", encoding="utf-8") as config_handle:
            json.dump(applied_overrides, config_handle, indent=2)

        print("\n" + "=" * 80)
        print(f"Ejecutando {experiment_id} -> {experiment_label}")
        print(json.dumps(applied_overrides, indent=2))

        previous_matches = set(
            path.resolve()
            for path in experiment_output_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
        )

        process_env = os.environ.copy()
        process_env["SIREN_TRAD_TRAINING_CONFIG_PATH"] = str(config_path)

        process = subprocess.Popen(
            PYTHON_COMMAND + [str(TRAINING_SCRIPT_PATH)],
            cwd=str(SCRIPT_DIR),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stream_process_output(process, log_path)
        return_code = process.wait()

        postprocess_json_path = find_new_postprocess_json(
            experiment_output_dir,
            run_name_prefix,
            previous_matches,
        )

        if return_code != 0:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={
                    "postprocess_json_path": (
                        str(postprocess_json_path) if postprocess_json_path else None
                    )
                },
                status="failed",
                return_code=return_code,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(f"El experimento {experiment_id} ha fallado con codigo {return_code}.")

            if not CONTINUE_ON_ERROR:
                break
            continue

        if postprocess_json_path is None:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={},
                status="missing_json",
                return_code=return_code,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(
                f"El experimento {experiment_id} termino pero no genero JSON de resultados."
            )

            if not CONTINUE_ON_ERROR:
                break
            continue

        with open(postprocess_json_path, "r", encoding="utf-8") as result_handle:
            postprocess_data = json.load(result_handle)
        postprocess_data["postprocess_json_path"] = str(postprocess_json_path)

        row = build_summary_row(
            experiment_id=experiment_id,
            overrides=applied_overrides,
            postprocess_data=postprocess_data,
            status="ok",
            return_code=return_code,
            log_path=log_path,
            sweep_round=round_number,
            round_dir=round_dir,
        )
        all_rows.append(row)

    ranked_rows = rank_rows(all_rows)
    csv_path, json_path = save_summary_files(round_dir, all_rows)

    print("\n" + "=" * 80)
    print(f"Resumen CSV guardado en: {csv_path}")
    print(f"Resumen JSON guardado en: {json_path}")

    if ranked_rows:
        print("\nMejor experimento de la ronda:")
        print(format_row_compact(ranked_rows[0]))

        print(f"\nTop {min(ROUND_REPORT_TOP_N, len(ranked_rows))} de la ronda:")
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            print(format_row_compact(row, rank=rank))
    else:
        print("\nNo hay experimentos exitosos para comparar en esta ronda.")

    return {
        "round_number": round_number,
        "round_dir": round_dir,
        "experiment_grid": experiment_grid,
        "rows": all_rows,
        "ranked_rows": ranked_rows,
        "csv_path": csv_path,
        "json_path": json_path,
    }


if __name__ == "__main__":
    run_group_name = datetime.now().strftime(
        "barrido_clasif_trad_iterativo_%Y%m%d_%H%M%S"
    )
    run_group_dir = SCRIPT_DIR / run_group_name
    run_group_dir.mkdir(parents=True, exist_ok=True)

    current_experiment_grid = build_experiment_grid()
    all_rows_global = []
    round_results = []
    seen_signatures = set()

    print(f"Directorio raiz del barrido: {run_group_dir}")
    print(
        "Modo iterativo: "
        f"{AUTO_ITERATIVE_REFINEMENT} | max rondas: {MAX_SWEEP_ROUNDS} | "
        f"top padres: {REFINEMENT_TOP_K}"
    )

    for round_number in range(1, MAX_SWEEP_ROUNDS + 1):
        if not current_experiment_grid:
            print("\nNo hay configuraciones nuevas que ejecutar. Fin del barrido.")
            break

        round_result = execute_round(
            experiment_grid=current_experiment_grid,
            round_number=round_number,
            run_group_dir=run_group_dir,
        )
        round_results.append(round_result)
        all_rows_global.extend(round_result["rows"])

        for experiment in round_result["experiment_grid"]:
            seen_signatures.add(build_experiment_signature_from_overrides(experiment))

        selected_parents = []
        next_round_grid = None

        if AUTO_ITERATIVE_REFINEMENT and round_number < MAX_SWEEP_ROUNDS:
            selected_parents = select_refinement_parents(
                round_result["ranked_rows"],
                max_parents=REFINEMENT_TOP_K,
            )
            next_round_grid = build_refinement_experiment_grid(
                parent_rows=selected_parents,
                seen_signatures=seen_signatures,
                round_number=round_number + 1,
            )

        round_report_path = save_round_report(
            round_dir=round_result["round_dir"],
            round_number=round_number,
            round_rows=round_result["rows"],
            ranked_rows=round_result["ranked_rows"],
            selected_parents=selected_parents,
            next_round_grid=next_round_grid,
        )
        print(f"\nReporte de ronda guardado en: {round_report_path}")

        if not AUTO_ITERATIVE_REFINEMENT or round_number >= MAX_SWEEP_ROUNDS:
            break

        if not next_round_grid:
            print(
                "\nNo se han generado configuraciones nuevas para la siguiente ronda. "
                "Fin del barrido iterativo."
            )
            break

        print(
            f"\nArrancando automaticamente la ronda {round_number + 1:02d} "
            f"con {len(next_round_grid)} nuevas configuraciones..."
        )
        current_experiment_grid = next_round_grid

    global_csv_path, global_json_path = save_summary_files(
        run_group_dir,
        all_rows_global,
        base_name="resumen_global",
    )
    global_report_path = save_global_report(
        run_group_dir=run_group_dir,
        all_rows=all_rows_global,
        round_results=round_results,
    )

    print("\n" + "#" * 80)
    print(f"Resumen global CSV guardado en: {global_csv_path}")
    print(f"Resumen global JSON guardado en: {global_json_path}")
    print(f"Reporte global guardado en: {global_report_path}")

    global_ranked_rows = rank_rows(all_rows_global)
    if global_ranked_rows:
        print("\nMejor experimento global:")
        print(format_row_compact(global_ranked_rows[0]))
    else:
        print("\nNo hay experimentos exitosos en el barrido iterativo.")
