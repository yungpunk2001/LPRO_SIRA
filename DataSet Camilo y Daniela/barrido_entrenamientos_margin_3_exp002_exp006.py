import csv
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Barrido dirigido para `entrenar_modelo_margin_3.py`.
#
# Objetivo:
# 1. Partir de las dos semillas mas relevantes del barrido anterior:
#    - exp002: harmonic_full + balanceo por chunks
#    - exp006: full + sin balanceo por chunks
# 2. Comparar de forma limpia:
#    - ventanas de 0.5 s frente a 1.0 s
#    - frontend lineal frente a log-mel
# 3. Espejar la matriz anterior con data augmentation para medir si aqui ayuda.
# 4. Recuperar la familia de exp008 "combinacion_prometedora" con la misma
#    matriz temporal/espectral para compararla contra los seeds anteriores.
# 5. Mantener unas pocas variantes exploratorias separadas del bloque central.
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT_PATH = SCRIPT_DIR / "entrenar_modelo_margin_3.py"
RUNS_ROOT_DIR = SCRIPT_DIR / "Modelos Atlas"


# ---------------------------------------------------------------------------
# Configuracion general del barrido
# ---------------------------------------------------------------------------
PYTHON_COMMAND = [os.environ.get("PYTHON_BIN") or sys.executable or "python3"]
CONTINUE_ON_ERROR = True
STREAM_TRAINING_LOGS = True
MAX_EXPERIMENTS = None

PRIMARY_RANK_METRIC = "validation_event_recall"
FALSE_ALARM_RANK_METRIC = "validation_false_alarm_episodes_per_min"
SECONDARY_RANK_METRIC = "validation_macro_event_coverage"
TERTIARY_RANK_METRIC = "validation_f2"
QUATERNARY_RANK_METRIC = "validation_auc_pr"

# Si es True, al terminar una ronda se genera un reporte y, si hay variantes
# nuevas no probadas, se lanza otra ronda automaticamente a partir de las
# mejores configuraciones encontradas.
AUTO_ITERATIVE_REFINEMENT = True

# Numero maximo total de rondas. 1 = solo barrido inicial; 2 = inicial + una
# ronda de refinamiento; 3 = dos refinamientos, etc.
MAX_SWEEP_ROUNDS = 2

# Cuantos padres se eligen tras cada ronda para generar la siguiente.
REFINEMENT_TOP_K = 4

# Numero maximo de hijos derivados de cada configuracion seleccionada.
REFINEMENT_MAX_CHILDREN_PER_PARENT = 6

# Cuantos experimentos mostrar en cada reporte de ronda.
ROUND_REPORT_TOP_N = 8

# Epocas por defecto del script de entrenamiento. Se replica aqui para dejar
# trazabilidad explicita en los CSV/JSON del barrido.
DEFAULT_TRAINING_EPOCHS = 50

# Mini-bloque opcional para relanzar solo los mejores experimentos al final del
# barrido con mas presupuesto de entrenamiento, sin contaminar la matriz
# principal ni el refinamiento automatico.
FINAL_RERUN_ENABLED = True
FINAL_RERUN_TOP_K = 3
FINAL_RERUN_EPOCHS = 80
FINAL_RERUN_DIR_NAME = "rerun_final"
FINAL_RERUN_PREFER_FEASIBLE = True
FINAL_RERUN_REQUIRE_EPOCH_BUDGET_EXHAUSTION = True


# ---------------------------------------------------------------------------
# Constantes compartidas para generar comparaciones coherentes.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
HOP_LENGTH = 512
DEFAULT_MEL_BINS = 128
CORE_CHUNK_LENGTHS_S = [0.5, 1.0]
CORE_FRONTENDS = ["linear_stft", "log_mel"]
CORE_SEED_KEYS = ["exp002", "exp006"]
AUGMENTATION_MIRROR_SEED_KEYS = ["exp002", "exp006"]
EXP008_SEED_KEYS = ["exp008"]


# ---------------------------------------------------------------------------
# Overrides fijos aplicados a todos los experimentos del barrido.
# ---------------------------------------------------------------------------
FIXED_OVERRIDES = {
    "RANDOM_SEED": 42,
    "SHOW_TRAINING_PLOTS": False,
    "USE_THRESHOLD_ANALYSIS": True,
    "SAVE_POSTPROCESSING_CONFIG": True,
    "SAVE_CONFUSION_REPORT": False,
    "USE_CLASS_WEIGHTS": True,
    "TRAIN_CHUNK_BATCH_SIZE": 64,
    "OVERLAP_S": 0.0,
    "TARGET_FALSE_ALARMS_PER_MIN": 1.0,
    "TARGET_FALSE_ALARM_EPISODES_PER_MIN": 1.0,
    "AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT": True,
    "AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES": [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    "AUTO_EVENT_RECALL_RETENTION": 0.95,
}


# ---------------------------------------------------------------------------
# Semillas heredadas del barrido anterior.
# ---------------------------------------------------------------------------
BASE_EXPERIMENT_SEEDS = {
    "exp002": {
        "base_experiment_label": "solo_balanceo_chunks",
        "base_summary": "Seed exp002: harmonic_full con balanceo por chunks.",
        "overrides": {
            "FEATURE_REPRESENTATION": "harmonic_full",
            "SPECTROGRAM_NORMALIZATION": "minmax",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": True,
            "USE_DATA_AUGMENTATION": False,
        },
    },
    "exp006": {
        "base_experiment_label": "solo_espectrograma_completo",
        "base_summary": "Seed exp006: espectrograma completo sin balanceo por chunks.",
        "overrides": {
            "FEATURE_REPRESENTATION": "full",
            "SPECTROGRAM_NORMALIZATION": "minmax",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": False,
            "USE_DATA_AUGMENTATION": False,
        },
    },
    "exp008": {
        "base_experiment_label": "combinacion_prometedora",
        "base_summary": (
            "Seed exp008: harmonic_full con normalizacion frequency, "
            "balanceo por chunks y data augmentation."
        ),
        "overrides": {
            "FEATURE_REPRESENTATION": "harmonic_full",
            "SPECTROGRAM_NORMALIZATION": "frequency",
            "CONV_FILTERS": [16, 32, 64],
            "DENSE_UNITS": 32,
            "USE_BALANCED_CHUNK_BATCHES": True,
            "USE_DATA_AUGMENTATION": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Variantes extra fuera de las matrices principales.
# ---------------------------------------------------------------------------
EXPLORATORY_VARIANTS = [
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "linear_stft",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Transfiere al mejor seed el balanceo por chunks que si ayudo en exp002.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 1.0,
        "frontend": "linear_stft",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Comprueba si 1.0 s y balanceo por chunks se refuerzan mutuamente.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "linear_stft",
        "label_suffix": "frequency_norm",
        "extra_overrides": {
            "SPECTROGRAM_NORMALIZATION": "frequency",
        },
        "notes": "Reevalua la normalizacion por frecuencia sobre la familia full.",
    },
    {
        "base_seed_key": "exp006",
        "chunk_length_s": 0.5,
        "frontend": "log_mel",
        "label_suffix": "balanced",
        "extra_overrides": {
            "USE_BALANCED_CHUNK_BATCHES": True,
        },
        "notes": "Comprueba si log-mel gana recall cuando se equilibra por chunks.",
    },
]


def seconds_tag(seconds):
    """Convierte 0.5 -> 0p5s para etiquetas de fichero y CSV."""
    return f"{seconds:.1f}".replace(".", "p") + "s"


def frontend_tag(frontend):
    """Nombre corto del frontend para etiquetas legibles."""
    if frontend == "linear_stft":
        return "linear"
    if frontend == "log_mel":
        return "logmel"
    raise ValueError("Frontend no soportado.")


def compute_time_frames(chunk_length_s, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Ajusta TIME_FRAMES a la duracion real del chunk para que 1.0 s no recorte
    la mitad del contexto temporal.
    """
    chunk_samples = int(round(chunk_length_s * sample_rate))
    return max(1, int(math.ceil(chunk_samples / hop_length)) + 1)


def build_temporal_overrides(chunk_length_s):
    """Genera overrides temporales coherentes para cada tamano de ventana."""
    return {
        "CHUNK_LENGTH_S": chunk_length_s,
        "OVERLAP_S": 0.0,
        "TIME_FRAMES": compute_time_frames(chunk_length_s),
    }


def build_frontend_overrides(frontend):
    """Devuelve los overrides necesarios para cada frontend espectral."""
    if frontend == "linear_stft":
        return {
            "SPECTRAL_FRONTEND": "linear_stft",
        }

    if frontend == "log_mel":
        return {
            "SPECTRAL_FRONTEND": "log_mel",
            "MEL_BINS": DEFAULT_MEL_BINS,
        }

    raise ValueError("SPECTRAL_FRONTEND debe ser 'linear_stft' o 'log_mel'.")


def build_experiment_notes(base_seed_key, frontend, notes):
    """Une una descripcion corta del seed y del cambio que se esta probando."""
    seed_summary = BASE_EXPERIMENT_SEEDS[base_seed_key]["base_summary"]
    frontend_note = (
        "Frontend lineal con la representacion heredada del seed."
        if frontend == "linear_stft"
        else "Frontend log-mel; FEATURE_REPRESENTATION queda solo como trazabilidad."
    )
    note_parts = [seed_summary, frontend_note]
    if notes:
        note_parts.append(notes)
    return " ".join(note_parts)


def make_experiment(
    base_seed_key,
    chunk_length_s,
    frontend,
    tier,
    notes=None,
    label_suffix=None,
    extra_overrides=None,
):
    """Construye una configuracion de experimento autocontenida y trazable."""
    base_seed = BASE_EXPERIMENT_SEEDS[base_seed_key]
    chunk_tag = seconds_tag(chunk_length_s)
    short_frontend = frontend_tag(frontend)
    label_parts = [base_seed_key, chunk_tag, short_frontend, tier]
    if label_suffix:
        label_parts.append(label_suffix)
    experiment_label = "_".join(label_parts)

    overrides = {
        "EXPERIMENT_LABEL": experiment_label,
        "EXPERIMENT_TIER": tier,
        "BASE_EXPERIMENT_ID": base_seed_key,
        "BASE_EXPERIMENT_LABEL": base_seed["base_experiment_label"],
        "COMPARISON_WINDOW_S": chunk_length_s,
        "COMPARISON_WINDOW_TAG": chunk_tag,
        "COMPARISON_FRONTEND": frontend,
        "COMPARISON_FRONTEND_TAG": short_frontend,
        "EXPERIMENT_NOTES": build_experiment_notes(base_seed_key, frontend, notes),
    }

    overrides.update(base_seed["overrides"])
    overrides.update(build_temporal_overrides(chunk_length_s))
    overrides.update(build_frontend_overrides(frontend))

    if extra_overrides:
        overrides.update(extra_overrides)

    return overrides


def build_family_matrix(seed_keys, tier, notes, extra_overrides=None, label_suffix=None):
    """Genera una matriz completa seed x ventana x frontend con una etiqueta comun."""
    experiments = []

    for base_seed_key in seed_keys:
        for chunk_length_s in CORE_CHUNK_LENGTHS_S:
            for frontend in CORE_FRONTENDS:
                experiments.append(
                    make_experiment(
                        base_seed_key=base_seed_key,
                        chunk_length_s=chunk_length_s,
                        frontend=frontend,
                        tier=tier,
                        notes=notes,
                        label_suffix=label_suffix,
                        extra_overrides=extra_overrides,
                    )
                )
    return experiments


def build_experiment_grid():
    """Genera las matrices principales y las variantes exploratorias adicionales."""
    experiments = []

    experiments.extend(
        build_family_matrix(
            seed_keys=CORE_SEED_KEYS,
            tier="core",
            notes="Parte de la matriz principal para comparar ventana y frontend.",
        )
    )

    experiments.extend(
        build_family_matrix(
            seed_keys=AUGMENTATION_MIRROR_SEED_KEYS,
            tier="augment",
            notes=(
                "Espejo directo del bloque core con data augmentation activada "
                "para aislar su impacto."
            ),
            extra_overrides={"USE_DATA_AUGMENTATION": True},
        )
    )

    experiments.extend(
        build_family_matrix(
            seed_keys=EXP008_SEED_KEYS,
            tier="exp008",
            notes=(
                "Familia derivada de exp008 combinacion_prometedora para probar "
                "ventana y frontend sobre esa configuracion base."
            ),
        )
    )

    for variant in EXPLORATORY_VARIANTS:
        experiments.append(
            make_experiment(
                base_seed_key=variant["base_seed_key"],
                chunk_length_s=variant["chunk_length_s"],
                frontend=variant["frontend"],
                tier="explore",
                notes=variant["notes"],
                label_suffix=variant.get("label_suffix"),
                extra_overrides=variant.get("extra_overrides"),
            )
        )

    return experiments


def parse_conv_filters(value):
    """Recupera la lista de filtros desde el resumen compacto."""
    if isinstance(value, str):
        return list(json.loads(value))
    if value is None:
        return [16, 32, 64]
    return list(value)


def build_experiment_signature_from_overrides(overrides):
    """Devuelve una firma canonica para evitar repetir la misma configuracion."""
    frontend = overrides.get("SPECTRAL_FRONTEND", "linear_stft")
    use_augmentation = bool(overrides.get("USE_DATA_AUGMENTATION", False))
    use_spectral_eq = bool(overrides.get("USE_SPECTRAL_EQ_AUGMENTATION", True))
    auto_false_alarm_episode_limit_candidates = overrides.get(
        "AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES",
        FIXED_OVERRIDES["AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES"],
    )

    return (
        frontend,
        overrides.get("FEATURE_REPRESENTATION"),
        overrides.get("SPECTROGRAM_NORMALIZATION"),
        tuple(overrides.get("CONV_FILTERS", [])),
        int(overrides.get("DENSE_UNITS", 32)),
        round(float(overrides.get("CHUNK_LENGTH_S", 0.5)), 4),
        round(float(overrides.get("OVERLAP_S", 0.0)), 4),
        int(overrides.get("TIME_FRAMES", compute_time_frames(overrides.get("CHUNK_LENGTH_S", 0.5)))),
        int(overrides.get("MEL_BINS", DEFAULT_MEL_BINS)) if frontend == "log_mel" else None,
        bool(overrides.get("USE_BALANCED_CHUNK_BATCHES", False)),
        bool(overrides.get("USE_CLASS_WEIGHTS", True)),
        use_augmentation,
        (
            round(float(overrides.get("AUGMENTATION_APPLY_PROB", 0.5)), 4)
            if use_augmentation
            else None
        ),
        use_spectral_eq if use_augmentation else None,
        (
            round(float(overrides.get("EQ_AUGMENTATION_PROB", 0.20)), 4)
            if use_augmentation and use_spectral_eq
            else None
        ),
        round(
            float(
                overrides.get(
                    "TARGET_FALSE_ALARM_EPISODES_PER_MIN",
                    overrides.get(
                        "TARGET_FALSE_ALARMS_PER_MIN",
                        FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"],
                    ),
                )
            ),
            4,
        ),
        bool(
            overrides.get(
                "AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT",
                FIXED_OVERRIDES["AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"],
            )
        ),
        tuple(
            round(float(limit), 4)
            for limit in auto_false_alarm_episode_limit_candidates
        ),
        round(
            float(
                overrides.get(
                    "AUTO_EVENT_RECALL_RETENTION",
                    FIXED_OVERRIDES["AUTO_EVENT_RECALL_RETENTION"],
                )
            ),
            4,
        ),
    )


def derive_overrides_from_row(row):
    """Reconstruye una configuracion reutilizable a partir de la fila resumen."""
    chunk_length_s = float(row.get("config_chunk_length_s") or row.get("comparison_window_s") or 0.5)
    frontend = row.get("config_spectral_frontend") or row.get("comparison_frontend") or "linear_stft"
    mel_bins = row.get("config_mel_bins")
    augmentation_apply_probability = row.get("config_augmentation_apply_probability")
    eq_augmentation_probability = row.get("config_eq_augmentation_probability")
    use_spectral_eq_augmentation = row.get("config_use_spectral_eq_augmentation")

    overrides = {
        "FEATURE_REPRESENTATION": row.get("config_feature_representation_requested"),
        "SPECTROGRAM_NORMALIZATION": row.get("config_spectrogram_normalization"),
        "CONV_FILTERS": parse_conv_filters(row.get("config_conv_filters")),
        "DENSE_UNITS": int(row.get("config_dense_units") or 32),
        "EPOCHS": int(row.get("config_epochs") or DEFAULT_TRAINING_EPOCHS),
        "USE_BALANCED_CHUNK_BATCHES": bool(row.get("config_use_balanced_chunk_batches")),
        "USE_DATA_AUGMENTATION": bool(row.get("config_use_data_augmentation")),
        "USE_CLASS_WEIGHTS": bool(
            FIXED_OVERRIDES.get("USE_CLASS_WEIGHTS", True)
            if row.get("config_use_class_weights") is None
            else row.get("config_use_class_weights")
        ),
        "TRAIN_CHUNK_BATCH_SIZE": int(
            row.get("config_train_chunk_batch_size") or FIXED_OVERRIDES["TRAIN_CHUNK_BATCH_SIZE"]
        ),
        "TARGET_FALSE_ALARM_EPISODES_PER_MIN": float(
            row.get("requested_false_alarm_episode_limit")
            or row.get("selected_false_alarm_episode_limit")
            or FIXED_OVERRIDES["TARGET_FALSE_ALARMS_PER_MIN"]
        ),
        "AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT": bool(
            FIXED_OVERRIDES["AUTO_CALIBRATE_FALSE_ALARM_EPISODE_LIMIT"]
            if row.get("config_auto_calibrate_false_alarm_episode_limit") is None
            else row.get("config_auto_calibrate_false_alarm_episode_limit")
        ),
        "AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES": (
            json.loads(row["config_auto_false_alarm_episode_limit_candidates"])
            if row.get("config_auto_false_alarm_episode_limit_candidates")
            else FIXED_OVERRIDES["AUTO_FALSE_ALARM_EPISODE_LIMIT_CANDIDATES"]
        ),
        "AUTO_EVENT_RECALL_RETENTION": float(
            row.get("config_auto_event_recall_retention")
            or FIXED_OVERRIDES["AUTO_EVENT_RECALL_RETENTION"]
        ),
        "SPECTRAL_FRONTEND": frontend,
    }
    overrides.update(build_temporal_overrides(chunk_length_s))

    if frontend == "log_mel":
        overrides["MEL_BINS"] = int(mel_bins or DEFAULT_MEL_BINS)

    if overrides["USE_DATA_AUGMENTATION"]:
        overrides["AUGMENTATION_APPLY_PROB"] = float(
            0.5 if augmentation_apply_probability is None else augmentation_apply_probability
        )
        overrides["USE_SPECTRAL_EQ_AUGMENTATION"] = bool(
            True if use_spectral_eq_augmentation is None else use_spectral_eq_augmentation
        )
        if overrides["USE_SPECTRAL_EQ_AUGMENTATION"]:
            overrides["EQ_AUGMENTATION_PROB"] = float(
                0.20 if eq_augmentation_probability is None else eq_augmentation_probability
            )

    return overrides


def build_experiment_signature_from_row(row):
    """Firma canonica a partir de una fila de resultados ya evaluada."""
    return build_experiment_signature_from_overrides(derive_overrides_from_row(row))


def build_round_dir_name(round_number):
    """Nombre de directorio estable para cada ronda."""
    return f"ronda_{round_number:02d}"


def format_row_compact(row, rank=None):
    """Representacion corta y legible de una fila para consola y reportes."""
    rank_prefix = f"{rank}. " if rank is not None else ""
    return (
        f"{rank_prefix}{row['experiment_id']} ({row['experiment_label']}) | "
        f"round={row.get('sweep_round')} | "
        f"tier={row.get('experiment_tier')} | "
        f"base={row.get('base_experiment_id')} | "
        f"frontend={row.get('config_spectral_frontend')} | "
        f"window={row.get('config_chunk_length_s')} s | "
        f"epochs={row.get('config_epochs')} | "
        f"feasible={safe_bool(row, 'selected_threshold_constraint_satisfied', default=False)} | "
        f"norm={row.get('config_spectrogram_normalization')} | "
        f"balanced={row.get('config_use_balanced_chunk_batches')} | "
        f"augment={row.get('config_use_data_augmentation')} | "
        f"val_event_recall={safe_metric(row, 'validation_event_recall', default=0.0):.4f} | "
        f"val_f2={safe_metric(row, 'validation_f2', default=0.0):.4f} | "
        f"val_recall={safe_metric(row, 'validation_recall', default=0.0):.4f} | "
        f"val_auc_pr={safe_metric(row, 'validation_auc_pr', default=0.0):.4f} | "
        f"fa_epi/min={safe_metric(row, 'validation_false_alarm_episodes_per_min', default=0.0):.2f} | "
        f"fa_chunk/min={safe_metric(row, 'validation_false_alarms_per_min', default=0.0):.2f} | "
        f"fa_limit={safe_metric(row, 'selected_false_alarm_episode_limit', default=0.0):.2f}"
    )


def safe_metric(row, key, default=-1.0):
    """Devuelve una metrica numerica o un valor por defecto si falta."""
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def safe_bool(row, key, default=False):
    """Devuelve un booleano normalizado a partir de una fila resumen."""
    value = row.get(key)
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


def get_effective_frontend(overrides, postprocess_data):
    """Obtiene el frontend real del experimento aunque falten datos en el JSON."""
    return postprocess_data.get(
        "spectral_frontend",
        overrides.get("SPECTRAL_FRONTEND", "linear_stft"),
    )


def get_effective_feature_representation(overrides, postprocess_data):
    """Indica la representacion efectiva usada por la CNN en cada frontend."""
    frontend = get_effective_frontend(overrides, postprocess_data)
    if frontend == "log_mel":
        return "n/a_log_mel"
    return postprocess_data.get(
        "feature_representation",
        overrides.get("FEATURE_REPRESENTATION"),
    )


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
    """Compacta los resultados de un experimento en una sola fila."""
    validation_metrics = postprocess_data.get("validation_metrics") or {}
    test_metrics = postprocess_data.get("test_metrics") or {}
    threshold_calibration_info = postprocess_data.get("threshold_calibration_info") or {}

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
        "comparison_window_s": overrides.get("COMPARISON_WINDOW_S"),
        "comparison_window_tag": overrides.get("COMPARISON_WINDOW_TAG"),
        "comparison_frontend": overrides.get("COMPARISON_FRONTEND"),
        "comparison_frontend_tag": overrides.get("COMPARISON_FRONTEND_TAG"),
        "experiment_notes": overrides.get("EXPERIMENT_NOTES"),
        "status": status,
        "return_code": return_code,
        "config_chunk_length_s": postprocess_data.get(
            "chunk_length_s",
            overrides.get("CHUNK_LENGTH_S"),
        ),
        "config_overlap_s": postprocess_data.get(
            "overlap_s",
            overrides.get("OVERLAP_S"),
        ),
        "config_decision_step_s": postprocess_data.get("decision_step_s"),
        "config_spectral_frontend": get_effective_frontend(overrides, postprocess_data),
        "config_feature_representation_requested": overrides.get("FEATURE_REPRESENTATION"),
        "config_feature_representation_effective": get_effective_feature_representation(
            overrides,
            postprocess_data,
        ),
        "config_spectrogram_normalization": postprocess_data.get(
            "spectrogram_normalization",
            overrides.get("SPECTROGRAM_NORMALIZATION"),
        ),
        "config_conv_filters": json.dumps(
            postprocess_data.get("conv_filters", overrides.get("CONV_FILTERS"))
        ),
        "config_dense_units": postprocess_data.get(
            "dense_units",
            overrides.get("DENSE_UNITS"),
        ),
        "config_epochs": overrides.get("EPOCHS", DEFAULT_TRAINING_EPOCHS),
        "config_time_frames": postprocess_data.get(
            "time_frames",
            overrides.get("TIME_FRAMES"),
        ),
        "config_padded_chunk_samples": postprocess_data.get("padded_chunk_samples"),
        "config_mel_bins": postprocess_data.get(
            "mel_bins",
            overrides.get("MEL_BINS"),
        ),
        "config_use_class_weights": postprocess_data.get(
            "use_class_weights",
            overrides.get("USE_CLASS_WEIGHTS"),
        ),
        "config_effective_use_class_weights": postprocess_data.get(
            "effective_use_class_weights"
        ),
        "config_use_balanced_chunk_batches": postprocess_data.get(
            "use_balanced_chunk_batches",
            overrides.get("USE_BALANCED_CHUNK_BATCHES"),
        ),
        "config_train_chunk_batch_size": postprocess_data.get(
            "train_chunk_batch_size",
            overrides.get("TRAIN_CHUNK_BATCH_SIZE"),
        ),
        "config_use_data_augmentation": postprocess_data.get(
            "use_data_augmentation",
            overrides.get("USE_DATA_AUGMENTATION"),
        ),
        "config_augmentation_apply_probability": postprocess_data.get(
            "augmentation_apply_probability"
        ),
        "config_use_spectral_eq_augmentation": postprocess_data.get(
            "use_spectral_eq_augmentation"
        ),
        "config_eq_augmentation_probability": postprocess_data.get(
            "eq_augmentation_probability"
        ),
        "config_eq_effective_conditional_probability": postprocess_data.get(
            "eq_effective_conditional_probability"
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
        "recommended_chunk_threshold": postprocess_data.get("recommended_chunk_threshold"),
        "requested_false_alarm_episode_limit": postprocess_data.get(
            "target_false_alarm_episodes_per_min"
        ),
        "selected_false_alarm_episode_limit": postprocess_data.get(
            "selected_false_alarm_episode_limit",
            postprocess_data.get("target_false_alarm_episodes_per_min"),
        ),
        "selected_threshold_constraint_satisfied": threshold_calibration_info.get(
            "selected_threshold_constraint_satisfied"
        ),
        "threshold_selection_metric": postprocess_data.get("threshold_selection_metric"),
        "model_path": postprocess_data.get("model_path"),
        "postprocess_json_path": postprocess_data.get("postprocess_json_path"),
        "log_path": str(log_path),
        "validation_event_recall": validation_metrics.get("event_recall"),
        "validation_macro_event_coverage": validation_metrics.get("macro_event_coverage"),
        "validation_precision": validation_metrics.get("precision"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1"),
        "validation_f2": validation_metrics.get("f2"),
        "validation_auc_pr": validation_metrics.get("auc_pr"),
        "validation_false_alarm_episodes_per_min": validation_metrics.get(
            "false_alarm_episodes_per_min"
        ),
        "validation_false_alarms_per_min": validation_metrics.get("false_alarms_per_min"),
        "validation_detected_positive_event_count": validation_metrics.get(
            "detected_positive_event_count"
        ),
        "validation_total_positive_event_count": validation_metrics.get(
            "total_positive_event_count"
        ),
        "test_event_recall": test_metrics.get("event_recall"),
        "test_macro_event_coverage": test_metrics.get("macro_event_coverage"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_f2": test_metrics.get("f2"),
        "test_auc_pr": test_metrics.get("auc_pr"),
        "test_false_alarm_episodes_per_min": test_metrics.get("false_alarm_episodes_per_min"),
        "test_false_alarms_per_min": test_metrics.get("false_alarms_per_min"),
    }


def stream_process_output(process, log_path):
    """Guarda la salida de un entrenamiento en log y opcionalmente la muestra."""
    with open(log_path, "w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            if STREAM_TRAINING_LOGS:
                print(line, end="")


def find_new_postprocess_json(artifacts_dir, run_name_prefix, previous_matches):
    """Localiza el JSON generado por el experimento recien terminado."""
    current_matches = set(
        path.resolve()
        for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
    )
    new_matches = sorted(current_matches - previous_matches)

    if not new_matches:
        return None

    return Path(new_matches[-1])


def find_existing_postprocess_json(artifacts_dir, run_name_prefix):
    """Devuelve el JSON de postprocesado existente mas reciente."""
    matches = list(artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def parse_experiment_number(value):
    """Acepta '5', 5 o 'exp_005' y devuelve el numero entero."""
    if value is None:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(?:exp_)?0*(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Identificador de experimento no valido: {value!r}"
        )
    experiment_number = int(match.group(1))
    if experiment_number < 1:
        raise argparse.ArgumentTypeError("El experimento debe ser >= 1.")
    return experiment_number


def iter_experiment_items(experiment_grid, start_exp=None, end_exp=None):
    """Itera manteniendo los numeros originales exp_001, exp_002, ..."""
    if MAX_EXPERIMENTS is not None:
        experiment_grid = experiment_grid[:MAX_EXPERIMENTS]

    for experiment_number, experiment_overrides in enumerate(experiment_grid, start=1):
        if start_exp is not None and experiment_number < start_exp:
            continue
        if end_exp is not None and experiment_number > end_exp:
            continue
        yield experiment_number, experiment_overrides


def build_experiment_context(
    experiment_number,
    experiment_overrides,
    configs_dir,
    logs_dir,
    artifacts_dir,
):
    """Construye rutas y overrides finales para un experimento numerado."""
    experiment_id = f"exp_{experiment_number:03d}"
    experiment_label = experiment_overrides.get("EXPERIMENT_LABEL", experiment_id)
    run_name_prefix = f"{experiment_id}_{experiment_label}"

    applied_overrides = dict(FIXED_OVERRIDES)
    applied_overrides.update(experiment_overrides)
    applied_overrides["RUN_OUTPUT_DIR"] = str(artifacts_dir)
    applied_overrides["RUN_NAME_PREFIX"] = run_name_prefix

    return {
        "experiment_id": experiment_id,
        "experiment_label": experiment_label,
        "run_name_prefix": run_name_prefix,
        "applied_overrides": applied_overrides,
        "config_path": configs_dir / f"{experiment_id}.json",
        "log_path": logs_dir / f"{experiment_id}.log",
    }


def save_experiment_config(config_path, applied_overrides):
    """Guarda un JSON de configuracion de entrenamiento."""
    with open(config_path, "w", encoding="utf-8") as config_handle:
        json.dump(applied_overrides, config_handle, indent=2)


def generate_round_configs(
    experiment_grid,
    round_number,
    run_group_dir,
    start_exp=None,
    end_exp=None,
    skip_completed=False,
    config_list_name="configs_restantes_cnn.txt",
):
    """Genera JSONs de una ronda y un listado para lanzarlos como job array."""
    round_dir = run_group_dir / build_round_dir_name(round_number)
    configs_dir = round_dir / "configs"
    logs_dir = round_dir / "logs"
    artifacts_dir = round_dir / "artefactos"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    experiment_items = list(
        iter_experiment_items(
            experiment_grid,
            start_exp=start_exp,
            end_exp=end_exp,
        )
    )
    config_paths = []

    print("\n" + "#" * 80)
    print(f"Generando configs de la ronda {round_number:02d}")
    print(f"Directorio de la ronda: {round_dir}")
    print(f"Experimentos candidatos: {len(experiment_items)}")

    for experiment_number, experiment_overrides in experiment_items:
        context = build_experiment_context(
            experiment_number,
            experiment_overrides,
            configs_dir,
            logs_dir,
            artifacts_dir,
        )
        save_experiment_config(
            context["config_path"],
            context["applied_overrides"],
        )

        existing_postprocess_json = find_existing_postprocess_json(
            artifacts_dir,
            context["run_name_prefix"],
        )
        if skip_completed and existing_postprocess_json is not None:
            print(
                "Omitiendo "
                f"{context['experiment_id']} -> {context['experiment_label']} "
                f"(ya completado: {existing_postprocess_json.name})"
            )
            continue

        config_paths.append(context["config_path"])
        print(
            "Config lista: "
            f"{context['experiment_id']} -> {context['experiment_label']}"
        )

    config_list_path = round_dir / config_list_name
    with open(config_list_path, "w", encoding="utf-8") as list_handle:
        for config_path in config_paths:
            list_handle.write(str(config_path) + "\n")

    print("\n" + "=" * 80)
    print(f"Listado para job array guardado en: {config_list_path}")
    print(f"Configs incluidas en el array: {len(config_paths)}")
    return config_list_path, config_paths


def parse_round_number_from_dir(round_dir):
    match = re.fullmatch(r"ronda_(\d+)", round_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def list_existing_round_numbers(run_group_dir):
    round_numbers = []
    for round_dir in sorted(run_group_dir.glob("ronda_*")):
        if not round_dir.is_dir():
            continue
        round_number = parse_round_number_from_dir(round_dir)
        if round_number is not None:
            round_numbers.append(round_number)
    return round_numbers


def parse_experiment_number_from_id(experiment_id):
    match = re.fullmatch(r"exp_(\d+)", experiment_id)
    if match is None:
        return None
    return int(match.group(1))


def summarize_round_from_configs(
    run_group_dir,
    round_number,
    start_exp=None,
    end_exp=None,
):
    """Reconstruye una ronda ya entrenada leyendo sus configs y JSONs."""
    round_dir = run_group_dir / build_round_dir_name(round_number)
    configs_dir = round_dir / "configs"
    logs_dir = round_dir / "logs"
    artifacts_dir = round_dir / "artefactos"

    if not configs_dir.exists():
        raise SystemExit(f"No existe el directorio de configs: {configs_dir}")

    config_paths = sorted(configs_dir.glob("exp_*.json"))
    rows = []
    experiment_grid = []

    print("\n" + "#" * 80)
    print(f"Reconstruyendo resumen de la ronda {round_number:02d}")
    print(f"Directorio de la ronda: {round_dir}")
    print(f"Configs encontradas: {len(config_paths)}")

    for config_path in config_paths:
        experiment_id = config_path.stem
        experiment_number = parse_experiment_number_from_id(experiment_id)
        if experiment_number is None:
            continue
        if start_exp is not None and experiment_number < start_exp:
            continue
        if end_exp is not None and experiment_number > end_exp:
            continue

        with open(config_path, "r", encoding="utf-8") as config_handle:
            applied_overrides = json.load(config_handle)

        experiment_grid.append(applied_overrides)
        experiment_label = applied_overrides.get("EXPERIMENT_LABEL", experiment_id)
        run_name_prefix = applied_overrides.get(
            "RUN_NAME_PREFIX",
            f"{experiment_id}_{experiment_label}",
        )
        experiment_artifacts_dir = Path(
            applied_overrides.get("RUN_OUTPUT_DIR", artifacts_dir)
        )
        log_path = logs_dir / f"{experiment_id}.log"

        postprocess_json_path = find_existing_postprocess_json(
            experiment_artifacts_dir,
            run_name_prefix,
        )
        if postprocess_json_path is None:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={},
                status="missing_json",
                return_code=None,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            rows.append(row)
            print(f"Sin resultado completo: {experiment_id} -> {experiment_label}")
            continue

        with open(postprocess_json_path, "r", encoding="utf-8") as result_handle:
            postprocess_data = json.load(result_handle)
        postprocess_data["postprocess_json_path"] = str(postprocess_json_path)

        row = build_summary_row(
            experiment_id=experiment_id,
            overrides=applied_overrides,
            postprocess_data=postprocess_data,
            status="ok",
            return_code=0,
            log_path=log_path,
            sweep_round=round_number,
            round_dir=round_dir,
        )
        rows.append(row)
        print(f"Resultado incluido: {experiment_id} -> {experiment_label}")

    ranked_rows = rank_rows(rows)
    csv_path, json_path = save_summary_files(round_dir, rows)

    print("\n" + "=" * 80)
    print(f"Resumen CSV guardado en: {csv_path}")
    print(f"Resumen JSON guardado en: {json_path}")
    if ranked_rows:
        print("\nMejor experimento de la ronda:")
        print(format_row_compact(ranked_rows[0]))
    else:
        print("\nNo hay experimentos exitosos para comparar en esta ronda.")

    return {
        "round_number": round_number,
        "round_dir": round_dir,
        "experiment_grid": experiment_grid,
        "rows": rows,
        "ranked_rows": ranked_rows,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def collect_seen_signatures(run_group_dir, max_round_number, fallback_grid=None):
    """Recoge firmas ya probadas para que el refinamiento no duplique configs."""
    seen_signatures = set()

    if fallback_grid:
        for experiment in fallback_grid:
            seen_signatures.add(build_experiment_signature_from_overrides(experiment))

    for round_number in list_existing_round_numbers(run_group_dir):
        if round_number > max_round_number:
            continue
        configs_dir = run_group_dir / build_round_dir_name(round_number) / "configs"
        if not configs_dir.exists():
            continue
        for config_path in sorted(configs_dir.glob("exp_*.json")):
            with open(config_path, "r", encoding="utf-8") as config_handle:
                overrides = json.load(config_handle)
            seen_signatures.add(build_experiment_signature_from_overrides(overrides))

    return seen_signatures


def rank_rows(rows):
    """Ordena experimentos exitosos por las metricas principales del proyecto."""
    successful_rows = [row for row in rows if row["status"] == "ok"]

    successful_rows.sort(
        key=lambda row: (
            1 if safe_bool(row, "selected_threshold_constraint_satisfied", default=False) else 0,
            safe_metric(row, PRIMARY_RANK_METRIC),
            safe_metric(row, SECONDARY_RANK_METRIC),
            -safe_metric(row, FALSE_ALARM_RANK_METRIC, default=1e9),
            safe_metric(row, TERTIARY_RANK_METRIC),
            safe_metric(row, QUATERNARY_RANK_METRIC),
            -safe_metric(row, "validation_false_alarms_per_min", default=1e9),
        ),
        reverse=True,
    )
    return successful_rows


def get_incomplete_rows(rows):
    """Filas que no tienen resultado completo y no deben alimentar el ranking final."""
    return [row for row in rows if row.get("status") != "ok"]


def abort_if_partial_ranking(rows, allow_partial_ranking, context):
    """Evita seleccionar padres para iteracion con una tanda incompleta."""
    incomplete_rows = get_incomplete_rows(rows)
    if not incomplete_rows or allow_partial_ranking:
        return

    missing_labels = [
        f"{row.get('experiment_id')}:{row.get('status')}"
        for row in incomplete_rows
    ]
    raise SystemExit(
        "No se puede construir el ranking de iteracion con resultados "
        f"incompletos en {context}. Pendientes/fallidos: "
        + ", ".join(missing_labels)
        + ". Revisa esos experimentos o usa --allow-partial-ranking "
        "si quieres hacerlo explicitamente."
    )


def save_summary_files(output_dir, rows, base_name="resumen_experimentos"):
    """Guarda un resumen de experimentos en CSV y JSON."""
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    fieldnames = list(rows[0].keys()) if rows else []

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as json_handle:
        json.dump(rows, json_handle, indent=2)

    return csv_path, json_path


def get_best_result_for_group(rows, predicate):
    """Devuelve el mejor experimento dentro de un subconjunto concreto."""
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return None
    return subset[0]


def print_best_result_for_group(rows, title, predicate):
    """Imprime el mejor experimento dentro de un subconjunto concreto."""
    best_row = get_best_result_for_group(rows, predicate)
    if best_row is None:
        return
    print(f"\n{title}:")
    print(format_row_compact(best_row))


def build_refinement_mutations(parent_row):
    """Devuelve mutaciones ordenadas por prioridad para refinar un buen candidato."""
    parent_overrides = derive_overrides_from_row(parent_row)
    current_chunk_length_s = float(parent_overrides["CHUNK_LENGTH_S"])
    current_frontend = parent_overrides["SPECTRAL_FRONTEND"]
    current_normalization = parent_overrides["SPECTROGRAM_NORMALIZATION"]
    current_balanced_batches = bool(parent_overrides["USE_BALANCED_CHUNK_BATCHES"])
    current_use_augmentation = bool(parent_overrides["USE_DATA_AUGMENTATION"])
    current_augmentation_prob = float(parent_overrides.get("AUGMENTATION_APPLY_PROB", 0.5))
    current_use_spectral_eq = bool(parent_overrides.get("USE_SPECTRAL_EQ_AUGMENTATION", True))
    current_eq_probability = float(parent_overrides.get("EQ_AUGMENTATION_PROB", 0.20))

    target_chunk_length_s = 1.0 if current_chunk_length_s < 0.75 else 0.5
    target_frontend = "log_mel" if current_frontend == "linear_stft" else "linear_stft"
    target_normalization = "frequency" if current_normalization == "minmax" else "minmax"

    mutations = []

    if current_use_augmentation:
        if not math.isclose(current_augmentation_prob, 0.35, rel_tol=0.0, abs_tol=1e-9):
            mutations.append(
                (
                    "aug_p35",
                    {"AUGMENTATION_APPLY_PROB": 0.35},
                    "Reduce la probabilidad de augmentacion para conservar mas chunks limpios.",
                )
            )
        if not math.isclose(current_augmentation_prob, 0.65, rel_tol=0.0, abs_tol=1e-9):
            mutations.append(
                (
                    "aug_p65",
                    {"AUGMENTATION_APPLY_PROB": 0.65},
                    "Aumenta la presencia de augmentacion para exigir mas robustez al modelo.",
                )
            )
        if current_use_spectral_eq:
            mutations.append(
                (
                    "eq_off",
                    {"USE_SPECTRAL_EQ_AUGMENTATION": False},
                    "Mantiene augmentacion general pero elimina la EQ espectral para aislar su efecto.",
                )
            )
            if not math.isclose(current_eq_probability, 0.10, rel_tol=0.0, abs_tol=1e-9):
                mutations.append(
                    (
                        "eq_p10",
                        {"EQ_AUGMENTATION_PROB": 0.10},
                        "Reduce la frecuencia de la EQ espectral dentro de la augmentacion.",
                    )
                )
    else:
        mutations.append(
            (
                "augment_p35",
                {
                    "USE_DATA_AUGMENTATION": True,
                    "AUGMENTATION_APPLY_PROB": 0.35,
                },
                "Activa data augmentation suave para comprobar si ayuda sin dominar el train.",
            )
        )
        mutations.append(
            (
                "augment_p65",
                {
                    "USE_DATA_AUGMENTATION": True,
                    "AUGMENTATION_APPLY_PROB": 0.65,
                },
                "Activa data augmentation intensa para comprobar si mejora la robustez.",
            )
        )

    mutations.append(
        (
            f"norm_{target_normalization}",
            {"SPECTROGRAM_NORMALIZATION": target_normalization},
            "Invierte la normalizacion del espectrograma manteniendo el resto constante.",
        )
    )
    mutations.append(
        (
            "balanced_on" if not current_balanced_batches else "balanced_off",
            {"USE_BALANCED_CHUNK_BATCHES": not current_balanced_batches},
            "Invierte el balanceo por chunks para revisar su impacto local.",
        )
    )
    mutations.append(
        (
            f"window_{seconds_tag(target_chunk_length_s)}",
            build_temporal_overrides(target_chunk_length_s),
            "Cambia la ventana temporal manteniendo el resto de hiperparametros.",
        )
    )
    mutations.append(
        (
            f"frontend_{frontend_tag(target_frontend)}",
            build_frontend_overrides(target_frontend),
            "Cambia solo el frontend espectral para contrastar lineal frente a log-mel.",
        )
    )

    return mutations


def make_refinement_experiment(parent_row, round_number, label_suffix, extra_overrides, notes):
    """Construye un hijo de refinamiento a partir de una configuracion ganadora."""
    base_overrides = derive_overrides_from_row(parent_row)
    merged_overrides = dict(base_overrides)
    merged_overrides.update(extra_overrides)

    chunk_length_s = float(merged_overrides.get("CHUNK_LENGTH_S", 0.5))
    merged_overrides.update(build_temporal_overrides(chunk_length_s))

    frontend = merged_overrides.get("SPECTRAL_FRONTEND", "linear_stft")
    if frontend == "log_mel":
        merged_overrides.setdefault("MEL_BINS", DEFAULT_MEL_BINS)

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
        "COMPARISON_WINDOW_S": chunk_length_s,
        "COMPARISON_WINDOW_TAG": seconds_tag(chunk_length_s),
        "COMPARISON_FRONTEND": frontend,
        "COMPARISON_FRONTEND_TAG": frontend_tag(frontend),
        "EXPERIMENT_NOTES": (
            f"Refinamiento automatico desde {parent_row['experiment_label']} "
            f"(ronda {parent_round}). {notes}"
        ),
    }
    overrides.update(merged_overrides)
    return overrides


def select_refinement_parents(ranked_rows, max_parents):
    """Selecciona padres prometedores preservando algo de diversidad por seed base."""
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


def parse_training_epoch_progress(log_path):
    """
    Recupera la ultima epoca completada y el presupuesto total desde el log.

    Se usa para decidir si merece la pena relanzar un experimento con mas
    epocas o si ya paro pronto por EarlyStopping.
    """
    if not log_path:
        return None, None

    path = Path(log_path)
    if not path.exists():
        return None, None

    last_completed_epoch = None
    configured_epoch_budget = None
    epoch_pattern = re.compile(r"^Epoch\s+(\d+)/(\d+)")

    with open(path, "r", encoding="utf-8", errors="replace") as log_handle:
        for line in log_handle:
            match = epoch_pattern.match(line)
            if match is None:
                continue
            last_completed_epoch = int(match.group(1))
            configured_epoch_budget = int(match.group(2))

    return last_completed_epoch, configured_epoch_budget


def row_exhausted_epoch_budget(row):
    """Indica si el entrenamiento original llego al tope de epocas configurado."""
    configured_epochs = int(row.get("config_epochs") or DEFAULT_TRAINING_EPOCHS)
    last_completed_epoch, logged_epoch_budget = parse_training_epoch_progress(
        row.get("log_path")
    )

    if last_completed_epoch is None:
        return False

    effective_epoch_budget = int(logged_epoch_budget or configured_epochs)
    return last_completed_epoch >= effective_epoch_budget


def select_final_rerun_rows(ranked_rows, max_parents):
    """
    Elige candidatos para el rerun final a partir del ranking global.

    Si hay experimentos factibles respecto al limite de falsas alarmas, se
    priorizan esos. Despues se reutiliza la misma logica de diversidad por seed
    base que en el refinamiento automatico.
    """
    candidate_rows = ranked_rows
    if FINAL_RERUN_REQUIRE_EPOCH_BUDGET_EXHAUSTION:
        candidate_rows = [
            row
            for row in candidate_rows
            if row_exhausted_epoch_budget(row)
        ]
        if not candidate_rows:
            return []

    if FINAL_RERUN_PREFER_FEASIBLE:
        feasible_rows = [
            row
            for row in candidate_rows
            if safe_bool(row, "selected_threshold_constraint_satisfied", default=False)
        ]
        if feasible_rows:
            candidate_rows = feasible_rows

    return select_refinement_parents(candidate_rows, max_parents)


def make_final_rerun_experiment(parent_row, rerun_rank):
    """Construye un relanzamiento final con mas epocas desde una fila ganadora."""
    overrides = derive_overrides_from_row(parent_row)
    overrides["EPOCHS"] = FINAL_RERUN_EPOCHS

    chunk_length_s = float(overrides.get("CHUNK_LENGTH_S", 0.5))
    frontend = overrides.get("SPECTRAL_FRONTEND", "linear_stft")
    base_experiment_id = parent_row.get("base_experiment_id") or "derived"
    parent_round = parent_row.get("sweep_round")
    parent_experiment_id = parent_row.get("experiment_id")

    experiment_label = (
        f"final_rerun_{rerun_rank:02d}_"
        f"{base_experiment_id}_r{parent_round}_{parent_experiment_id}_"
        f"e{FINAL_RERUN_EPOCHS}"
    )

    metadata_overrides = {
        "EXPERIMENT_LABEL": experiment_label,
        "EXPERIMENT_TIER": "final_rerun",
        "BASE_EXPERIMENT_ID": parent_row.get("base_experiment_id"),
        "BASE_EXPERIMENT_LABEL": parent_row.get("base_experiment_label"),
        "PARENT_EXPERIMENT_ID": parent_experiment_id,
        "PARENT_EXPERIMENT_LABEL": parent_row.get("experiment_label"),
        "PARENT_SWEEP_ROUND": parent_round,
        "COMPARISON_WINDOW_S": chunk_length_s,
        "COMPARISON_WINDOW_TAG": seconds_tag(chunk_length_s),
        "COMPARISON_FRONTEND": frontend,
        "COMPARISON_FRONTEND_TAG": frontend_tag(frontend),
        "EXPERIMENT_NOTES": (
            f"Rerun final desde {parent_row.get('experiment_label')} "
            f"(ronda {parent_round}) elevando el presupuesto a "
            f"{FINAL_RERUN_EPOCHS} epocas sin cambiar el resto de hiperparametros."
        ),
    }

    metadata_overrides.update(overrides)
    return metadata_overrides


def build_final_rerun_grid(ranked_rows):
    """Genera la mini-bateria final a partir del ranking global del barrido."""
    selected_rows = select_final_rerun_rows(ranked_rows, FINAL_RERUN_TOP_K)
    experiments = [
        make_final_rerun_experiment(parent_row=row, rerun_rank=index)
        for index, row in enumerate(selected_rows, start=1)
    ]
    return selected_rows, experiments


def build_refinement_experiment_grid(parent_rows, seen_signatures, round_number):
    """Construye la siguiente ronda a partir de las mejores configuraciones."""
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
    """Guarda un reporte Markdown corto con lo mejor de la ronda."""
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

        report_lines.extend(
            [
                "",
                "## Mejores por grupo",
            ]
        )
        group_definitions = [
            ("Matriz core", lambda row: row["experiment_tier"] == "core"),
            ("Bloque augment", lambda row: row["experiment_tier"] == "augment"),
            ("Familia exp008", lambda row: row["experiment_tier"] == "exp008"),
        ]
        for title, predicate in group_definitions:
            best_row = get_best_result_for_group(ranked_rows, predicate)
            if best_row is not None:
                report_lines.append(f"- {title}: {format_row_compact(best_row)}")
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
                f"frontend={experiment['SPECTRAL_FRONTEND']} | "
                f"window={experiment['CHUNK_LENGTH_S']} s | "
                f"norm={experiment['SPECTROGRAM_NORMALIZATION']} | "
                f"balanced={experiment['USE_BALANCED_CHUNK_BATCHES']} | "
                f"augment={experiment['USE_DATA_AUGMENTATION']}"
            )

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def save_final_rerun_report(rerun_dir, selected_rows, rerun_rows, ranked_rows):
    """Guarda un reporte breve del mini-bloque de rerun final."""
    report_path = rerun_dir / "reporte_rerun_final.md"
    report_lines = [
        "# Reporte rerun final",
        "",
        f"- Candidatos seleccionados: {len(selected_rows)}",
        f"- Experimentos relanzados: {len(rerun_rows)}",
        f"- Epocas del rerun final: {FINAL_RERUN_EPOCHS}",
    ]

    if selected_rows:
        report_lines.extend(["", "## Padres elegidos"])
        for rank, row in enumerate(selected_rows, start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    if ranked_rows:
        report_lines.extend(["", f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))} del rerun final"])
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")
    else:
        report_lines.extend(["", "No hay experimentos exitosos en el rerun final."])

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def save_global_report(run_group_dir, all_rows, round_results, final_rerun_result=None):
    """Guarda un reporte global con el consolidado de todas las rondas."""
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

    if final_rerun_result is not None:
        rerun_ranked_rows = final_rerun_result["ranked_rows"]
        report_lines.extend(
            [
                "",
                "## Rerun final",
                f"- Experimentos relanzados: {len(final_rerun_result['rows'])}",
                f"- Directorio: {final_rerun_result['rerun_dir']}",
            ]
        )
        if rerun_ranked_rows:
            report_lines.append(f"- Mejor rerun final: {format_row_compact(rerun_ranked_rows[0])}")

    if ranked_rows:
        report_lines.extend(["", f"## Top {min(ROUND_REPORT_TOP_N, len(ranked_rows))} global"])
        for rank, row in enumerate(ranked_rows[:ROUND_REPORT_TOP_N], start=1):
            report_lines.append(f"{rank}. {format_row_compact(row)}")

    with open(report_path, "w", encoding="utf-8") as report_handle:
        report_handle.write("\n".join(report_lines) + "\n")

    return report_path


def execute_round(
    experiment_grid,
    round_number,
    run_group_dir,
    start_exp=None,
    end_exp=None,
    skip_completed=False,
    summarize_only=False,
):
    """Ejecuta una ronda completa de barrido y devuelve su resumen."""
    experiment_items = list(
        iter_experiment_items(
            experiment_grid,
            start_exp=start_exp,
            end_exp=end_exp,
        )
    )
    round_experiment_grid = [
        experiment_overrides
        for _, experiment_overrides in experiment_items
    ]

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
    print(f"Numero de experimentos a ejecutar: {len(experiment_items)}")

    all_rows = []

    for experiment_number, experiment_overrides in experiment_items:
        context = build_experiment_context(
            experiment_number,
            experiment_overrides,
            configs_dir,
            logs_dir,
            artifacts_dir,
        )
        experiment_id = context["experiment_id"]
        experiment_label = context["experiment_label"]
        run_name_prefix = context["run_name_prefix"]
        applied_overrides = context["applied_overrides"]
        config_path = context["config_path"]
        log_path = context["log_path"]

        save_experiment_config(config_path, applied_overrides)

        print("\n" + "=" * 80)
        print(f"Ejecutando {experiment_id} -> {experiment_label}")
        print(json.dumps(applied_overrides, indent=2))

        existing_postprocess_json = find_existing_postprocess_json(
            artifacts_dir,
            run_name_prefix,
        )
        if existing_postprocess_json is not None and (skip_completed or summarize_only):
            with open(existing_postprocess_json, "r", encoding="utf-8") as result_handle:
                postprocess_data = json.load(result_handle)
            postprocess_data["postprocess_json_path"] = str(existing_postprocess_json)

            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data=postprocess_data,
                status="ok",
                return_code=0,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(
                f"Resultado existente para {experiment_id}: "
                f"{existing_postprocess_json}"
            )
            continue

        if summarize_only:
            row = build_summary_row(
                experiment_id=experiment_id,
                overrides=applied_overrides,
                postprocess_data={},
                status="missing_json",
                return_code=None,
                log_path=log_path,
                sweep_round=round_number,
                round_dir=round_dir,
            )
            all_rows.append(row)
            print(
                f"Sin resultado completo para {experiment_id}; "
                "no se entrena porque --summarize-only esta activo."
            )
            continue

        previous_matches = set(
            path.resolve()
            for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
        )

        process_env = os.environ.copy()
        process_env["SIREN_TRAINING_CONFIG_PATH"] = str(config_path)

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
            artifacts_dir,
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
            print(f"El experimento {experiment_id} termino pero no genero JSON de resultados.")

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

        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento de la matriz core",
            predicate=lambda row: row["experiment_tier"] == "core",
        )
        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento con data augmentation",
            predicate=lambda row: row["experiment_tier"] == "augment",
        )
        print_best_result_for_group(
            ranked_rows,
            title="Mejor experimento de la familia exp008",
            predicate=lambda row: row["experiment_tier"] == "exp008",
        )
    else:
        print("\nNo hay experimentos exitosos para comparar en esta ronda.")

    return {
        "round_number": round_number,
        "round_dir": round_dir,
        "experiment_grid": round_experiment_grid,
        "rows": all_rows,
        "ranked_rows": ranked_rows,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def execute_final_rerun(final_rerun_grid, run_group_dir):
    """Ejecuta un mini-bloque final separado del barrido iterativo principal."""
    rerun_dir = run_group_dir / FINAL_RERUN_DIR_NAME
    configs_dir = rerun_dir / "configs"
    logs_dir = rerun_dir / "logs"
    artifacts_dir = rerun_dir / "artefactos"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print("Rerun final")
    print(f"Script de entrenamiento: {TRAINING_SCRIPT_PATH}")
    print(f"Directorio del rerun final: {rerun_dir}")
    print(f"Numero de experimentos a relanzar: {len(final_rerun_grid)}")
    print(f"Epocas del rerun final: {FINAL_RERUN_EPOCHS}")

    all_rows = []

    for experiment_number, experiment_overrides in enumerate(final_rerun_grid, start=1):
        experiment_id = f"final_{experiment_number:03d}"
        experiment_label = experiment_overrides.get("EXPERIMENT_LABEL", experiment_id)
        run_name_prefix = f"{experiment_id}_{experiment_label}"

        applied_overrides = dict(FIXED_OVERRIDES)
        applied_overrides.update(experiment_overrides)
        applied_overrides["RUN_OUTPUT_DIR"] = str(artifacts_dir)
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
            for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
        )

        process_env = os.environ.copy()
        process_env["SIREN_TRAINING_CONFIG_PATH"] = str(config_path)

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
            artifacts_dir,
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
                sweep_round="final_rerun",
                round_dir=rerun_dir,
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
                sweep_round="final_rerun",
                round_dir=rerun_dir,
            )
            all_rows.append(row)
            print(f"El experimento {experiment_id} termino pero no genero JSON de resultados.")

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
            sweep_round="final_rerun",
            round_dir=rerun_dir,
        )
        all_rows.append(row)

    ranked_rows = rank_rows(all_rows)
    csv_path, json_path = save_summary_files(rerun_dir, all_rows)

    print("\n" + "=" * 80)
    print(f"Resumen CSV del rerun final guardado en: {csv_path}")
    print(f"Resumen JSON del rerun final guardado en: {json_path}")

    if ranked_rows:
        print("\nMejor experimento del rerun final:")
        print(format_row_compact(ranked_rows[0]))
    else:
        print("\nNo hay experimentos exitosos en el rerun final.")

    return {
        "rerun_dir": rerun_dir,
        "rows": all_rows,
        "ranked_rows": ranked_rows,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el barrido CNN o genera configs para continuarlo "
            "con jobs individuales/job arrays."
        )
    )
    parser.add_argument(
        "--run-group-dir",
        type=Path,
        default=None,
        help=(
            "Directorio raiz del barrido. Si es relativo, se interpreta dentro "
            "de 'Modelos Atlas'. Si se omite, se crea uno nuevo."
        ),
    )
    parser.add_argument(
        "--generate-configs-only",
        action="store_true",
        help="Genera JSONs y un listado para job array, sin entrenar.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Reconstruye resumenes/reportes desde *_postprocesado.json, "
            "sin lanzar entrenamientos."
        ),
    )
    parser.add_argument(
        "--summarize-all-rounds",
        action="store_true",
        help=(
            "Reconstruye resumenes de todas las carpetas ronda_* existentes "
            "y genera un ranking global sin entrenar."
        ),
    )
    parser.add_argument(
        "--generate-refinement-configs-only",
        action="store_true",
        help=(
            "Consolida la ronda 01, calcula el ranking completo y genera "
            "configs de la ronda 02 para lanzarlas como job array."
        ),
    )
    parser.add_argument(
        "--start-exp",
        type=parse_experiment_number,
        default=None,
        help="Primer experimento a incluir, por ejemplo 5 o exp_005.",
    )
    parser.add_argument(
        "--end-exp",
        type=parse_experiment_number,
        default=None,
        help="Ultimo experimento a incluir, por ejemplo 24 o exp_024.",
    )
    parser.add_argument(
        "--round-number",
        type=int,
        default=1,
        help="Ronda a consolidar con --summarize-only. Por defecto: 1.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Omite experimentos que ya tengan *_postprocesado.json.",
    )
    parser.add_argument(
        "--allow-partial-ranking",
        action="store_true",
        help=(
            "Permite rankear aunque falten experimentos. Por defecto se bloquea "
            "para evitar iteraciones sesgadas por tandas incompletas."
        ),
    )
    parser.add_argument(
        "--config-list-name",
        default="configs_restantes_cnn.txt",
        help="Nombre del listado de configs generado en modo --generate-configs-only.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_SWEEP_ROUNDS,
        help="Numero maximo de rondas si se ejecuta el barrido.",
    )
    parser.add_argument(
        "--no-auto-refinement",
        action="store_true",
        help="Desactiva la generacion/ejecucion automatica de rondas refinadas.",
    )
    parser.add_argument(
        "--no-final-rerun",
        action="store_true",
        help="Desactiva el rerun final de mejores experimentos.",
    )
    return parser.parse_args()


def resolve_run_group_dir(run_group_dir_arg):
    if run_group_dir_arg is None:
        run_group_name = datetime.now().strftime(
            "barrido_margin_3_iterativo_%Y%m%d_%H%M%S"
        )
        return RUNS_ROOT_DIR / run_group_name

    if run_group_dir_arg.is_absolute():
        return run_group_dir_arg

    return RUNS_ROOT_DIR / run_group_dir_arg


def main():
    args = parse_args()
    if (
        args.start_exp is not None
        and args.end_exp is not None
        and args.start_exp > args.end_exp
    ):
        raise SystemExit("--start-exp no puede ser mayor que --end-exp.")
    if args.summarize_only and args.generate_configs_only:
        raise SystemExit("--summarize-only no se puede combinar con --generate-configs-only.")
    if args.summarize_all_rounds and args.generate_configs_only:
        raise SystemExit(
            "--summarize-all-rounds no se puede combinar con --generate-configs-only."
        )
    if args.generate_refinement_configs_only and args.generate_configs_only:
        raise SystemExit(
            "--generate-refinement-configs-only no se puede combinar con "
            "--generate-configs-only."
        )
    if args.summarize_all_rounds and args.generate_refinement_configs_only:
        raise SystemExit(
            "--summarize-all-rounds no se puede combinar con "
            "--generate-refinement-configs-only."
        )

    run_group_dir = resolve_run_group_dir(args.run_group_dir)
    run_group_dir.mkdir(parents=True, exist_ok=True)

    current_experiment_grid = build_experiment_grid()
    all_rows_global = []
    round_results = []
    seen_signatures = set()
    final_rerun_result = None
    auto_iterative_refinement = (
        AUTO_ITERATIVE_REFINEMENT and not args.no_auto_refinement
    )
    final_rerun_enabled = FINAL_RERUN_ENABLED and not args.no_final_rerun

    if args.generate_configs_only:
        generate_round_configs(
            experiment_grid=current_experiment_grid,
            round_number=1,
            run_group_dir=run_group_dir,
            start_exp=args.start_exp,
            end_exp=args.end_exp,
            skip_completed=args.skip_completed,
            config_list_name=args.config_list_name,
        )
        return

    if args.summarize_all_rounds:
        round_numbers = list_existing_round_numbers(run_group_dir)
        if not round_numbers:
            raise SystemExit(f"No se encontraron carpetas ronda_* en: {run_group_dir}")

        for round_number in round_numbers:
            round_result = summarize_round_from_configs(
                run_group_dir=run_group_dir,
                round_number=round_number,
                start_exp=args.start_exp if round_number == args.round_number else None,
                end_exp=args.end_exp if round_number == args.round_number else None,
            )
            round_results.append(round_result)
            all_rows_global.extend(round_result["rows"])

        abort_if_partial_ranking(
            all_rows_global,
            allow_partial_ranking=args.allow_partial_ranking,
            context="resumen_global",
        )

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
        return

    if args.summarize_only or args.generate_refinement_configs_only:
        if args.round_number == 1:
            round_result = execute_round(
                experiment_grid=current_experiment_grid,
                round_number=1,
                run_group_dir=run_group_dir,
                start_exp=args.start_exp,
                end_exp=args.end_exp,
                skip_completed=True,
                summarize_only=True,
            )
        else:
            round_result = summarize_round_from_configs(
                run_group_dir=run_group_dir,
                round_number=args.round_number,
                start_exp=args.start_exp,
                end_exp=args.end_exp,
            )
        round_results.append(round_result)
        all_rows_global.extend(round_result["rows"])

        abort_if_partial_ranking(
            round_result["rows"],
            allow_partial_ranking=args.allow_partial_ranking,
            context="ronda_01",
        )

        selected_parents = []
        next_round_grid = None
        if auto_iterative_refinement:
            next_round_number = args.round_number + 1
            selected_parents = select_refinement_parents(
                round_result["ranked_rows"],
                max_parents=REFINEMENT_TOP_K,
            )
            seen_signatures = collect_seen_signatures(
                run_group_dir=run_group_dir,
                max_round_number=args.round_number,
                fallback_grid=current_experiment_grid,
            )
            next_round_grid = build_refinement_experiment_grid(
                parent_rows=selected_parents,
                seen_signatures=seen_signatures,
                round_number=next_round_number,
            )

        round_report_path = save_round_report(
            round_dir=round_result["round_dir"],
            round_number=1,
            round_rows=round_result["rows"],
            ranked_rows=round_result["ranked_rows"],
            selected_parents=selected_parents,
            next_round_grid=next_round_grid,
        )
        print(f"\nReporte de ronda guardado en: {round_report_path}")

        if args.generate_refinement_configs_only:
            if not next_round_grid:
                print("\nNo hay configuraciones de refinamiento que generar.")
                return
            generate_round_configs(
                experiment_grid=next_round_grid,
                round_number=args.round_number + 1,
                run_group_dir=run_group_dir,
                skip_completed=args.skip_completed,
                config_list_name=args.config_list_name,
            )
            return

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
        return

    print(f"Directorio raiz del barrido: {run_group_dir}")
    print(
        "Modo iterativo: "
        f"{auto_iterative_refinement} | max rondas: {args.max_rounds} | "
        f"top padres: {REFINEMENT_TOP_K}"
    )

    for round_number in range(1, args.max_rounds + 1):
        if not current_experiment_grid:
            print("\nNo hay configuraciones nuevas que ejecutar. Fin del barrido.")
            break

        round_result = execute_round(
            experiment_grid=current_experiment_grid,
            round_number=round_number,
            run_group_dir=run_group_dir,
            start_exp=args.start_exp if round_number == 1 else None,
            end_exp=args.end_exp if round_number == 1 else None,
            skip_completed=args.skip_completed,
        )
        round_results.append(round_result)
        all_rows_global.extend(round_result["rows"])

        for experiment in round_result["experiment_grid"]:
            seen_signatures.add(build_experiment_signature_from_overrides(experiment))

        selected_parents = []
        next_round_grid = None

        if auto_iterative_refinement and round_number < args.max_rounds:
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

        if not auto_iterative_refinement or round_number >= args.max_rounds:
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

    if final_rerun_enabled:
        global_ranked_rows_before_rerun = rank_rows(all_rows_global)
        selected_final_rerun_rows, final_rerun_grid = build_final_rerun_grid(
            global_ranked_rows_before_rerun
        )

        if final_rerun_grid:
            final_rerun_result = execute_final_rerun(
                final_rerun_grid=final_rerun_grid,
                run_group_dir=run_group_dir,
            )
            final_rerun_report_path = save_final_rerun_report(
                rerun_dir=final_rerun_result["rerun_dir"],
                selected_rows=selected_final_rerun_rows,
                rerun_rows=final_rerun_result["rows"],
                ranked_rows=final_rerun_result["ranked_rows"],
            )
            all_rows_global.extend(final_rerun_result["rows"])
            print(f"\nReporte del rerun final guardado en: {final_rerun_report_path}")
        else:
            print("\nNo hay candidatos validos para ejecutar el rerun final.")

    global_csv_path, global_json_path = save_summary_files(
        run_group_dir,
        all_rows_global,
        base_name="resumen_global",
    )
    global_report_path = save_global_report(
        run_group_dir=run_group_dir,
        all_rows=all_rows_global,
        round_results=round_results,
        final_rerun_result=final_rerun_result,
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


if __name__ == "__main__":
    main()
