import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Script de barrido de experimentos para `entrenar_modelo_clasif_trad.py`.
#
# Uso esperado:
# 1. Ajustar solo esta cabecera para decidir que variantes comparar.
# 2. Lanzar este script una vez.
# 3. Revisar el resumen CSV/JSON y quedarse con el mejor experimento.
#
# El ranking principal se basa en F2 de validacion del modelo final
# recalibrado, priorizando recall sin ignorar falsas alarmas.
# ---------------------------------------------------------------------------


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT_PATH = SCRIPT_DIR / "entrenar_modelo_clasif_trad.py"
LOCAL_VENV_PYTHON = SCRIPT_DIR.parent / ".venv" / "Scripts" / "python.exe"


# ---------------------------------------------------------------------------
# Configuracion del barrido
# ---------------------------------------------------------------------------
PYTHON_COMMAND = (
    [str(LOCAL_VENV_PYTHON)]
    if LOCAL_VENV_PYTHON.exists()
    else ["py", "-3"]
)
CONTINUE_ON_ERROR = True
STREAM_TRAINING_LOGS = True
MAX_EXPERIMENTS = None

PRIMARY_RANK_METRIC = "validation_refit_f2"
SECONDARY_RANK_METRIC = "validation_refit_recall"
TERTIARY_RANK_METRIC = "validation_refit_auc_pr"


# ---------------------------------------------------------------------------
# Overrides fijos comunes a todos los experimentos.
# ---------------------------------------------------------------------------
FIXED_OVERRIDES = {
    "RANDOM_SEED": 42,
    "SAVE_EXPERIMENT_REPORT": True,
    "SAVE_POSTPROCESSING_CONFIG": True,
    "SHOW_RF_PLOT": False,
    "TARGET_FALSE_ALARMS_PER_MIN": 1.0,
    "CHUNK_LENGTH_S": 0.5,
    "OVERLAP_S": 0.125,
    "USE_SPECTRAL_EQ_AUGMENTATION": True,
    "EQ_AUGMENTATION_PROB": 0.20,
}


# ---------------------------------------------------------------------------
# Bateria inicial controlada de experimentos.
#
# Se compara el impacto de:
# - overlap
# - class weights
# - data augmentation
# - pitch shift dentro de la augmentacion
# ---------------------------------------------------------------------------
EXPERIMENT_BATTERY = [
    {
        "EXPERIMENT_LABEL": "baseline_limpio",
        "USE_OVERLAP": False,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "solo_overlap",
        "USE_OVERLAP": True,
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
        "EXPERIMENT_LABEL": "overlap_y_augmentacion",
        "USE_OVERLAP": True,
        "USE_CLASS_WEIGHTS": False,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": False,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 1,
    },
    {
        "EXPERIMENT_LABEL": "overlap_y_class_weights",
        "USE_OVERLAP": True,
        "USE_CLASS_WEIGHTS": True,
        "USE_DATA_AUGMENTATION": False,
    },
    {
        "EXPERIMENT_LABEL": "combinacion_prometedora",
        "USE_OVERLAP": True,
        "USE_CLASS_WEIGHTS": True,
        "USE_DATA_AUGMENTATION": True,
        "USE_PITCH_SHIFT_AUGMENTATION": False,
        "AUGMENTATION_APPLY_PROB": 0.5,
        "AUGMENTATION_EXTRA_COPIES": 1,
    },
]


def build_experiment_grid():
    return [dict(experiment) for experiment in EXPERIMENT_BATTERY]


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
):
    validation_metrics = postprocess_data.get("validation_metrics") or {}
    validation_metrics_refit = postprocess_data.get("validation_metrics_refit") or {}
    test_metrics = postprocess_data.get("test_metrics") or {}

    return {
        "experiment_id": experiment_id,
        "experiment_label": overrides.get("EXPERIMENT_LABEL"),
        "status": status,
        "return_code": return_code,
        "winner_name": postprocess_data.get("winner_name"),
        "config_chunk_length_s": overrides.get("CHUNK_LENGTH_S"),
        "config_use_overlap": overrides.get("USE_OVERLAP"),
        "config_overlap_s": overrides.get("OVERLAP_S"),
        "config_use_class_weights": overrides.get("USE_CLASS_WEIGHTS"),
        "config_use_data_augmentation": overrides.get("USE_DATA_AUGMENTATION"),
        "config_use_pitch_shift_augmentation": overrides.get(
            "USE_PITCH_SHIFT_AUGMENTATION"
        ),
        "config_use_spectral_eq_augmentation": overrides.get(
            "USE_SPECTRAL_EQ_AUGMENTATION"
        ),
        "config_eq_augmentation_prob": overrides.get("EQ_AUGMENTATION_PROB"),
        "config_augmentation_apply_prob": overrides.get("AUGMENTATION_APPLY_PROB"),
        "config_augmentation_extra_copies": overrides.get(
            "AUGMENTATION_EXTRA_COPIES"
        ),
        "recommended_chunk_threshold": postprocess_data.get("recommended_chunk_threshold"),
        "bundle_path": postprocess_data.get("bundle_path"),
        "run_output_dir": postprocess_data.get("run_output_dir"),
        "postprocess_json_path": postprocess_data.get("postprocess_json_path"),
        "log_path": str(log_path),
        "validation_precision": validation_metrics.get("precision"),
        "validation_recall": validation_metrics.get("recall"),
        "validation_f1": validation_metrics.get("f1"),
        "validation_f2": validation_metrics.get("f2"),
        "validation_auc_pr": validation_metrics.get("auc_pr"),
        "validation_false_alarms_per_min": validation_metrics.get(
            "false_alarms_per_min"
        ),
        "validation_refit_precision": validation_metrics_refit.get("precision"),
        "validation_refit_recall": validation_metrics_refit.get("recall"),
        "validation_refit_f1": validation_metrics_refit.get("f1"),
        "validation_refit_f2": validation_metrics_refit.get("f2"),
        "validation_refit_auc_pr": validation_metrics_refit.get("auc_pr"),
        "validation_refit_false_alarms_per_min": validation_metrics_refit.get(
            "false_alarms_per_min"
        ),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_f2": test_metrics.get("f2"),
        "test_auc_pr": test_metrics.get("auc_pr"),
        "test_false_alarms_per_min": test_metrics.get("false_alarms_per_min"),
    }


def stream_process_output(process, log_path):
    with open(log_path, "w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            if STREAM_TRAINING_LOGS:
                print(line, end="")


def find_new_postprocess_json(artifacts_dir, run_name_prefix, previous_matches):
    current_matches = set(
        path.resolve()
        for path in artifacts_dir.glob(f"{run_name_prefix}_*_postprocesado.json")
    )
    new_matches = sorted(current_matches - previous_matches)
    if not new_matches:
        return None
    return Path(new_matches[-1])


def rank_rows(rows):
    successful_rows = [row for row in rows if row["status"] == "ok"]
    successful_rows.sort(
        key=lambda row: (
            safe_metric(row, PRIMARY_RANK_METRIC),
            safe_metric(row, SECONDARY_RANK_METRIC),
            safe_metric(row, TERTIARY_RANK_METRIC),
            -safe_metric(
                row,
                "validation_refit_false_alarms_per_min",
                default=1e9,
            ),
        ),
        reverse=True,
    )
    return successful_rows


def save_summary_files(output_dir, rows):
    csv_path = output_dir / "resumen_experimentos.csv"
    json_path = output_dir / "resumen_experimentos.json"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as json_handle:
        json.dump(rows, json_handle, indent=2)

    return csv_path, json_path


if __name__ == "__main__":
    run_group_name = datetime.now().strftime("barrido_clasif_trad_%Y%m%d_%H%M%S")
    run_group_dir = SCRIPT_DIR / run_group_name
    configs_dir = run_group_dir / "configs"
    logs_dir = run_group_dir / "logs"
    artifacts_dir = run_group_dir / "artefactos"

    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    experiment_grid = build_experiment_grid()
    if MAX_EXPERIMENTS is not None:
        experiment_grid = experiment_grid[:MAX_EXPERIMENTS]

    print(f"Script de entrenamiento: {TRAINING_SCRIPT_PATH}")
    print(f"Comando Python: {' '.join(PYTHON_COMMAND)}")
    print(f"Directorio del barrido: {run_group_dir}")
    print(f"Numero de experimentos a ejecutar: {len(experiment_grid)}")

    all_rows = []

    for experiment_number, experiment_overrides in enumerate(experiment_grid, start=1):
        experiment_id = f"exp_{experiment_number:03d}"
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
            artifacts_dir,
            run_name_prefix,
            previous_matches,
        )

        if return_code != 0:
            row = build_summary_row(
                experiment_id,
                applied_overrides,
                {
                    "postprocess_json_path": (
                        str(postprocess_json_path) if postprocess_json_path else None
                    )
                },
                status="failed",
                return_code=return_code,
                log_path=log_path,
            )
            all_rows.append(row)
            print(f"El experimento {experiment_id} ha fallado con codigo {return_code}.")

            if not CONTINUE_ON_ERROR:
                break
            continue

        if postprocess_json_path is None:
            row = build_summary_row(
                experiment_id,
                applied_overrides,
                {},
                status="missing_json",
                return_code=return_code,
                log_path=log_path,
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
            experiment_id,
            applied_overrides,
            postprocess_data,
            status="ok",
            return_code=return_code,
            log_path=log_path,
        )
        all_rows.append(row)

    ranked_rows = rank_rows(all_rows)
    csv_path, json_path = save_summary_files(run_group_dir, all_rows)

    print("\n" + "=" * 80)
    print(f"Resumen CSV guardado en: {csv_path}")
    print(f"Resumen JSON guardado en: {json_path}")

    if ranked_rows:
        best_row = ranked_rows[0]
        print("\nMejor experimento segun validation_refit_f2:")
        print(json.dumps(best_row, indent=2))

        print("\nTop 5 experimentos:")
        for row in ranked_rows[:5]:
            print(
                f"{row['experiment_id']} ({row['experiment_label']}) | "
                f"winner={row['winner_name']} | "
                f"val_refit_f2={safe_metric(row, 'validation_refit_f2', default=0.0):.4f} | "
                f"val_refit_recall={safe_metric(row, 'validation_refit_recall', default=0.0):.4f} | "
                f"val_refit_auc_pr={safe_metric(row, 'validation_refit_auc_pr', default=0.0):.4f} | "
                f"false_alarms/min={safe_metric(row, 'validation_refit_false_alarms_per_min', default=0.0):.2f}"
            )
    else:
        print("\nNo hay experimentos exitosos para comparar.")
