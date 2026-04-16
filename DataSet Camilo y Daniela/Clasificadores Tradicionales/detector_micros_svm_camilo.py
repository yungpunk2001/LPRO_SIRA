import argparse
import queue
import sys
import warnings
from pathlib import Path

import numpy as np
import sounddevice as sd

from clasif_trad_utils import (
    extract_feature_vector,
    predict_positive_probability,
    prompt_user_to_select_inference_bundle,
)

warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent

# Fallback legado. Si se deja en None, se usa el valor indicado por CLI y,
# si tampoco se pasa por CLI, el valor calibrado en validacion del bundle.
THRESHOLD_OVERRIDE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detector en tiempo real para clasificadores tradicionales "
            "(SVM, Random Forest, KNN)."
        )
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Umbral manual para la decision binaria por chunk. Si no se "
            "indica, se usa el umbral recomendado del bundle cargado."
        ),
    )
    return parser.parse_args()


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
    bundle: dict,
    sr: int,
    chunk_samples: int,
    step_samples: int,
    decision_step_sec: float,
    positive_label: str,
    runtime_threshold: float,
) -> None:
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    audio_buffer = np.empty(0, dtype=np.float32)
    chunk_index = 0

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        blocksize=step_samples,
        dtype="float32",
        callback=build_stream_callback(audio_queue),
    ):
        print("\nEscuchando audio en vivo. Pulsa Ctrl+C para detener.\n")

        while True:
            block = audio_queue.get().reshape(-1).astype(np.float32, copy=False)
            audio_buffer = np.concatenate([audio_buffer, block])

            while len(audio_buffer) >= chunk_samples:
                y_chunk = audio_buffer[:chunk_samples].copy()
                audio_buffer = audio_buffer[step_samples:]

                features = extract_feature_vector(y_chunk, sr=sr)
                x_scaled = bundle["scaler"].transform(features.reshape(1, -1))
                positive_prob = predict_positive_probability(bundle, x_scaled)
                chunk_start_sec = chunk_index * decision_step_sec
                rms = float(np.sqrt(np.mean(np.square(y_chunk))))

                if positive_prob >= runtime_threshold:
                    print(
                        f"[{chunk_index + 1:05d}] t={chunk_start_sec:8.2f}s | "
                        f"ALERTA {positive_label.upper()} | "
                        f"probabilidad={positive_prob * 100:.1f}% | "
                        f"rms={rms:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{chunk_index + 1:05d}] t={chunk_start_sec:8.2f}s | "
                        f"Sin alerta | "
                        f"prob_{positive_label}={positive_prob * 100:.1f}% | "
                        f"rms={rms:.4f}",
                        flush=True,
                    )

                chunk_index += 1


def main() -> None:
    args = parse_args()

    print("Seleccion del clasificador tradicional para inferencia.")
    try:
        bundle = prompt_user_to_select_inference_bundle(models_dir=SCRIPT_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return
    except ValueError as exc:
        print(f"ERROR de metadatos del modelo: {exc}")
        return
    except (EOFError, KeyboardInterrupt):
        print("\nSeleccion cancelada por el usuario.")
        return

    metadata = bundle["metadata"]
    sr = int(metadata["sample_rate"])
    chunk_sec = float(metadata["chunk_seconds"])
    overlap_sec = float(metadata["overlap_seconds"])
    decision_step_sec = float(metadata["decision_step_seconds"])
    chunk_samples = int(round(chunk_sec * sr))
    step_samples = int(round(decision_step_sec * sr))
    if chunk_samples <= 0 or step_samples <= 0:
        print(
            "ERROR de metadatos del modelo: chunk_seconds y overlap_seconds "
            "deben producir un paso temporal positivo."
        )
        return

    positive_label = metadata.get("positive_label", "siren")
    runtime_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else (
            float(THRESHOLD_OVERRIDE)
            if THRESHOLD_OVERRIDE is not None
            else float(bundle["runtime_threshold"])
        )
    )
    bundle_path = metadata.get("bundle_path", "desconocido")

    print(
        "Sistema listo. "
        f"Modelo cargado: {metadata.get('model_name', metadata.get('winner_name', 'desconocido'))} | "
        f"Clase positiva: {positive_label} | "
        f"Umbral: {runtime_threshold:.2f} | "
        f"sr={sr} | chunk={chunk_sec:.3f}s | "
        f"overlap={overlap_sec:.3f}s | paso={decision_step_sec:.3f}s"
    )
    print(f"Bundle: {bundle_path}")

    try:
        run_stream(
            bundle=bundle,
            sr=sr,
            chunk_samples=chunk_samples,
            step_samples=step_samples,
            decision_step_sec=decision_step_sec,
            positive_label=positive_label,
            runtime_threshold=runtime_threshold,
        )
    except KeyboardInterrupt:
        print("\nSistema detenido por el usuario.")


if __name__ == "__main__":
    main()
