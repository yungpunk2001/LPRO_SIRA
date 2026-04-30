import time
from pathlib import Path

import numpy as np

import config_cnn as ccnn
import config_doa as cdoa
import cnn_runtime
import trad_runtime


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SR = cdoa.AUDIO['rate']


def project_relative_path(path_value: str | Path) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(BASE_DIR).as_posix()
        except ValueError:
            return path.as_posix()


def resolve_configured_directory(config_key: str, default_value: str) -> Path:
    configured_dir = ccnn.MODEL.get(config_key, default_value)
    return Path(cnn_runtime.resolve_path(configured_dir, BASE_DIR))


def resolve_cnn_directory() -> Path:
    legacy_dir = ccnn.MODEL.get('directory')
    return resolve_configured_directory('cnn_directory', legacy_dir or '../Modelos/CNN')


def resolve_traditional_directory() -> Path:
    return resolve_configured_directory('traditional_directory', '../Modelos/Tradicionales')


def discover_model_paths(model_dir: Path, pattern: str, model_kind: str) -> list[Path]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de modelos {model_kind}: "
            f"{project_relative_path(model_dir)}"
        )
    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"La ruta de modelos {model_kind} no es una carpeta: "
            f"{project_relative_path(model_dir)}"
        )
    return sorted(path for path in model_dir.glob(pattern) if path.is_file())


def model_key_candidates(
    artifact_path: Path,
    model_dir: Path,
    model_name: str,
) -> list[str]:
    candidates = [model_name, artifact_path.name, artifact_path.stem]

    try:
        relative_path = artifact_path.relative_to(model_dir)
        candidates.append(relative_path.as_posix())
        candidates.append(relative_path.with_suffix('').as_posix())
    except ValueError:
        pass

    project_path = project_relative_path(artifact_path)
    candidates.append(project_path)
    candidates.append(Path(project_path).with_suffix('').as_posix())

    return list(dict.fromkeys(candidates))


def configured_override_for(
    override_key: str,
    artifact_path: Path,
    model_dir: Path,
    model_name: str,
) -> float | None:
    overrides = ccnn.MODEL.get(override_key, {})
    for candidate in model_key_candidates(artifact_path, model_dir, model_name):
        if candidate in overrides:
            return float(overrides[candidate])
    return None


def threshold_override_for(
    artifact_path: Path,
    model_dir: Path,
    model_name: str,
) -> float | None:
    return configured_override_for(
        'threshold_overrides',
        artifact_path,
        model_dir,
        model_name,
    )


def required_positive_seconds_for(
    artifact_path: Path,
    model_dir: Path,
    model_name: str,
) -> float:
    override = configured_override_for(
        'required_positive_seconds_overrides',
        artifact_path,
        model_dir,
        model_name,
    )
    required_seconds = (
        override
        if override is not None
        else float(ccnn.MODEL.get('required_positive_seconds', 1.0))
    )
    if required_seconds <= 0:
        raise ValueError(
            "required_positive_seconds debe ser mayor que 0 para "
            f"{model_name}: {required_seconds}"
        )
    return required_seconds


def resolve_detection_channel(input_channels: int, is_respeaker: bool = True) -> int:
    if input_channels <= 0:
        raise ValueError(f"input_channels debe ser positivo: {input_channels}")

    config_key = 'respeaker_detection_channel' if is_respeaker else 'fallback_detection_channel'
    preferred_channel = int(ccnn.MODEL.get(config_key, 4 if is_respeaker else 0))
    if 0 <= preferred_channel < input_channels:
        return preferred_channel
    return 0


def detection_audio_channel(audio_buffer: np.ndarray, audio_channel: int) -> int:
    if audio_buffer.ndim != 2:
        raise ValueError(
            "El buffer de audio debe tener forma [muestras, canales], "
            f"pero tiene shape={audio_buffer.shape}."
        )

    channel = int(audio_channel)
    channel_count = int(audio_buffer.shape[1])
    if channel < 0 or channel >= channel_count:
        raise ValueError(
            f"Canal de deteccion invalido: {channel}. "
            f"El buffer tiene {channel_count} canal(es)."
        )
    return channel


def build_detection_state(
    *,
    kind: str,
    name: str,
    path: str,
    config_path: str,
    config: dict,
    required_positive_s: float,
    model=None,
    bundle=None,
) -> dict:
    window_samp = int(round(config['chunk_length_s'] * SR))
    step_samp = int(round(config['decision_step_s'] * SR))
    if window_samp <= 0 or step_samp <= 0:
        raise ValueError(
            f"Configuracion temporal invalida para {name}: "
            f"window_samp={window_samp}, step_samp={step_samp}"
        )

    return {
        'kind': kind,
        'name': name,
        'path': path,
        'config_path': config_path,
        'model': model,
        'bundle': bundle,
        'config': config,
        'required_positive_s': float(required_positive_s),
        'positive_streak_s': 0.0,
        'window_samp': window_samp,
        'step_samp': step_samp,
        'accumulator': 0,
        'period_results': [],
        'period_latency_ms': 0.0,
    }


def load_cnn_models(model_dir: Path) -> list[dict]:
    pattern = ccnn.MODEL.get('cnn_pattern', ccnn.MODEL.get('pattern', '**/*.keras'))
    states = []
    for model_path in discover_model_paths(model_dir, pattern, 'CNN'):
        model_path_str = str(model_path.resolve())
        model_name = f"CNN:{model_path.relative_to(model_dir).with_suffix('').as_posix()}"
        runtime_config, config_path = cnn_runtime.load_runtime_config(
            model_path_str,
            threshold_override=threshold_override_for(model_path, model_dir, model_name),
        )
        model = cnn_runtime.load_model_for_inference(model_path_str)
        runtime_config = cnn_runtime.apply_model_shape_hints(model, runtime_config)
        cnn_runtime.validate_model_against_runtime(model, runtime_config)

        states.append(
            build_detection_state(
                kind='cnn',
                name=model_name,
                path=project_relative_path(model_path),
                config_path=project_relative_path(config_path),
                config=runtime_config,
                required_positive_s=required_positive_seconds_for(
                    model_path,
                    model_dir,
                    model_name,
                ),
                model=model,
            )
        )
    return states


def load_traditional_models(model_dir: Path) -> list[dict]:
    pattern = ccnn.MODEL.get(
        'traditional_pattern',
        '**/*clasificador_tradicional_*_bundle.joblib',
    )
    states = []
    for bundle_path in discover_model_paths(model_dir, pattern, 'tradicionales'):
        bundle_path_str = str(bundle_path.resolve())
        model_name = f"TRAD:{bundle_path.relative_to(model_dir).with_suffix('').as_posix()}"
        bundle, runtime_config, config_path = trad_runtime.load_traditional_bundle(
            bundle_path_str
        )
        threshold_override = threshold_override_for(bundle_path, model_dir, model_name)
        if threshold_override is not None:
            runtime_config['chunk_threshold'] = threshold_override
            runtime_config['threshold_source'] = 'manual_config'

        states.append(
            build_detection_state(
                kind='traditional',
                name=model_name,
                path=project_relative_path(bundle_path),
                config_path=project_relative_path(config_path),
                config=runtime_config,
                required_positive_s=required_positive_seconds_for(
                    bundle_path,
                    model_dir,
                    model_name,
                ),
                bundle=bundle,
            )
        )
    return states


def validate_detection_states(detection_states: list[dict]) -> float:
    if not detection_states:
        raise FileNotFoundError(
            "No se encontro ningun modelo valido de deteccion.\n"
            f"Carpeta CNN: {project_relative_path(resolve_cnn_directory())}\n"
            f"Carpeta tradicionales: {project_relative_path(resolve_traditional_directory())}\n"
            "Coloca las CNN en Modelos/CNN y los bundles tradicionales en "
            "Modelos/Tradicionales, cada uno con su JSON de postprocesado."
        )

    max_vote_window_sec = max(
        state['config']['chunk_length_s'] for state in detection_states
    )
    for state in detection_states:
        if state['config']['decision_step_s'] > max_vote_window_sec:
            raise ValueError(
                f"El paso de decision de {state['name']} "
                f"({state['config']['decision_step_s']:.3f}s) supera la ventana "
                f"global de votacion ({max_vote_window_sec:.3f}s)."
            )
    return max_vote_window_sec


def infer_cnn_state(
    state: dict,
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> dict:
    channel = detection_audio_channel(audio_buffer, audio_channel)
    chunk_cnn = audio_buffer[-state['window_samp']:, channel].astype(np.float32)
    runtime_config = state['config']
    chunk_modelo = cnn_runtime.resample_and_pad(
        chunk_cnn,
        orig_sr=SR,
        target_sr=runtime_config['sample_rate'],
        target_length=runtime_config['chunk_samples'],
    )
    features_cnn = cnn_runtime.extract_features_from_array(
        chunk_modelo,
        runtime_config,
    )
    # Las CNN entrenadas en este proyecto devuelven una salida escalar p(siren).
    probability = cnn_runtime.predict_chunk_probability(state['model'], features_cnn)
    return {
        'probability': probability,
        'positive': probability >= runtime_config['chunk_threshold'],
        'label': 'siren' if probability >= runtime_config['chunk_threshold'] else 'background',
    }


def infer_traditional_state(
    state: dict,
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> dict:
    channel = detection_audio_channel(audio_buffer, audio_channel)
    chunk_trad = audio_buffer[-state['window_samp']:, channel].astype(np.float32)
    runtime_config = state['config']
    chunk_modelo = cnn_runtime.resample_and_pad(
        chunk_trad,
        orig_sr=SR,
        target_sr=runtime_config['sample_rate'],
        target_length=runtime_config['chunk_samples'],
    )
    features = trad_runtime.extract_feature_vector(
        chunk_modelo,
        sr=runtime_config['sample_rate'],
    )
    return trad_runtime.predict_traditional(
        state['bundle'],
        features,
        runtime_config['chunk_threshold'],
    )


def infer_detection_state(
    state: dict,
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> dict:
    if state['kind'] == 'cnn':
        return infer_cnn_state(state, audio_buffer, audio_channel)
    if state['kind'] == 'traditional':
        return infer_traditional_state(state, audio_buffer, audio_channel)
    raise ValueError(f"Tipo de modelo de deteccion no soportado: {state['kind']}")


def apply_positive_streak(state: dict, result: dict) -> dict:
    qualified_result = dict(result)
    raw_positive = bool(result.get('positive', False))

    if raw_positive:
        if state['positive_streak_s'] <= 0:
            state['positive_streak_s'] = float(state['config']['chunk_length_s'])
        else:
            state['positive_streak_s'] += float(state['config']['decision_step_s'])
    else:
        state['positive_streak_s'] = 0.0

    positive_streak_s = float(state['positive_streak_s'])
    required_positive_s = float(state['required_positive_s'])
    qualified_positive = raw_positive and positive_streak_s >= required_positive_s

    qualified_result['raw_positive'] = raw_positive
    qualified_result['positive'] = qualified_positive
    qualified_result['positive_streak_s'] = positive_streak_s
    qualified_result['required_positive_s'] = required_positive_s
    qualified_result['raw_label'] = result.get('label', 'siren' if raw_positive else 'background')
    qualified_result['label'] = 'siren' if qualified_positive else 'background'
    return qualified_result


def run_detection_state(
    state: dict,
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> dict:
    return apply_positive_streak(
        state,
        infer_detection_state(state, audio_buffer, audio_channel),
    )


def update_detection_models(
    detection_states: list[dict],
    audio_buffer: np.ndarray,
    n_samples: int,
    audio_channel: int = 0,
) -> None:
    for state in detection_states:
        state['accumulator'] += n_samples
        if state['accumulator'] >= state['step_samp']:
            state['accumulator'] -= state['step_samp']
            t_inicio = time.time()
            result = run_detection_state(state, audio_buffer, audio_channel)
            state['period_latency_ms'] += (time.time() - t_inicio) * 1000
            state['period_results'].append(result)


def select_period_result(
    state: dict,
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> dict:
    if not state['period_results']:
        t_inicio = time.time()
        result = run_detection_state(state, audio_buffer, audio_channel)
        state['period_latency_ms'] += (time.time() - t_inicio) * 1000
        state['period_results'].append(result)

    qualified_results = [
        result for result in state['period_results']
        if result.get('positive', False)
    ]
    numeric_qualified_results = [
        result for result in qualified_results
        if result.get('probability') is not None
    ]
    if numeric_qualified_results:
        return max(numeric_qualified_results, key=lambda item: item['probability'])
    if qualified_results:
        return max(
            qualified_results,
            key=lambda item: item.get('positive_streak_s', 0.0),
        )

    numeric_results = [
        result for result in state['period_results']
        if result.get('probability') is not None
    ]
    if numeric_results:
        return max(numeric_results, key=lambda item: item['probability'])

    return max(
        state['period_results'],
        key=lambda item: item.get('positive_streak_s', 0.0),
    )


def format_vote_part(state: dict, result: dict) -> str:
    sign = '+' if result['positive'] else '-'
    probability = result.get('probability')
    streak = result.get('positive_streak_s', 0.0)
    required = result.get('required_positive_s', state.get('required_positive_s', 1.0))
    streak_text = f"{streak:.2f}/{required:.2f}s"
    if probability is None:
        raw_label = result.get('raw_label', result.get('label', 'background'))
        return f"{state['name']}={raw_label}[{streak_text}]({sign})"
    return f"{state['name']}={probability*100:.1f}%[{streak_text}]({sign})"


def vote_detection_models(
    detection_states: list[dict],
    audio_buffer: np.ndarray,
    audio_channel: int = 0,
) -> tuple[bool, float, float, str]:
    positive_votes = 0
    positive_probs = []
    binary_positive_votes = 0
    detail_parts = []
    total_latency_ms = 0.0

    for state in detection_states:
        result = select_period_result(state, audio_buffer, audio_channel)
        if result['positive']:
            positive_votes += 1
            if result.get('probability') is not None:
                positive_probs.append(float(result['probability']))
            else:
                binary_positive_votes += 1

        detail_parts.append(format_vote_part(state, result))
        total_latency_ms += state['period_latency_ms']
        state['period_results'].clear()
        state['period_latency_ms'] = 0.0

    total_models = len(detection_states)
    final_detection = positive_votes * 2 >= total_models
    final_probability = (
        float(sum(positive_probs) / len(positive_probs))
        if positive_probs
        else (1.0 if binary_positive_votes else 0.0)
    )
    detail = " | ".join(detail_parts)
    detail = (
        f"{detail} => votos {positive_votes}/{total_models} => "
        f"{'SIRENA' if final_detection else 'Ruido'} "
        f"(prob+ media={final_probability*100:.1f}%)"
    )
    return final_detection, final_probability, total_latency_ms, detail
