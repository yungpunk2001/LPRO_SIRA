import time
import threading

import librosa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.gridspec import GridSpec

import config_doa as cdoa
import detection_manager
from doa_music import doa_music
from doa_tracker_single import DOATrackerSingle


SR = cdoa.AUDIO['rate']
FMAX_VISUAL = 8000


# =========================================================
# 1. CONFIGURACION DEL SISTEMA Y RELOJES MULTITASA
# =========================================================
CNN_MODEL_DIR = detection_manager.resolve_cnn_directory()
TRAD_MODEL_DIR = detection_manager.resolve_traditional_directory()

print(">>> [1/4] Cargando modelos de deteccion...")
cnn_states = detection_manager.load_cnn_models(CNN_MODEL_DIR)
traditional_states = detection_manager.load_traditional_models(TRAD_MODEL_DIR)
detection_states = cnn_states + traditional_states

VOTE_WINDOW_SEC = detection_manager.validate_detection_states(detection_states)
VOTE_STEP_SAMP = int(round(VOTE_WINDOW_SEC * SR))
if VOTE_STEP_SAMP <= 0:
    raise ValueError("La ventana global de votacion de deteccion no es valida.")

print(f"    CNN: {len(cnn_states)} | Tradicionales: {len(traditional_states)}")
print(f"    Ventana de votacion: {VOTE_WINDOW_SEC:.3f}s")
required_positive_values = sorted(
    {round(state['required_positive_s'], 6) for state in detection_states}
)
if len(required_positive_values) == 1:
    print(f"    Racha positiva requerida: {required_positive_values[0]:.3f}s")
else:
    print(
        "    Rachas positivas requeridas: "
        + ", ".join(f"{value:.3f}s" for value in required_positive_values)
    )


def list_input_devices(audio_interface):
    devices = []
    for index in range(audio_interface.get_device_count()):
        info = audio_interface.get_device_info_by_index(index)
        max_channels = int(info.get('maxInputChannels', 0))
        if max_channels > 0:
            devices.append((index, info, max_channels))
    return devices


def choose_input_device_dialog(input_devices):
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        print(f"[AUDIO] No se pudo abrir dialogo grafico: {exc}")
        return choose_input_device_console(input_devices)

    selected = {'device': None}
    labels = [
        f"ID {index}: {info['name']} ({channels} canales)"
        for index, info, channels in input_devices
    ]

    def accept_selection():
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "Seleccion de microfono",
                "Selecciona un microfono de entrada.",
            )
            return
        selected['device'] = input_devices[int(selection[0])]
        root.destroy()

    def cancel_selection():
        root.destroy()

    root = tk.Tk()
    root.title("Seleccionar microfono")
    root.geometry("680x320")
    root.resizable(True, True)
    root.protocol("WM_DELETE_WINDOW", cancel_selection)

    label = tk.Label(
        root,
        text=(
            "No se encontro ReSpeaker. "
            "Selecciona otro microfono para depuracion local."
        ),
        anchor="w",
    )
    label.pack(fill="x", padx=12, pady=(12, 6))

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12, pady=6)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    for label_text in labels:
        listbox.insert(tk.END, label_text)
    if labels:
        listbox.selection_set(0)

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=(6, 12))
    tk.Button(buttons, text="Usar microfono", command=accept_selection).pack(
        side="right",
        padx=(6, 0),
    )
    tk.Button(buttons, text="Cancelar", command=cancel_selection).pack(side="right")

    root.mainloop()
    return selected['device']


def choose_input_device_console(input_devices):
    print("\nMicrofonos de entrada disponibles:")
    for option, (index, info, channels) in enumerate(input_devices, start=1):
        print(f"  {option}. ID {index}: {info['name']} ({channels} canales)")
    try:
        choice = int(input("Selecciona microfono para depuracion local: ").strip())
    except Exception:
        return None
    if choice < 1 or choice > len(input_devices):
        return None
    return input_devices[choice - 1]


def choose_input_device(audio_interface):
    input_devices = list_input_devices(audio_interface)
    if not input_devices:
        return None

    for device in input_devices:
        index, info, _channels = device
        if "ReSpeaker" in info['name']:
            print(f"    ReSpeaker detectado en ID {index}: {info['name']}")
            return device

    print("    ReSpeaker no detectado. Abriendo selector de microfonos...")
    return choose_input_device_dialog(input_devices)


DOA_WINDOW_SEC = cdoa.AUDIO['t_frame']
DOA_WINDOW_SAMP = int(DOA_WINDOW_SEC * SR)
DOA_STEP_SAMP = DOA_WINDOW_SAMP

VISUAL_STEP_SAMP = int(0.05 * SR)  # 20 FPS
VISUAL_TIME_SEC = 5.0
VISUAL_FRAMES = int(VISUAL_TIME_SEC / 0.05)
N_FFT_VISUAL = 512

MAX_DETECTION_WINDOW_SAMP = max(state['window_samp'] for state in detection_states)
MAX_BUFFER = max(
    MAX_DETECTION_WINDOW_SAMP,
    DOA_WINDOW_SAMP,
    VOTE_STEP_SAMP,
    int(1.0 * SR),
    N_FFT_VISUAL,
)
MICRO_CHUNK_SAMP = int(0.025 * SR)
DETECTION_TICK_SAMP = MICRO_CHUNK_SAMP


# =========================================================
# 2. INICIALIZACION DE MODELOS DOA Y HARDWARE
# =========================================================
print(">>> [2/4] Inicializando Motor DoA...")
tracker = DOATrackerSingle(cdoa.TRACKER)
phi_mics = np.arange(4) * (2 * np.pi / 4)
micpos = np.column_stack(
    (
        cdoa.AUDIO['radius_m'] * np.cos(phi_mics),
        cdoa.AUDIO['radius_m'] * np.sin(phi_mics),
    )
)

print(">>> [3/4] Configurando PyAudio...")
audio = pyaudio.PyAudio()
selected_device = choose_input_device(audio)
if selected_device is None:
    print("\nERROR: No se selecciono ningun microfono de entrada.")
    audio.terminate()
    exit()

dev_index, dev_info, dev_max_channels = selected_device
INPUT_CHANNELS = max(1, min(cdoa.AUDIO['channels_hw'], int(dev_max_channels)))
DOA_AVAILABLE = INPUT_CHANNELS >= 4
if DOA_AVAILABLE:
    if INPUT_CHANNELS >= 5:
        DOA_CHANNEL_SLICE = slice(1, 5)
        print("    DoA activo usando canales 1:5.")
    else:
        DOA_CHANNEL_SLICE = slice(0, 4)
        print("    DoA activo usando canales 0:4.")
else:
    DOA_CHANNEL_SLICE = None
    print(
        "    DoA desactivado: el microfono seleccionado tiene "
        f"{INPUT_CHANNELS} canal(es)."
    )

stream = audio.open(
    format=pyaudio.paInt16,
    channels=INPUT_CHANNELS,
    rate=SR,
    input=True,
    input_device_index=dev_index,
    frames_per_buffer=MICRO_CHUNK_SAMP,
)

freqs_visual = np.fft.rfftfreq(N_FFT_VISUAL, d=1 / SR)
idx_fmax = min(len(freqs_visual), np.searchsorted(freqs_visual, FMAX_VISUAL, side='right'))

print(">>> [4/4] Dashboard configurado.")


# =========================================================
# 3. VARIABLES COMPARTIDAS
# =========================================================
data_lock = threading.Lock()
ui_lock = threading.Lock()
stop_event = threading.Event()

buffer_audio = np.zeros((MAX_BUFFER, INPUT_CHANNELS), dtype=np.float32)

muestras_totales = 0
procesado_doa = 0
procesado_visual = 0
procesado_deteccion = 0
procesado_votacion = 0
t_interno = 0.0

ui_wave = np.zeros(int(SR / 10), dtype=np.float32)
ui_angulos = []
ui_P_music = np.zeros(720, dtype=np.float32)
ui_theta_scan = np.arange(0, 360, 0.5)
ui_fft_y = np.full(idx_fmax, -110.0, dtype=np.float32)
ui_spec = np.full((idx_fmax, VISUAL_FRAMES), -110.0, dtype=np.float32)
ui_prob = 0.0
ui_sirena = False
ui_detalle = "Inicializando deteccion"

hist_discarded_t, hist_discarded_ang = [], []
hist_low_t, hist_low_ang = [], []
hist_max_t, hist_max_ang = [], []
hist_track_t, hist_track_ang = [], []


def snapshot_audio():
    with data_lock:
        return buffer_audio.copy(), muestras_totales, t_interno


def limpiar_historial(l_t, l_ang, t_actual):
    while len(l_t) > 0 and l_t[0] < t_actual - 20.0:
        l_t.pop(0)
        l_ang.pop(0)


def resumen_votacion(detalle_votacion: str) -> str:
    partes = [parte.strip() for parte in detalle_votacion.split("=>")]
    if len(partes) >= 3:
        return f"{partes[-2]} => {partes[-1]}"
    return detalle_votacion


# =========================================================
# 4. HILOS DE EJECUCION
# =========================================================
def hilo_captura():
    global buffer_audio, muestras_totales, t_interno

    while not stop_event.is_set():
        try:
            data = stream.read(MICRO_CHUNK_SAMP, exception_on_overflow=False)
            nuevo_bloque = (
                np.frombuffer(data, dtype=np.int16)
                .reshape(-1, INPUT_CHANNELS)
                .astype(np.float32)
                / 32768.0
            )
            n_nuevos = len(nuevo_bloque)

            with data_lock:
                buffer_audio = np.roll(buffer_audio, -n_nuevos, axis=0)
                buffer_audio[-n_nuevos:, :] = nuevo_bloque
                muestras_totales += n_nuevos
                t_interno += n_nuevos / SR
        except Exception as exc:
            print(f"[CAPTURA] Error leyendo tarjeta de sonido: {exc}")
            time.sleep(0.01)


def hilo_doa_y_visuales():
    global procesado_doa, procesado_visual
    global ui_wave, ui_angulos, ui_P_music, ui_theta_scan, ui_fft_y, ui_spec
    global hist_discarded_t, hist_discarded_ang
    global hist_low_t, hist_low_ang, hist_max_t, hist_max_ang
    global hist_track_t, hist_track_ang

    while not stop_event.is_set():
        _, total_muestras, _ = snapshot_audio()
        trabajo = False

        if total_muestras - procesado_visual >= VISUAL_STEP_SAMP:
            procesado_visual += VISUAL_STEP_SAMP
            trabajo = True
            audio_local, _, _ = snapshot_audio()
            try:
                wave = audio_local[-int(SR):, 0][::10]

                chunk_visual = audio_local[-N_FFT_VISUAL:, 0]
                ventana = np.hanning(len(chunk_visual))
                fft_complex = np.fft.rfft(chunk_visual * ventana, n=N_FFT_VISUAL)
                fft_mag = np.abs(fft_complex) / (N_FFT_VISUAL / 2.0)
                fft_db = librosa.amplitude_to_db(fft_mag, ref=1.0, top_db=120)[:idx_fmax]

                with ui_lock:
                    ui_wave = wave
                    ui_fft_y = fft_db
                    ui_spec = np.roll(ui_spec, -1, axis=1)
                    ui_spec[:, -1] = fft_db
            except Exception as exc:
                print(f"[VISUAL] Error FFT: {exc}")

        if total_muestras - procesado_doa >= DOA_STEP_SAMP:
            procesado_doa += DOA_STEP_SAMP
            trabajo = True
            if not DOA_AVAILABLE:
                tracker.actualizar([], [])
                with ui_lock:
                    ui_P_music = np.zeros_like(ui_P_music)
                    ui_angulos = []
                continue

            audio_local, _, t_actual = snapshot_audio()
            try:
                mics_raw = audio_local[-DOA_WINDOW_SAMP:, DOA_CHANNEL_SLICE].astype(
                    np.float64
                )
                mics_norm = mics_raw - np.mean(mics_raw, axis=0)
                mics_norm = mics_norm / (
                    np.max(np.abs(mics_norm), axis=0) + np.finfo(float).eps
                )

                theta_est, conf_est, P_mus, t_scan = doa_music(mics_norm, micpos, SR)
                ang_validos = tracker.actualizar(theta_est, conf_est)

                with ui_lock:
                    ui_P_music = np.asarray(P_mus)
                    ui_theta_scan = np.asarray(t_scan)
                    ui_angulos = list(ang_validos)

                    if len(theta_est) > 0:
                        max_conf_idx = int(np.argmax(conf_est))
                        for p, theta in enumerate(theta_est):
                            if conf_est[p] < cdoa.TRACKER['conf_keep']:
                                hist_discarded_t.append(t_actual)
                                hist_discarded_ang.append(theta)
                            elif p == max_conf_idx:
                                hist_max_t.append(t_actual)
                                hist_max_ang.append(theta)
                            else:
                                hist_low_t.append(t_actual)
                                hist_low_ang.append(theta)

                    if len(ui_angulos) > 0:
                        if len(hist_track_ang) > 0 and not np.isnan(hist_track_ang[-1]):
                            salto = min(
                                abs(ui_angulos[0] - hist_track_ang[-1]),
                                360 - abs(ui_angulos[0] - hist_track_ang[-1]),
                            )
                            if salto > cdoa.TRACKER['gate_deg']:
                                hist_track_t.append(t_actual - 0.001)
                                hist_track_ang.append(np.nan)
                        hist_track_t.append(t_actual)
                        hist_track_ang.append(ui_angulos[0])
                    else:
                        if len(hist_track_ang) > 0 and not np.isnan(hist_track_ang[-1]):
                            hist_track_t.append(t_actual)
                            hist_track_ang.append(np.nan)

                    limpiar_historial(hist_discarded_t, hist_discarded_ang, t_actual)
                    limpiar_historial(hist_low_t, hist_low_ang, t_actual)
                    limpiar_historial(hist_max_t, hist_max_ang, t_actual)
                    limpiar_historial(hist_track_t, hist_track_ang, t_actual)
            except Exception as exc:
                print(f"[DOA] Error: {exc}")

        if not trabajo:
            time.sleep(0.005)


def hilo_deteccion():
    global procesado_deteccion, procesado_votacion
    global ui_prob, ui_sirena, ui_detalle

    while not stop_event.is_set():
        _, total_muestras, _ = snapshot_audio()
        trabajo = False

        while total_muestras - procesado_deteccion >= DETECTION_TICK_SAMP:
            procesado_deteccion += DETECTION_TICK_SAMP
            audio_local, _, _ = snapshot_audio()
            try:
                detection_manager.update_detection_models(
                    detection_states,
                    audio_local,
                    DETECTION_TICK_SAMP,
                )
            except Exception as exc:
                print(f"[DETECCION] Error actualizando modelos: {exc}")
            trabajo = True

        if total_muestras - procesado_votacion >= VOTE_STEP_SAMP:
            procesado_votacion += VOTE_STEP_SAMP
            audio_local, _, _ = snapshot_audio()
            try:
                sirena, prob, latencia_ms, detalle = detection_manager.vote_detection_models(
                    detection_states,
                    audio_local,
                )
                resumen = resumen_votacion(detalle)
                with ui_lock:
                    ui_sirena = bool(sirena)
                    ui_prob = float(prob)
                    ui_detalle = resumen
                print(f"[DET] {resumen} | latencia={latencia_ms:.1f} ms")
            except Exception as exc:
                print(f"[DETECCION] Error votando modelos: {exc}")
            trabajo = True

        if not trabajo:
            time.sleep(0.01)


threading.Thread(target=hilo_captura, daemon=True).start()
threading.Thread(target=hilo_doa_y_visuales, daemon=True).start()
threading.Thread(target=hilo_deteccion, daemon=True).start()


# =========================================================
# 5. HILO PRINCIPAL: INTERFAZ GRAFICA MATPLOTLIB
# =========================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9))
fig.canvas.manager.set_window_title('SIRA - Dashboard Local Debug Martin')
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.5, 1, 1])

ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_ylim(0, 1)
ax_polar.set_yticks([])
ax_polar.set_title("Radar DoA (Continuo)", color='#00f3ff')
puntos_plot, = ax_polar.plot(
    [],
    [],
    'ro',
    markersize=15,
    alpha=0.9,
    markeredgecolor='black',
)

ax_spec = fig.add_subplot(gs[0, 1])
ax_spec.set_xlim(0, 360)
ax_spec.set_ylim(0, 1.05)
ax_spec.set_title("Pseudo-Espectro MUSIC", color='#00f3ff')
ax_spec.grid(True, linestyle='--', alpha=0.3)
line_spec, = ax_spec.plot([], [], color='#00f3ff', linewidth=2)

ax_alert = fig.add_subplot(gs[0, 2])
ax_alert.axis('off')
texto_alerta = ax_alert.text(
    0.5,
    0.62,
    "INICIANDO...",
    ha='center',
    va='center',
    fontsize=24,
    weight='bold',
)
texto_prob = ax_alert.text(0.5, 0.38, "0.0%", ha='center', va='center', fontsize=18)
texto_detalle = ax_alert.text(
    0.5,
    0.18,
    "",
    ha='center',
    va='center',
    fontsize=10,
    wrap=True,
)

ax_hist = fig.add_subplot(gs[1, 0:2])
ax_hist.set_xlim(-20.0, 0)
ax_hist.set_ylim(0, 360)
ax_hist.set_yticks(np.arange(0, 361, 90))
ax_hist.set_title("Historial de Tracking (20s)")
ax_hist.grid(True, linestyle='--', alpha=0.3)
line_hist_discarded, = ax_hist.plot([], [], 'ko', markersize=4, alpha=0.4)
line_hist_low, = ax_hist.plot([], [], 'o', color='lightcoral', markersize=4, alpha=0.5)
line_hist_max, = ax_hist.plot([], [], 'go', markersize=6, alpha=0.8)
line_hist_track, = ax_hist.plot([], [], 'b-', linewidth=2.5)

ax_wave = fig.add_subplot(gs[1, 2])
ax_wave.set_xlim(0, 1.0)
ax_wave.set_ylim(-1, 1)
ax_wave.set_title("Forma de Onda (1 Seg)")
ax_wave.set_xticks([])
x_wave = np.linspace(0, 1.0, int(SR / 10))
line_wave, = ax_wave.plot(x_wave, np.zeros_like(x_wave), color='#ff003c', linewidth=1)

ax_cnn_spec = fig.add_subplot(gs[2, 0:2])
ax_cnn_spec.set_title(f"Espectrograma Visual ({VISUAL_TIME_SEC}s)", color='yellow')
ax_cnn_spec.set_ylabel("Frecuencia (Hz)")
ax_cnn_spec.set_xlabel("Tiempo Relativo (s)")
img_spec = ax_cnn_spec.imshow(
    np.zeros((idx_fmax, VISUAL_FRAMES)),
    aspect='auto',
    origin='lower',
    cmap='magma',
    vmin=-110,
    vmax=10,
    extent=[-VISUAL_TIME_SEC, 0, 0, FMAX_VISUAL],
)

ax_fft = fig.add_subplot(gs[2, 2])
ax_fft.set_title("Espectro FFT Instantaneo", color='yellow')
ax_fft.set_xlabel("Frecuencia (Hz)")
ax_fft.set_ylabel("Amplitud (dB)")
ax_fft.set_xlim(0, FMAX_VISUAL)
ax_fft.set_ylim(-110, 10)
ax_fft.grid(True, linestyle='--', alpha=0.3)
line_fft, = ax_fft.plot([], [], color='yellow', linewidth=1.5)

plt.tight_layout()


def refrescar_pantalla(_frame):
    try:
        with data_lock:
            t_actual = t_interno

        with ui_lock:
            wave = np.asarray(ui_wave).copy()
            theta_scan = np.asarray(ui_theta_scan).copy()
            P_music = np.asarray(ui_P_music).copy()
            angulos = list(ui_angulos)
            fft_y = np.asarray(ui_fft_y).copy()
            spec = np.asarray(ui_spec).copy()
            prob = float(ui_prob)
            sirena = bool(ui_sirena)
            detalle = str(ui_detalle)
            discarded_t = list(hist_discarded_t)
            discarded_ang = list(hist_discarded_ang)
            low_t = list(hist_low_t)
            low_ang = list(hist_low_ang)
            max_t = list(hist_max_t)
            max_ang = list(hist_max_ang)
            track_t = list(hist_track_t)
            track_ang = list(hist_track_ang)

        line_wave.set_ydata(wave)
        line_spec.set_data(theta_scan, P_music)

        if len(angulos) > 0:
            puntos_plot.set_data(np.radians(angulos), np.ones(len(angulos)) * 0.8)
        else:
            puntos_plot.set_data([], [])

        def dibujar_hist(linea, l_t, l_ang):
            linea.set_data([t - t_actual for t in l_t], l_ang)

        dibujar_hist(line_hist_discarded, discarded_t, discarded_ang)
        dibujar_hist(line_hist_low, low_t, low_ang)
        dibujar_hist(line_hist_max, max_t, max_ang)
        dibujar_hist(line_hist_track, track_t, track_ang)

        line_fft.set_data(freqs_visual[:idx_fmax], fft_y)
        img_spec.set_data(spec)

        texto_prob.set_text(f"{prob*100:.1f}%")
        texto_detalle.set_text(detalle)
        if sirena:
            ax_alert.set_facecolor('#4a0000')
            texto_alerta.set_text("SIRENA")
            texto_alerta.set_color('#ff4444')
            texto_prob.set_color('#ff4444')
            texto_detalle.set_color('#ffaaaa')
        else:
            ax_alert.set_facecolor('#002200')
            texto_alerta.set_text("RUIDO")
            texto_alerta.set_color('#44ff44')
            texto_prob.set_color('#44ff44')
            texto_detalle.set_color('#aaffaa')
    except Exception as exc:
        print(f"[UI] Error dibujando interfaz: {exc}")


print("\nINICIANDO DASHBOARD LOCAL DEBUG...")
ani = animation.FuncAnimation(
    fig,
    refrescar_pantalla,
    interval=33,
    blit=False,
    cache_frame_data=False,
)
plt.show()

print("\n>>> Deteniendo dashboard...")
stop_event.set()
stream.stop_stream()
stream.close()
audio.terminate()
