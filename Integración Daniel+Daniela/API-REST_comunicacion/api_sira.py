import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from collections import deque
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SIRA Core API - Final")

# Permitir conexiones desde el display (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MODELOS DE DATOS ---

class DatosDeteccion(BaseModel):
    sirena: bool
    probabilidad: float
    tipo_vehiculo: str
    latencia_inferencia_ms: float
    fps: float
    t0_captura: float 

class DatosDOA(BaseModel):
    angulo: int
    tendencia: str

class DatosAudio(BaseModel):
    waveform_summary: List[float]
    fft_data: List[float]
    mfcc_features: List[float]

# --- 2. ESTADO GLOBAL Y BUFFERS ---
historial_prob = deque(maxlen=150) # 15 segs a 10 FPS
logs_eventos = deque(maxlen=10)

estado_actual = {
    "deteccion": {
        "sirena": False, "probabilidad": 0.0, "tipo_vehiculo": "Ninguno",
        "latencia_inferencia_ms": 0.0, "fps": 0.0, "t0_captura": 0.0
    },
    "doa": {"angulo": 0, "tendencia": "Estable"},
    "audio": {"waveform_summary": [], "mfcc_features": []},
    "metricas_modelo": {
        "accuracy": 0.9457, 
        "f1_score": 0.8981,
        "confusion": {"TP": 4342, "TN": 12812, "FP": 533, "FN": 452}
    },
    "config": {"threshold": 0.65},
    "ultima_conexion_micro": time.time()
}

# --- 3. LÓGICA DE SEGURIDAD (WATCHDOG) ---

async def watchdog_sira():
    while True:
        await asyncio.sleep(1)
        tiempo_sin_datos = time.time() - estado_actual["ultima_conexion_micro"]
        if tiempo_sin_datos > 2.0 and estado_actual["deteccion"]["probabilidad"] > 0.0:
            estado_actual["deteccion"]["probabilidad"] = 0.0
            estado_actual["deteccion"]["sirena"] = False
            estado_actual["deteccion"]["tipo_vehiculo"] = "PERDIDA_SENAL"
            logs_eventos.appendleft(f"[{time.strftime('%H:%M:%S')}] SYS: Timeout de datos")

@app.on_event("startup")
async def startup():
    asyncio.create_task(watchdog_sira())

# --- 4. ENDPOINTS ---

@app.get("/estado_completo")
async def obtener_estado():
    # Fusionamos buffers en el retorno
    return {
        **estado_actual,
        "historial_15s": list(historial_prob),
        "logs": list(logs_eventos),
        "server_time": time.time()
    }

@app.post("/update_deteccion")
async def update_deteccion(datos: DatosDeteccion):
    estado_actual["ultima_conexion_micro"] = time.time()
    estado_actual["deteccion"] = datos.model_dump()
    historial_prob.append(datos.probabilidad)
    if datos.sirena:
        log = f"[{time.strftime('%H:%M:%S')}] ALERT: {datos.tipo_vehiculo}"
        if not logs_eventos or log != logs_eventos[0]:
            logs_eventos.appendleft(log)
    return {"status": "ok"}

@app.post("/update_doa")
async def update_doa(datos: DatosDOA):
    estado_actual["doa"] = datos.model_dump()
    return {"status": "ok"}

@app.post("/update_audio")
async def update_audio(datos: DatosAudio):
    estado_actual["audio"] = datos.model_dump()
    return {"status": "ok"}