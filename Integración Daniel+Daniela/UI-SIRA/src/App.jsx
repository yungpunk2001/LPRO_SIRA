import React, { useEffect, useState } from 'react';
import { BarChart3, ShieldAlert, Smartphone, Volume2, Zap } from 'lucide-react';

import logoSira from './assets/SIRA_LOGO_SIN_FONDO.png';

const API_URL = 'http://localhost:8000/estado_completo';
const DEFAULT_WAVEFORM = Array(100).fill(0);
const DEFAULT_HISTORY = Array(50).fill(0);
const DEFAULT_DATA = {
  deteccion: {
    sirena: false,
    probabilidad: 0,
    tipo_vehiculo: 'Ninguno',
    latencia_inferencia_ms: 0,
    fps: 0,
    t0_captura: 0,
  },
  doa: { angulo: 0, tendencia: 'Estable' },
  audio: { waveform_summary: DEFAULT_WAVEFORM, fft_data: [], mfcc_features: [] },
  metricas_modelo: {
    accuracy: 0,
    f1_score: 0,
    confusion: { TP: 0, TN: 0, FP: 0, FN: 0 },
  },
  config: { threshold: 0 },
  historial_15s: DEFAULT_HISTORY,
  logs: ['Esperando datos de la API...'],
};

const normalizeApiData = (apiData = {}) => {
  const waveformSummary = Array.isArray(apiData.audio?.waveform_summary)
    ? apiData.audio.waveform_summary
    : [];
  const historial = Array.isArray(apiData.historial_15s) ? apiData.historial_15s : [];
  const logs = Array.isArray(apiData.logs) ? apiData.logs : [];

  return {
    ...DEFAULT_DATA,
    ...apiData,
    deteccion: {
      ...DEFAULT_DATA.deteccion,
      ...apiData.deteccion,
    },
    doa: {
      ...DEFAULT_DATA.doa,
      ...apiData.doa,
    },
    audio: {
      ...DEFAULT_DATA.audio,
      ...apiData.audio,
      waveform_summary: waveformSummary.length ? waveformSummary : DEFAULT_WAVEFORM,
    },
    metricas_modelo: {
      ...DEFAULT_DATA.metricas_modelo,
      ...apiData.metricas_modelo,
      confusion: {
        ...DEFAULT_DATA.metricas_modelo.confusion,
        ...apiData.metricas_modelo?.confusion,
      },
    },
    config: {
      ...DEFAULT_DATA.config,
      ...apiData.config,
    },
    historial_15s: historial.length ? historial : DEFAULT_HISTORY,
    logs: logs.length ? logs : DEFAULT_DATA.logs,
  };
};

const formatRatioAsPercent = (value) => `${(value * 100).toFixed(1)}%`;
const getWaveformPeak = (samples) =>
  samples.length ? Math.max(...samples.map((sample) => Math.abs(sample))) : 0;
const getApiStatusMeta = (status) => {
  if (status === 'connected') {
    return {
      label: 'API conectada',
      dotClass: 'bg-emerald-400 shadow-[0_0_12px_rgba(74,222,128,0.8)]',
      textClass: 'text-emerald-300',
      borderClass: 'border-emerald-400/20',
      bgClass: 'bg-emerald-500/10',
    };
  }

  if (status === 'error') {
    return {
      label: 'API desconectada',
      dotClass: 'bg-red-400 shadow-[0_0_12px_rgba(248,113,113,0.8)]',
      textClass: 'text-red-300',
      borderClass: 'border-red-400/20',
      bgClass: 'bg-red-500/10',
    };
  }

  return {
    label: 'Conectando API',
    dotClass: 'bg-amber-300 shadow-[0_0_12px_rgba(252,211,77,0.8)]',
    textClass: 'text-amber-200',
    borderClass: 'border-amber-300/20',
    bgClass: 'bg-amber-500/10',
  };
};

const GradienteText = ({ children }) => (
  <span className="bg-gradient-to-r from-[#22d3ee] to-[#c084fc] bg-clip-text text-transparent font-black italic">
    {children}
  </span>
);

function App() {
  const [mobileTab, setMobileTab] = useState('radar');
  const [data, setData] = useState(DEFAULT_DATA);
  const [showAllLogs, setShowAllLogs] = useState(false);
  const [apiStatus, setApiStatus] = useState('connecting');
  const [lastApiSync, setLastApiSync] = useState(null);

  const roundedAngle = Math.round(data.doa.angulo);
  const waveformPeak = getWaveformPeak(data.audio.waveform_summary);

  useEffect(() => {
    const fetchDatos = async () => {
      try {
        const response = await fetch(API_URL);
        if (!response.ok) {
          setApiStatus('error');
          return;
        }
        const json = await response.json();
        setData(normalizeApiData(json));
        setApiStatus('connected');
        setLastApiSync(new Date());
      } catch (error) {
        setApiStatus('error');
      }
    };

    fetchDatos();
    const interval = setInterval(fetchDatos, 300);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#020408] p-4 text-white md:p-8">
      <MobileLayout
        data={data}
        roundedAngle={roundedAngle}
        apiStatus={apiStatus}
        lastApiSync={lastApiSync}
        mobileTab={mobileTab}
        setMobileTab={setMobileTab}
        showAllLogs={showAllLogs}
        setShowAllLogs={setShowAllLogs}
      />

      <div className="hidden min-h-screen flex-col gap-6 md:flex">
        <header className="relative flex items-center justify-between overflow-hidden rounded-[2.5rem] border border-white/10 bg-white/5 px-10 py-6 shadow-xl">
          <div className="absolute left-20 top-1/2 h-64 w-64 -translate-y-1/2 rounded-full bg-cyan-500/5 blur-[80px]" />

          <div className="relative z-10 flex items-center gap-8">
            <img
              src={logoSira}
              alt="SIRA Logo"
              className="h-20 w-20 object-contain drop-shadow-[0_0_15px_rgba(34,211,238,0.2)]"
            />

            <div className="flex flex-col justify-center">
              <h1 className="text-4xl font-black leading-none tracking-tighter text-white">
                <GradienteText>SIRA SYSTEM</GradienteText>
              </h1>
              <p className="mt-1 text-[11px] font-medium uppercase tracking-[0.4em] text-slate-400">
                Smart Intelligent Response Assistant
              </p>
              <ApiStatusBadge apiStatus={apiStatus} lastApiSync={lastApiSync} />
            </div>
          </div>

          <div className="relative z-10 flex items-center gap-10">
            <div className="flex w-32 flex-col gap-2">
              <div className="flex items-end justify-between">
                <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  ACCURACY
                </span>
                <span className="font-mono text-xl font-black leading-none text-cyan-400">
                  {formatRatioAsPercent(data.metricas_modelo.accuracy)}
                </span>
              </div>

              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full bg-cyan-500 shadow-[0_0_12px_rgba(34,211,238,0.6)] transition-all duration-500"
                  style={{ width: `${data.metricas_modelo.accuracy * 100}%` }}
                />
              </div>
            </div>

            <div className="flex w-32 flex-col gap-2">
              <div className="flex items-end justify-between">
                <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  F1-SCORE
                </span>
                <span className="font-mono text-xl font-black leading-none text-purple-400">
                  {formatRatioAsPercent(data.metricas_modelo.f1_score)}
                </span>
              </div>

              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full bg-purple-500 shadow-[0_0_12px_rgba(192,132,252,0.6)] transition-all duration-500"
                  style={{ width: `${data.metricas_modelo.f1_score * 100}%` }}
                />
              </div>
            </div>

            <div className="ml-4 border-l border-white/10 pl-8">
              <Stat
                label="LATENCIA"
                value={`${data.deteccion.latencia_inferencia_ms.toFixed(0)}ms`}
              />
            </div>
          </div>
        </header>

        <main className="grid flex-1 grid-cols-12 gap-6">
          <div className="col-span-3 flex flex-col gap-6">
            <Panel titulo="PROBABILIDAD DETECCION">
              <div className="flex flex-col gap-4">
                <div className="flex h-16 items-end gap-1">
                  {Array.from({ length: 20 }).map((_, i) => {
                    const threshold = i / 20;
                    const isActive = data.deteccion.probabilidad > threshold;

                    return (
                      <div
                        key={i}
                        className={`flex-1 rounded-sm transition-all duration-150 ${
                          isActive
                            ? i > 15
                              ? 'bg-red-500 shadow-[0_0_8px_red]'
                              : 'bg-cyan-400 shadow-[0_0_8px_cyan]'
                            : 'bg-white/5'
                        }`}
                        style={{ height: `${40 + i * 3}%` }}
                      />
                    );
                  })}
                </div>

                <div className="flex items-end justify-between">
                  <div>
                    <div className="mb-1 text-[9px] font-bold uppercase text-slate-500">
                      Calculo en Tiempo Real
                    </div>
                    <div
                      className={`font-mono text-4xl font-black italic ${
                        data.deteccion.sirena ? 'text-red-500' : 'text-white'
                      }`}
                    >
                      {(data.deteccion.probabilidad * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div
                    className={`mb-1 rounded-full p-2 ${
                      data.deteccion.sirena ? 'animate-ping bg-red-500' : 'bg-slate-700'
                    }`}
                  >
                    <ShieldAlert size={16} />
                  </div>
                </div>
              </div>
            </Panel>

            <Panel titulo="FRECUENCIA AUDIO">
              <div className="flex h-24 items-center gap-[2px] rounded-xl bg-black/30 px-2">
                {data.audio.waveform_summary.slice(0, 60).map((v, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-full bg-purple-500/30"
                    style={{ height: `${Math.max(15, Math.abs(v) * 100)}%` }}
                  />
                ))}
              </div>
            </Panel>

            <Panel titulo="LOCALIZACION ANGULAR">
              <div className="relative flex h-24 items-center overflow-hidden rounded-xl border border-white/5 bg-black/40">
                <div
                  className="absolute left-1/2 flex items-center gap-8 transition-transform duration-500 ease-out"
                  style={{ transform: `translateX(calc(-${data.doa.angulo}px * 2))` }}
                >
                  {[...Array(13)].map((_, i) => {
                    const deg = i * 30;

                    return (
                      <div key={i} className="flex min-w-[40px] flex-col items-center">
                        <div
                          className={`w-[2px] ${
                            deg % 90 === 0 ? 'h-6 bg-cyan-400' : 'h-4 bg-slate-600'
                          }`}
                        />
                        <span className="mt-1 font-mono text-[10px] text-slate-400">
                          {deg}°
                        </span>
                      </div>
                    );
                  })}
                </div>

                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <div className="z-10 h-full w-[2px] bg-red-500/50 shadow-[0_0_10px_red]" />
                  <div className="absolute top-0 -translate-y-1 transform text-red-500">
                    <div className="h-0 w-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-red-500" />
                  </div>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between px-2">
                <div className="font-mono text-2xl font-black italic text-white">
                  {roundedAngle}°
                  <span className="ml-1 text-xs uppercase text-slate-500"> Azimut</span>
                </div>

                <div className="flex gap-2 text-[10px] font-bold">
                  <span
                    className={data.doa.angulo < 180 ? 'text-cyan-400' : 'text-slate-600'}
                  >
                    IZQ
                  </span>
                  <span
                    className={data.doa.angulo >= 180 ? 'text-cyan-400' : 'text-slate-600'}
                  >
                    DER
                  </span>
                </div>
              </div>
            </Panel>

            <Panel titulo="CAPTURA DE ONDA (REAL-TIME)">
              <div className="relative h-28 overflow-hidden rounded-xl border border-white/10 bg-slate-950">
                <div
                  className="absolute inset-0 opacity-20"
                  style={{
                    backgroundImage:
                      'linear-gradient(#1e293b 1px, transparent 1px), linear-gradient(90deg, #1e293b 1px, transparent 1px)',
                    backgroundSize: '20px 20px',
                  }}
                />

                <svg viewBox="0 0 200 100" className="absolute inset-0 h-full w-full">
                  <path
                    d={`M 0 50 ${data.audio.waveform_summary
                      .slice(0, 40)
                      .map((v, i) => `L ${i * 5} ${50 + v * 40}`)
                      .join(' ')}`}
                    fill="none"
                    stroke={data.deteccion.sirena ? '#ef4444' : '#c084fc'}
                    strokeWidth="1.5"
                    className="transition-all duration-150"
                  />
                </svg>

                <div className="absolute bottom-2 right-2 top-2 w-1 overflow-hidden rounded-full bg-white/5">
                  <div
                    className="absolute bottom-0 w-full bg-purple-500 transition-all duration-300"
                    style={{
                      height: `${waveformPeak * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div className="mt-3 flex items-center gap-3">
                <Volume2
                  size={14}
                  className={data.deteccion.sirena ? 'text-red-500' : 'text-slate-400'}
                />
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                  <div
                    className="h-full animate-pulse bg-gradient-to-r from-purple-500 to-cyan-500"
                    style={{ width: '60%' }}
                  />
                </div>
                <span className="font-mono text-[10px] text-slate-500">RMS</span>
              </div>
            </Panel>
          </div>

          <div className="relative col-span-6 flex items-center justify-center overflow-hidden rounded-[4rem] border border-cyan-500/10 bg-white/[0.03] shadow-[inset_0_1px_0_rgba(255,255,255,0.05),0_0_60px_rgba(34,211,238,0.1)]">
            <div
              className={`absolute inset-0 transition-opacity duration-1000 ${
                data.deteccion.sirena ? 'opacity-30' : 'opacity-0'
              }`}
              style={{ background: 'radial-gradient(circle, #ef4444 0%, transparent 70%)' }}
            />

            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,211,238,0.15),transparent_60%)]" />

            <div className="relative flex h-[560px] w-[560px] items-center justify-center scale-150 transition-transform duration-500">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="absolute rounded-full border border-white/5"
                  style={{
                    inset: `${i * 13}%`,
                    boxShadow: `0 0 15px rgba(34, 211, 238, ${0.05 / i})`,
                  }}
                />
              ))}

              {data.deteccion.sirena &&
                [0, 1, 2].map((waveIndex) => {
                  const radius = 45 - waveIndex * 7;
                  const spread = 15;
                  const startAngle = ((180 - spread) * Math.PI) / 180;
                  const endAngle = ((180 + spread) * Math.PI) / 180;
                  const x1 = 50 + radius * Math.cos(startAngle);
                  const y1 = 50 + radius * Math.sin(startAngle);
                  const x2 = 50 + radius * Math.cos(endAngle);
                  const y2 = 50 + radius * Math.sin(endAngle);

                  return (
                    <div
                      key={waveIndex}
                      className="absolute inset-0"
                      style={{
                        transform: `rotate(${data.doa.angulo - 90}deg)`,
                        transformOrigin: '50% 50%',
                      }}
                    >
                      <svg viewBox="0 0 100 100" className="h-full w-full">
                        <path
                          d={`M ${x1} ${y1} A ${radius} ${radius} 0 0 0 ${x2} ${y2}`}
                          fill="none"
                          stroke="#ef4444"
                          strokeWidth={1.6 + waveIndex * 0.25}
                          strokeLinecap="round"
                          className="animate-[sirenImpact_1.2s_ease-out_infinite]"
                          style={{
                            animationDelay: `${waveIndex * 0.15}s`,
                            filter: 'drop-shadow(0 0 6px #ef4444)',
                            opacity: 0.95 - waveIndex * 0.16,
                          }}
                        />
                      </svg>
                    </div>
                  );
                })}

              <div
                className={`relative z-10 flex h-64 w-64 items-center justify-center rounded-full border-2 transition-all duration-700 ${
                  data.deteccion.sirena
                    ? 'border-red-500 bg-red-950/40 shadow-[0_0_70px_rgba(239,68,68,0.5)]'
                    : 'border-cyan-400/20 bg-slate-900/60 shadow-[0_0_50px_rgba(34,211,238,0.15)]'
                }`}
              >
                <svg
                  viewBox="0 0 60 100"
                  className="h-40 w-40 transition-transform duration-500"
                  style={{
                    filter:
                      'drop-shadow(0 10px 15px rgba(0,0,0,0.5)) drop-shadow(0 0 10px rgba(34,211,238,0.3))',
                  }}
                >
                  <defs>
                    <linearGradient
                      id="cuerpoCocheIntenso"
                      x1="0%"
                      y1="0%"
                      x2="100%"
                      y2="100%"
                    >
                      <stop offset="0%" stopColor="#22d3ee" />
                      <stop offset="100%" stopColor="#c084fc" />
                    </linearGradient>
                  </defs>

                  <rect
                    x="10"
                    y="10"
                    width="40"
                    height="80"
                    rx="14"
                    fill="url(#cuerpoCocheIntenso)"
                  />
                  <rect x="14" y="35" width="32" height="30" rx="6" fill="#0f172a" fillOpacity="0.85" />
                  <path d="M14 35 Q 30 25 46 35 L 44 45 Q 30 38 16 45 Z" fill="white" fillOpacity="0.25" />
                  <path d="M16 60 Q 30 65 44 60 L 42 68 Q 30 72 18 68 Z" fill="white" fillOpacity="0.15" />
                  <rect x="4" y="38" width="6" height="10" rx="2" fill="url(#cuerpoCocheIntenso)" />
                  <rect x="50" y="38" width="6" height="10" rx="2" fill="url(#cuerpoCocheIntenso)" />
                  <rect
                    x="14"
                    y="84"
                    width="10"
                    height="4"
                    rx="1"
                    fill={data.deteccion.sirena ? '#ef4444' : '#1e293b'}
                    className="transition-colors duration-300"
                  />
                  <rect
                    x="36"
                    y="84"
                    width="10"
                    height="4"
                    rx="1"
                    fill={data.deteccion.sirena ? '#ef4444' : '#1e293b'}
                    className="transition-colors duration-300"
                  />
                </svg>

                <div className="absolute -bottom-8 flex min-w-[120px] items-center justify-center rounded-full border border-white/20 bg-slate-900 px-8 py-3 shadow-[0_0_25px_rgba(34,211,238,0.2)]">
                  {data.deteccion.sirena ? (
                    <span className="font-mono text-xl font-black tracking-tighter text-white">
                      {roundedAngle}°
                    </span>
                  ) : (
                    <div className="flex h-5 items-center gap-2.5">
                      <span className="h-2.5 w-2.5 rounded-full bg-cyan-500 animate-[bounce_1s_infinite_0ms] shadow-[0_0_10px_#22d3ee]" />
                      <span className="h-2.5 w-2.5 rounded-full bg-cyan-500 animate-[bounce_1s_infinite_200ms] shadow-[0_0_10px_#22d3ee]" />
                      <span className="h-2.5 w-2.5 rounded-full bg-cyan-500 animate-[bounce_1s_infinite_400ms] shadow-[0_0_10px_#22d3ee]" />
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="col-span-3 flex flex-col gap-6">
            <Panel titulo="FFT - DISTRIBUCION DE FRECUENCIAS">
              <div className="relative flex h-28 items-end gap-[1px] overflow-hidden rounded-xl border border-white/5 bg-black/20 px-1">
                <div className="absolute left-0 top-1/4 z-0 w-full border-t border-dashed border-red-500/30" />

                {data.audio.waveform_summary.map((v, i) => {
                  const barHeight = Math.abs(v) * 100;

                  return (
                    <div
                      key={i}
                      className={`flex-1 transition-all duration-200 ${
                        barHeight > 70 ? 'bg-red-500 shadow-[0_0_5px_red]' : 'bg-cyan-500/40'
                      }`}
                      style={{ height: `${Math.max(5, barHeight)}%` }}
                    />
                  );
                })}
              </div>

              <div className="mt-2 flex justify-between text-[7px] font-bold uppercase tracking-[0.2em] text-slate-500">
                <span>0 Hz</span>
                <span className="text-cyan-400">Pico Sirena: 1.5kHz</span>
                <span>8 kHz</span>
              </div>
            </Panel>

            <Panel titulo="ESPECTROGRAMA TEMPORAL">
              <div className="grid h-28 grid-cols-10 gap-1">
                {Array.from({ length: 80 }).map((_, i) => {
                  const opacity = Math.random() * (data.deteccion.sirena ? 1 : 0.3);

                  return (
                    <div
                      key={i}
                      className="rounded-sm transition-opacity duration-500"
                      style={{
                        backgroundColor: data.deteccion.sirena ? '#ef4444' : '#22d3ee',
                        opacity,
                      }}
                    />
                  );
                })}
              </div>

              <div className="mt-3 flex items-center justify-between">
                <div className="flex gap-1">
                  <div className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                  <span className="text-[9px] font-bold text-slate-400">
                    ANALIZANDO PATRON ARMONICO
                  </span>
                </div>
                <span className="font-mono text-[9px] text-slate-600">HISTORIAL 5s</span>
              </div>
            </Panel>

            <Panel titulo="RENDIMIENTO DEL MODELO (IA)">
              <div className="flex flex-col gap-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="group relative overflow-hidden rounded-2xl border border-white/5 bg-white/5 p-4">
                    <div className="absolute -right-2 -top-2 h-12 w-12 bg-cyan-500/10 blur-xl transition-colors group-hover:bg-cyan-500/20" />
                    <p className="mb-1 text-[8px] font-black uppercase tracking-widest text-slate-500">
                      Accuracy
                    </p>
                    <p className="font-mono text-3xl font-black leading-none text-white">0.982</p>
                    <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-white/10">
                      <div className="h-full w-[98.2%] bg-cyan-500 shadow-[0_0_8px_rgba(34,211,238,0.6)] transition-all duration-1000" />
                    </div>
                  </div>

                  <div className="group relative overflow-hidden rounded-2xl border border-white/5 bg-white/5 p-4">
                    <div className="absolute -right-2 -top-2 h-12 w-12 bg-purple-500/10 blur-xl transition-colors group-hover:bg-purple-500/20" />
                    <p className="mb-1 text-[8px] font-black uppercase tracking-widest text-slate-500">
                      F1-Score
                    </p>
                    <p className="font-mono text-3xl font-black leading-none text-purple-400">
                      0.975
                    </p>
                    <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-white/10">
                      <div className="h-full w-[97.5%] bg-purple-500 shadow-[0_0_8px_rgba(192,132,252,0.6)] transition-all duration-1000" />
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="px-1 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-400">
                    Matriz de Confusion
                  </p>

                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex flex-col gap-1.5">
                      <div className="flex h-12 flex-col items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10">
                        <span className="font-mono text-lg font-black leading-none text-cyan-400">
                          98.2%
                        </span>
                        <span className="text-[6px] font-bold uppercase text-slate-500">
                          True Pos
                        </span>
                      </div>

                      <div className="flex h-12 flex-col items-center justify-center rounded-xl border border-white/5 bg-white/[0.02] opacity-40">
                        <span className="font-mono text-lg font-black leading-none text-red-500">
                          1.8%
                        </span>
                        <span className="text-[6px] font-bold uppercase text-slate-500">
                          False Neg
                        </span>
                      </div>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="flex h-12 flex-col items-center justify-center rounded-xl border border-white/5 bg-white/[0.02] opacity-40">
                        <span className="font-mono text-lg font-black leading-none text-red-500">
                          0.5%
                        </span>
                        <span className="text-[6px] font-bold uppercase text-slate-500">
                          False Pos
                        </span>
                      </div>

                      <div className="flex h-12 flex-col items-center justify-center rounded-xl border border-purple-500/20 bg-purple-500/10">
                        <span className="font-mono text-lg font-black leading-none text-purple-400">
                          99.5%
                        </span>
                        <span className="text-[6px] font-bold uppercase text-slate-500">
                          True Neg
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel titulo="EVENTOS RECIENTES">
              <div
                className={`overflow-hidden transition-all duration-500 ease-in-out ${
                  showAllLogs ? 'h-[300px]' : 'h-[60px]'
                }`}
              >
                <div className="flex flex-col gap-2">
                  <div
                    onClick={() => setShowAllLogs(!showAllLogs)}
                    className="group cursor-pointer"
                  >
                    {data.logs.slice(-1).map((log, i) => {
                      const isAlert = log.includes('ALERT');

                      return (
                        <div
                          key={i}
                          className={`flex items-center gap-3 rounded-xl border p-2.5 transition-all ${
                            isAlert
                              ? 'border-red-500/40 bg-red-500/20'
                              : 'border-cyan-500/20 bg-cyan-500/10 group-hover:border-cyan-500/40'
                          }`}
                        >
                          <div
                            className={`rounded-lg p-1 ${
                              isAlert
                                ? 'bg-red-500/20 text-red-500'
                                : 'bg-cyan-500/20 text-cyan-500'
                            }`}
                          >
                            <Zap
                              size={14}
                              className={showAllLogs ? 'rotate-180 transition-transform' : ''}
                            />
                          </div>

                          <div className="flex flex-1 flex-col">
                            <span
                              className={`text-[9px] font-black ${
                                isAlert ? 'text-red-400' : 'uppercase text-cyan-400'
                              }`}
                            >
                              {isAlert ? 'ULTIMA ALERTA CRITICA' : 'ULTIMO EVENTO'}
                            </span>
                            <span className="w-40 truncate font-mono text-[10px] text-slate-200">
                              {log}
                            </span>
                          </div>

                          <span className="text-[8px] font-bold text-slate-500">
                            {showAllLogs ? 'CERRAR' : 'VER MAS'}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="scrollbar-hide mt-1 flex flex-col gap-2 overflow-y-auto pr-1">
                    {data.logs
                      .slice(-6, -1)
                      .reverse()
                      .map((log, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-3 rounded-lg border border-white/5 bg-white/[0.02] p-2 opacity-60"
                        >
                          <div className="h-1 w-1 rounded-full bg-slate-500" />
                          <span className="truncate font-mono text-[9px] text-slate-400">
                            {log}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>

              {!showAllLogs && (
                <div className="mt-3 flex items-center justify-center gap-2 border-t border-white/5 py-1.5">
                  <div className="h-1 w-1 animate-ping rounded-full bg-cyan-500" />
                  <span className="text-[7px] font-black uppercase tracking-widest text-slate-500">
                    Monitoreo en curso
                  </span>
                </div>
              )}
            </Panel>
          </div>
        </main>

        <style
          dangerouslySetInnerHTML={{
            __html: `
              @keyframes sirenImpact {
                0% { transform: scale(1.3) rotate(var(--tw-rotate)); opacity: 0; }
                50% { opacity: 0.8; stroke-width: 3; }
                100% { transform: scale(0.6) rotate(var(--tw-rotate)); opacity: 0; }
              }
              @keyframes sirenImpactAdv {
                0% { transform: scaleX(1.4) scaleY(1.3) rotate(var(--tw-rotate)); opacity: 0; }
                50% { opacity: 0.9; stroke-width: 4; }
                100% { transform: scaleX(0.4) scaleY(0.7) rotate(var(--tw-rotate)); opacity: 0; }
              }
              @keyframes bounce {
                0%, 100% { transform: scale(1); opacity: 0.3; }
                50% { transform: scale(1.6); opacity: 1; }
              }
              
            `,
          }}
        />
      </div>
    </div>
  );
}

const MobileLayout = ({
  data,
  roundedAngle,
  apiStatus,
  lastApiSync,
  mobileTab,
  setMobileTab,
  showAllLogs,
  setShowAllLogs,
}) => (
  <div className="pb-24 pt-16 md:hidden">
    <div className="mb-4 rounded-[1.75rem] border border-white/10 bg-white/5 px-4 py-4 shadow-xl">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <img src={logoSira} alt="SIRA Logo" className="h-11 w-11 object-contain" />
          <div>
            <p className="text-lg font-black leading-none">
              <GradienteText>SIRA</GradienteText>
            </p>
            <p className="mt-1 text-[10px] uppercase tracking-[0.28em] text-slate-500">
              Event Mobile View
            </p>
            <ApiStatusBadge apiStatus={apiStatus} lastApiSync={lastApiSync} compact />
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-right">
          <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">DoA</p>
          <p className="font-mono text-lg font-black text-white">{roundedAngle}°</p>
        </div>
      </div>
    </div>

    {mobileTab === 'radar' ? (
      <MobileRadarView data={data} roundedAngle={roundedAngle} />
    ) : (
      <MobileStatsView
        data={data}
        roundedAngle={roundedAngle}
        showAllLogs={showAllLogs}
        setShowAllLogs={setShowAllLogs}
      />
    )}

    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-white/10 bg-[#05080f]/95 px-4 py-3 backdrop-blur-xl">
      <div className="mx-auto grid max-w-sm grid-cols-2 gap-3">
        <button
          onClick={() => setMobileTab('radar')}
          className={`flex items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-black transition-all ${
            mobileTab === 'radar'
              ? 'border-cyan-400/30 bg-cyan-500/10 text-cyan-300'
              : 'border-white/10 bg-white/[0.03] text-slate-400'
          }`}
        >
          <Smartphone size={16} />
          <span>Principal</span>
        </button>

        <button
          onClick={() => setMobileTab('stats')}
          className={`flex items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-black transition-all ${
            mobileTab === 'stats'
              ? 'border-cyan-400/30 bg-cyan-500/10 text-cyan-300'
              : 'border-white/10 bg-white/[0.03] text-slate-400'
          }`}
        >
          <BarChart3 size={16} />
          <span>Stats</span>
        </button>
      </div>
    </nav>
  </div>
);

const MobileRadarView = ({ data, roundedAngle }) => (
  <main className="rounded-[2rem] border border-cyan-500/10 bg-white/[0.04] px-2 py-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.05),0_0_35px_rgba(34,211,238,0.06)]">
    <div className="relative flex min-h-[70vh] items-center justify-center overflow-hidden rounded-[1.75rem]">
      <div
        className={`absolute inset-0 transition-opacity duration-1000 ${
          data.deteccion.sirena ? 'opacity-30' : 'opacity-0'
        }`}
        style={{ background: 'radial-gradient(circle, #ef4444 0%, transparent 70%)' }}
      />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,211,238,0.15),transparent_60%)]" />

      <div className="relative flex h-[320px] w-[320px] items-center justify-center">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="absolute rounded-full border border-white/5"
            style={{
              inset: `${i * 13}%`,
              boxShadow: `0 0 15px rgba(34, 211, 238, ${0.05 / i})`,
            }}
          />
        ))}

        {data.deteccion.sirena &&
          [0, 1, 2].map((waveIndex) => {
            const radius = 45 - waveIndex * 7;
            const spread = 15;
            const startAngle = ((180 - spread) * Math.PI) / 180;
            const endAngle = ((180 + spread) * Math.PI) / 180;
            const x1 = 50 + radius * Math.cos(startAngle);
            const y1 = 50 + radius * Math.sin(startAngle);
            const x2 = 50 + radius * Math.cos(endAngle);
            const y2 = 50 + radius * Math.sin(endAngle);

            return (
              <div
                key={waveIndex}
                className="absolute inset-0"
                style={{
                  transform: `rotate(${data.doa.angulo - 90}deg)`,
                  transformOrigin: '50% 50%',
                }}
              >
                <svg viewBox="0 0 100 100" className="h-full w-full">
                  <path
                    d={`M ${x1} ${y1} A ${radius} ${radius} 0 0 0 ${x2} ${y2}`}
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth={1.6 + waveIndex * 0.25}
                    strokeLinecap="round"
                    className="animate-[sirenImpact_1.2s_ease-out_infinite]"
                    style={{
                      animationDelay: `${waveIndex * 0.15}s`,
                      filter: 'drop-shadow(0 0 6px #ef4444)',
                      opacity: 0.95 - waveIndex * 0.16,
                    }}
                  />
                </svg>
              </div>
            );
          })}

        <div
          className={`relative z-10 flex h-44 w-44 items-center justify-center rounded-full border-2 transition-all duration-700 ${
            data.deteccion.sirena
              ? 'border-red-500 bg-red-950/40 shadow-[0_0_70px_rgba(239,68,68,0.5)]'
              : 'border-cyan-400/20 bg-slate-900/60 shadow-[0_0_50px_rgba(34,211,238,0.15)]'
          }`}
        >
          <svg
            viewBox="0 0 60 100"
            className="h-28 w-28 transition-transform duration-500"
            style={{
              filter:
                'drop-shadow(0 10px 15px rgba(0,0,0,0.5)) drop-shadow(0 0 10px rgba(34,211,238,0.3))',
            }}
          >
            <defs>
              <linearGradient id="cuerpoCocheMobile" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#22d3ee" />
                <stop offset="100%" stopColor="#c084fc" />
              </linearGradient>
            </defs>

            <rect x="10" y="10" width="40" height="80" rx="14" fill="url(#cuerpoCocheMobile)" />
            <rect x="14" y="35" width="32" height="30" rx="6" fill="#0f172a" fillOpacity="0.85" />
            <path d="M14 35 Q 30 25 46 35 L 44 45 Q 30 38 16 45 Z" fill="white" fillOpacity="0.25" />
            <path d="M16 60 Q 30 65 44 60 L 42 68 Q 30 72 18 68 Z" fill="white" fillOpacity="0.15" />
            <rect x="4" y="38" width="6" height="10" rx="2" fill="url(#cuerpoCocheMobile)" />
            <rect x="50" y="38" width="6" height="10" rx="2" fill="url(#cuerpoCocheMobile)" />
            <rect x="14" y="84" width="10" height="4" rx="1" fill={data.deteccion.sirena ? '#ef4444' : '#1e293b'} />
            <rect x="36" y="84" width="10" height="4" rx="1" fill={data.deteccion.sirena ? '#ef4444' : '#1e293b'} />
          </svg>

          <div className="absolute -bottom-6 flex min-w-[96px] items-center justify-center rounded-full border border-white/20 bg-slate-900 px-6 py-2.5 shadow-[0_0_25px_rgba(34,211,238,0.2)]">
            <span className="font-mono text-lg font-black tracking-tighter text-white">
              {roundedAngle}°
            </span>
          </div>
        </div>
      </div>
    </div>
  </main>
);

const MobileStatsView = ({ data, roundedAngle, showAllLogs, setShowAllLogs }) => (
  <main className="flex flex-col gap-4">
    <div className="grid grid-cols-3 gap-2">
      <MiniStat label="Prob." value={`${(data.deteccion.probabilidad * 100).toFixed(0)}%`} tone={data.deteccion.sirena ? 'red' : 'cyan'} />
      <MiniStat
        label="F1"
        value={formatRatioAsPercent(data.metricas_modelo.f1_score)}
        tone="cyan"
      />
      <MiniStat label="Lat." value={`${data.deteccion.latencia_inferencia_ms.toFixed(0)}ms`} tone="purple" />
    </div>

    <Panel titulo="PROBABILIDAD DETECCION">
      <div className="flex flex-col gap-4">
        <div className="flex h-14 items-end gap-1">
          {Array.from({ length: 20 }).map((_, i) => {
            const threshold = i / 20;
            const isActive = data.deteccion.probabilidad > threshold;
            return (
              <div
                key={i}
                className={`flex-1 rounded-sm ${isActive ? (i > 15 ? 'bg-red-500' : 'bg-cyan-400') : 'bg-white/5'}`}
                style={{ height: `${40 + i * 3}%` }}
              />
            );
          })}
        </div>
        <div className="flex items-end justify-between">
          <div>
            <p className="text-[9px] font-bold uppercase text-slate-500">Calculo</p>
            <p className={`font-mono text-3xl font-black italic ${data.deteccion.sirena ? 'text-red-500' : 'text-white'}`}>
              {(data.deteccion.probabilidad * 100).toFixed(1)}%
            </p>
          </div>
          <div className="rounded-full bg-slate-800 p-2">
            <ShieldAlert size={14} />
          </div>
        </div>
      </div>
    </Panel>

    <Panel titulo="LOCALIZACION ANGULAR">
      <div className="flex items-center justify-between rounded-xl border border-white/5 bg-black/30 px-3 py-3">
        <span className="font-mono text-2xl font-black text-white">{roundedAngle}°</span>
        <div className="flex gap-2 text-[10px] font-bold">
          <span className={data.doa.angulo < 180 ? 'text-cyan-400' : 'text-slate-600'}>IZQ</span>
          <span className={data.doa.angulo >= 180 ? 'text-cyan-400' : 'text-slate-600'}>DER</span>
        </div>
      </div>
    </Panel>

    <Panel titulo="FRECUENCIA AUDIO">
      <div className="flex h-20 items-center gap-[2px] rounded-xl bg-black/30 px-2">
        {data.audio.waveform_summary.slice(0, 60).map((v, i) => (
          <div
            key={i}
            className="flex-1 rounded-full bg-purple-500/30"
            style={{ height: `${Math.max(15, Math.abs(v) * 100)}%` }}
          />
        ))}
      </div>
    </Panel>

    <Panel titulo="CAPTURA DE ONDA">
      <div className="relative h-24 overflow-hidden rounded-xl border border-white/10 bg-slate-950">
        <svg viewBox="0 0 200 100" className="absolute inset-0 h-full w-full">
          <path
            d={`M 0 50 ${data.audio.waveform_summary
              .slice(0, 40)
              .map((v, i) => `L ${i * 5} ${50 + v * 40}`)
              .join(' ')}`}
            fill="none"
            stroke={data.deteccion.sirena ? '#ef4444' : '#c084fc'}
            strokeWidth="1.5"
          />
        </svg>
      </div>
    </Panel>

    <Panel titulo="FFT">
      <div className="flex h-24 items-end gap-[1px] rounded-xl border border-white/5 bg-black/20 px-1">
        {data.audio.waveform_summary.map((v, i) => {
          const barHeight = Math.abs(v) * 100;
          return (
            <div
              key={i}
              className={`flex-1 ${barHeight > 70 ? 'bg-red-500' : 'bg-cyan-500/40'}`}
              style={{ height: `${Math.max(5, barHeight)}%` }}
            />
          );
        })}
      </div>
    </Panel>

    <Panel titulo="ESPECTROGRAMA">
      <div className="grid h-24 grid-cols-10 gap-1">
        {Array.from({ length: 80 }).map((_, i) => (
          <div
            key={i}
            className="rounded-sm"
            style={{
              backgroundColor: data.deteccion.sirena ? '#ef4444' : '#22d3ee',
              opacity: Math.random() * (data.deteccion.sirena ? 1 : 0.3),
            }}
          />
        ))}
      </div>
    </Panel>

    <Panel titulo="RENDIMIENTO DEL MODELO">
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
          <p className="text-[8px] font-black uppercase tracking-widest text-slate-500">Accuracy</p>
          <p className="mt-1 font-mono text-2xl font-black text-white">
            {data.metricas_modelo.accuracy.toFixed(4)}
          </p>
        </div>
        <div className="rounded-2xl border border-white/5 bg-white/5 p-3">
          <p className="text-[8px] font-black uppercase tracking-widest text-slate-500">F1-Score</p>
          <p className="mt-1 font-mono text-2xl font-black text-purple-400">
            {data.metricas_modelo.f1_score.toFixed(4)}
          </p>
        </div>
      </div>
    </Panel>

    <Panel titulo="EVENTOS RECIENTES">
      <div
        className={`overflow-hidden transition-all duration-500 ease-in-out ${
          showAllLogs ? 'h-[220px]' : 'h-[60px]'
        }`}
      >
        <div className="flex flex-col gap-2">
          <div onClick={() => setShowAllLogs(!showAllLogs)} className="group cursor-pointer">
            {data.logs.slice(-1).map((log, i) => {
              const isAlert = log.includes('ALERT');
              return (
                <div
                  key={i}
                  className={`flex items-center gap-3 rounded-xl border p-2.5 ${
                    isAlert ? 'border-red-500/40 bg-red-500/20' : 'border-cyan-500/20 bg-cyan-500/10'
                  }`}
                >
                  <div className={`rounded-lg p-1 ${isAlert ? 'bg-red-500/20 text-red-500' : 'bg-cyan-500/20 text-cyan-500'}`}>
                    <Zap size={14} className={showAllLogs ? 'rotate-180 transition-transform' : ''} />
                  </div>
                  <div className="flex flex-1 flex-col">
                    <span className={`text-[9px] font-black ${isAlert ? 'text-red-400' : 'uppercase text-cyan-400'}`}>
                      {isAlert ? 'ULTIMA ALERTA CRITICA' : 'ULTIMO EVENTO'}
                    </span>
                    <span className="truncate font-mono text-[10px] text-slate-200">{log}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex flex-col gap-2">
            {data.logs
              .slice(-5, -1)
              .reverse()
              .map((log, i) => (
                <div key={i} className="rounded-lg border border-white/5 bg-white/[0.02] p-2 text-[9px] font-mono text-slate-400 opacity-60">
                  {log}
                </div>
              ))}
          </div>
        </div>
      </div>
    </Panel>
  </main>
);

const MiniStat = ({ label, value, tone }) => {
  const toneClass =
    tone === 'red'
      ? 'text-red-400'
      : tone === 'purple'
        ? 'text-purple-400'
        : 'text-cyan-400';

  return (
    <div className="rounded-2xl border border-white/8 bg-black/20 px-3 py-2">
      <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">{label}</p>
      <p className={`mt-1 font-mono text-base font-black ${toneClass}`}>{value}</p>
    </div>
  );
};

const ApiStatusBadge = ({ apiStatus, lastApiSync, compact = false }) => {
  const { label, dotClass, textClass, borderClass, bgClass } = getApiStatusMeta(apiStatus);
  const lastSyncLabel = lastApiSync
    ? `Ult. sync ${lastApiSync.toLocaleTimeString('es-ES')}`
    : 'Sin datos todavia';

  return (
    <div
      className={`mt-3 inline-flex items-center gap-2 rounded-full border px-3 py-1.5 ${
        compact ? 'text-[9px]' : 'text-[10px]'
      } ${borderClass} ${bgClass}`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${dotClass}`} />
      <span className={`font-black uppercase tracking-[0.22em] ${textClass}`}>{label}</span>
      <span className="font-mono text-slate-500">{lastSyncLabel}</span>
    </div>
  );
};

const Panel = ({ titulo, children }) => (
  <div className="relative z-10 rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-md">
    <h3 className="mb-5 text-[10px] font-black uppercase tracking-[0.3em] text-slate-500">
      {titulo}
    </h3>
    {children}
  </div>
);

const Stat = ({ label, value }) => (
  <div className="text-right">
    <p className="text-[9px] font-bold uppercase tracking-widest text-slate-500">
      {label}
    </p>
    <p className="font-mono text-2xl font-black leading-tight text-white">{value}</p>
  </div>
);

export default App;
