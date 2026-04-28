Write-Host ">>> Iniciando instalacion del entorno SIRA..." -ForegroundColor Green

# Crear entorno virtual
python -m venv venv_sira
Write-Host ">>> Entorno virtual 'venv_sira' creado." -ForegroundColor Cyan

# Activar entorno (ruta Windows)
$envPath = ".\venv_sira\Scripts\activate.ps1"
Invoke-Expression $envPath

# Actualizar pip
python -m pip install --upgrade pip

# Instalar TODO (FastAPI, IA, Audio, Matemáticas)
Write-Host ">>> Descargando e instalando librerias (esto puede tardar)..." -ForegroundColor Yellow
pip install fastapi uvicorn pydantic requests
pip install tensorflow keras librosa sounddevice
pip install numpy scipy matplotlib

Write-Host ">>> INSTALACION COMPLETADA CON EXITO." -ForegroundColor Green
Write-Host ">>> Para activar el entorno manualmente en el futuro usa: .\venv_sira\Scripts\activate" -ForegroundColor Cyan
Pause