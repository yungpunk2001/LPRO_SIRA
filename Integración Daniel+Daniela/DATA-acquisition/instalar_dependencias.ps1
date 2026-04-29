$ErrorActionPreference = "Stop"

Write-Host ">>> Iniciando instalacion del entorno SIRA DATA-acquisition..." -ForegroundColor Green

# TensorFlow en Windows debe ejecutarse con una version compatible de Python.
py -3.10 -m venv venv_sira
Write-Host ">>> Entorno virtual 'venv_sira' creado/actualizado." -ForegroundColor Cyan

$python = ".\venv_sira\Scripts\python.exe"

& $python -m pip install --upgrade pip

Write-Host ">>> Descargando e instalando librerias necesarias..." -ForegroundColor Yellow
& $python -m pip install numpy scipy librosa scikit-learn joblib tensorflow requests pyaudio matplotlib

Write-Host ">>> INSTALACION COMPLETADA CON EXITO." -ForegroundColor Green
Write-Host ">>> Para activar el entorno: .\venv_sira\Scripts\Activate.ps1" -ForegroundColor Cyan
