@echo off
setlocal
cd /d "%~dp0"
where uv >/dev/null 2>nul
if errorlevel 1 (
    echo No se encontro "uv". Instalalo con: winget install astral-sh.uv
    pause
    exit /b 1
)
uv sync --quiet
if errorlevel 1 (
    echo Fallo la instalacion de dependencias. Revisa tu conexion e intentalo de nuevo.
    pause
    exit /b 1
)
uv run pythonw main.py
if errorlevel 1 (
    echo La aplicacion termino con un error. Revisa data\logs\app.log
    pause
)
