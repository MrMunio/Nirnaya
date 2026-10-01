@echo off
REM ==============================================================================
REM Nirnaya Decision Engine - Streamlit Frontend Launcher (Windows CMD)
REM ==============================================================================

set CONDA_ENV_PY=C:\Users\USER\.conda\envs\nirnaya-poc\python.exe

if exist "%CONDA_ENV_PY%" (
    set PYTHON_BIN=%CONDA_ENV_PY%
) else (
    set PYTHON_BIN=python
)

echo [Nirnaya UI] Starting Streamlit Showcase with %PYTHON_BIN%...
cd /d "%~dp0\.."
"%PYTHON_BIN%" -m streamlit run streamlit_app\app.py --server.port 8501 --server.headless false
pause
