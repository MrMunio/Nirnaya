# ==============================================================================
# Nirnaya Decision Engine - Streamlit Frontend Launcher (PowerShell)
# ==============================================================================

$CondaPy = "C:\Users\USER\.conda\envs\nirnaya-poc\python.exe"
if (Test-Path $CondaPy) {
    $PythonBin = $CondaPy
} else {
    $PythonBin = "python"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "⚡ Starting Nirnaya Streamlit Decision Studio UI" -ForegroundColor Green
Write-Host "📍 Project Root: $ProjectDir" -ForegroundColor DarkGray
Write-Host "🐍 Python Executable: $PythonBin" -ForegroundColor DarkGray
Write-Host "🌐 URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

& $PythonBin -m streamlit run "$ProjectDir\streamlit_app\app.py" --server.port 8501 --server.headless false
