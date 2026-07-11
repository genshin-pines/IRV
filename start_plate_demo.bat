@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] Python venv not found: %PYTHON%
  pause
  exit /b 1
)

echo [1/2] Killing stale processes...
taskkill /f /im mediamtx.exe >nul 2>&1
taskkill /f /im ffmpeg.exe >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'server.py' -or $_.CommandLine -match 'live_server.py') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
timeout /t 1 /nobreak >nul

echo [2/2] Starting services...
echo   Image/video upload: http://localhost:8003
echo   Realtime monitor:   http://localhost:8004
echo.
echo Close this window to stop everything.
echo.

start "Plate Web Service :8003" /D "%ROOT%cv_modules\hyperlpr_demo" cmd /k ""%PYTHON%" server.py"
start "Realtime Plate Service :8004" /D "%ROOT%cv_modules\hyperlpr_demo" cmd /k ""%PYTHON%" live_server.py"

echo All services started.
echo.
pause
