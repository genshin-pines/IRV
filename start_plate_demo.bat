@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "MEDIAMTX=%ROOT%tools\mediamtx\mediamtx.exe"
set "FFMPEG=%ROOT%tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
set "VIDEO=%ROOT%cv_modules\hyperlpr_demo\test_video\test12.mp4"
set "RTSP_URL=rtsp://127.0.0.1:8554/live/test12"

if not exist "%PYTHON%" (
  echo [ERROR] Python venv not found: %PYTHON%
  pause
  exit /b 1
)

if not exist "%MEDIAMTX%" (
  echo [ERROR] MediaMTX not found: %MEDIAMTX%
  pause
  exit /b 1
)

if not exist "%FFMPEG%" (
  echo [ERROR] FFmpeg not found: %FFMPEG%
  pause
  exit /b 1
)

if not exist "%VIDEO%" (
  echo [ERROR] Test video not found: %VIDEO%
  pause
  exit /b 1
)

echo [1/4] Starting MediaMTX...
start "MediaMTX" /D "%ROOT%tools\mediamtx" "%MEDIAMTX%"

timeout /t 2 /nobreak >nul

echo [2/4] Pushing test video to %RTSP_URL% ...
start "FFmpeg Push test12" /D "%ROOT%" "%FFMPEG%" ^
  -re -stream_loop -1 ^
  -i "%VIDEO%" ^
  -vf "scale=1280:-2,fps=15" ^
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p ^
  -g 15 -keyint_min 15 -bf 0 -sc_threshold 0 ^
  -b:v 2500k -maxrate 2500k -bufsize 1000k ^
  -an -rtsp_transport tcp -f rtsp "%RTSP_URL%"

echo [3/4] Starting image/video recognition web service...
start "Plate Web Service :8003" /D "%ROOT%cv_modules\hyperlpr_demo" cmd /k ""%PYTHON%" server.py"

echo [4/4] Starting realtime recognition web service...
start "Realtime Plate Service :8004" /D "%ROOT%cv_modules\hyperlpr_demo" cmd /k ""%PYTHON%" live_server.py"

echo.
echo Done.
echo Image/video web: http://localhost:8003
echo Realtime web:    http://localhost:8004
echo RTSP stream:     %RTSP_URL%
echo.
pause
