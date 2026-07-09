param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "[IRV] Working directory: $PSScriptRoot"

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "[IRV] Creating virtual environment..."
  python -m venv .venv
}

Write-Host "[IRV] Upgrading pip..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[IRV] Installing requirements..."
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

function Test-IrvHealth {
  param([int]$CheckPort)
  try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$CheckPort/api/health" -TimeoutSec 2
    return $response.ok -eq $true
  } catch {
    return $false
  }
}

if (Test-IrvHealth -CheckPort $Port) {
  Write-Host "[IRV] Backend is already running: http://127.0.0.1:$Port"
  Write-Host "[IRV] Swagger:                    http://127.0.0.1:$Port/docs"
  exit 0
}

while (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -ErrorAction SilentlyContinue) {
  Write-Host "[IRV] Port $Port is busy; trying $($Port + 1)..."
  $Port += 1
}

Write-Host "[IRV] Starting backend: http://127.0.0.1:$Port"
Write-Host "[IRV] Swagger:          http://127.0.0.1:$Port/docs"
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
