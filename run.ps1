param(
  [int]$Port = 8000,
  [switch]$Install,
  [switch]$Foreground
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-IrvPython {
  $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython) {
    return $venvPython
  }

  $candidates = @(
    $env:IRV_PYTHON,
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Python313\python.exe",
    "C:\Python312\python.exe"
  ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
  if ($candidates.Count -gt 0) {
    return $candidates[0]
  }

  $command = Get-Command python -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }
  throw "Python not found. Set IRV_PYTHON or install Python 3.10+."
}

function Test-IrvHealth {
  param([int]$CheckPort)
  try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$CheckPort/api/health" -TimeoutSec 2
    return $response.ok -eq $true
  } catch {
    return $false
  }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $venvPython)) {
  $basePython = Get-IrvPython
  Write-Host "[IRV] Creating virtual environment with $basePython"
  & $basePython -m venv .venv
}

$python = Get-IrvPython
Write-Host "[IRV] Python: $(& $python -V)"

if ($Install) {
  Write-Host "[IRV] Installing project dependencies..."
  & $python -m pip install -r requirements.txt
}

if (Test-IrvHealth -CheckPort $Port) {
  Write-Host "[IRV] Backend is already running: http://127.0.0.1:$Port"
  Write-Host "[IRV] Gesture settings:             http://127.0.0.1:$Port/gesture-settings"
  exit 0
}

if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -ErrorAction SilentlyContinue) {
  throw "Port $Port is occupied. Use .\run.ps1 -Port <port> or stop the process using it."
}

$arguments = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Foreground) {
  Write-Host "[IRV] Starting foreground server: http://127.0.0.1:$Port"
  & $python @arguments
  exit $LASTEXITCODE
}

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $logDir "server-$stamp.out.log"
$stderr = Join-Path $logDir "server-$stamp.err.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

for ($attempt = 0; $attempt -lt 20; $attempt++) {
  Start-Sleep -Milliseconds 500
  if (Test-IrvHealth -CheckPort $Port) {
    Write-Host "[IRV] Started (PID $($process.Id)): http://127.0.0.1:$Port"
    Write-Host "[IRV] Gesture settings:          http://127.0.0.1:$Port/gesture-settings"
    Write-Host "[IRV] Logs: $stdout"
    exit 0
  }
  if ($process.HasExited) {
    break
  }
}

if (Test-Path -LiteralPath $stderr) {
  Get-Content -LiteralPath $stderr -Tail 40
}
throw "IRV did not become healthy. Check $stderr"
