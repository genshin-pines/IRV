param(
  [int]$Port = 8000,
  [switch]$Reload
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "[IRV] Working directory: $PSScriptRoot"
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$venvDir = if ($env:IRV_VENV_DIR) { $env:IRV_VENV_DIR } else { Join-Path $workspaceRoot ".venvs\IRV_main2" }
$venvPython = Join-Path $venvDir "Scripts\python.exe"
Write-Host "[IRV] Python environment: $venvDir"

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

if (!(Test-Path $venvPython)) {
  Write-Host "[IRV] Creating external virtual environment..."
  New-Item -ItemType Directory -Force -Path (Split-Path $venvDir -Parent) | Out-Null
  python -m venv $venvDir
}

Write-Host "[IRV] Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "[IRV] Installing requirements..."
& $venvPython -m pip install -r requirements.txt

# HyperLPR declares the CPU onnxruntime package, which conflicts with DirectML.
Write-Host "[IRV] Installing HyperLPR without its CPU ONNX Runtime dependency..."
& $venvPython -m pip install --no-deps hyperlpr3==0.1.3

# hyperlpr3 may pull in the CPU onnxruntime package and overwrite DirectML's
# shared module. Repair that conflict only when the GPU provider is missing.
$onnxProviders = & $venvPython -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
if ($onnxProviders -notmatch "DmlExecutionProvider") {
  Write-Host "[IRV] Repairing ONNX Runtime DirectML provider..."
  & $venvPython -m pip uninstall -y onnxruntime
  & $venvPython -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4
}

$torchCudaReady = & $venvPython -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
if ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and $torchCudaReady -ne "yes") {
  Write-Host "[IRV] Installing CUDA 12.6 PyTorch for traffic gesture recognition..."
  & $venvPython -m pip install --force-reinstall --no-deps torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
}

while (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -ErrorAction SilentlyContinue) {
  Write-Host "[IRV] Port $Port is busy; trying $($Port + 1)..."
  $Port += 1
}

Write-Host "[IRV] Starting backend: http://127.0.0.1:$Port"
Write-Host "[IRV] Swagger:          http://127.0.0.1:$Port/docs"
if ($Reload) {
  Write-Host "[IRV] Development reload enabled. Avoid uploading videos while code is reloading."
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
} else {
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
}
