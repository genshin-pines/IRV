param(
  [int]$Port = 8000,
  [switch]$SkipCudaTorch
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
# Avoid a locked per-user pip cache on lab/demo computers.
if (-not $env:PIP_CACHE_DIR) {
  $env:PIP_CACHE_DIR = Join-Path $PSScriptRoot ".pip-cache"
}

Write-Host "[IRV] Working directory: $PSScriptRoot"

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "[IRV] Creating virtual environment..."
  python -m venv .venv
}

Write-Host "[IRV] Upgrading pip..."
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
  throw "pip 升级失败，请检查网络或 .venv 权限。"
}

Write-Host "[IRV] Installing requirements..."
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
  throw "依赖安装失败，后端未启动。请根据上方 pip 错误修复后重试。"
}

# hyperlpr3 may pull in the CPU onnxruntime package and overwrite DirectML's
# shared module. Repair that conflict only when the GPU provider is missing.
$onnxProviders = & ".\.venv\Scripts\python.exe" -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
if ($onnxProviders -notmatch "DmlExecutionProvider") {
  Write-Host "[IRV] Repairing ONNX Runtime DirectML provider..."
  & ".\.venv\Scripts\python.exe" -m pip uninstall -y onnxruntime
  & ".\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4
}

if (-not $SkipCudaTorch) {
  $torchCudaReady = & ".\.venv\Scripts\python.exe" -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
  if ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and $torchCudaReady -ne "yes") {
    Write-Host "[IRV] Installing CUDA 12.6 PyTorch for traffic gesture recognition..."
    & ".\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  }
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
