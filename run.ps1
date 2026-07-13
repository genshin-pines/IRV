param(
  [int]$Port = 8000,
  [switch]$Reload
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "[IRV] Working directory: $PSScriptRoot"
# ── 虚拟环境 ──────────────────────────────────────────
$venvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path $venvPython)) {
  Write-Host "[IRV] Creating virtual environment..."
  python -m venv .venv
  # 强制安装（首次创建）
  $forceInstall = $true
} else {
  # 用 requirements.txt 的 hash 判断依赖是否需要更新
  $reqHash = (Get-FileHash -Path "requirements.txt" -Algorithm SHA256).Hash
  $hashFile = ".\.venv\.req-hash"
  $cachedHash = if (Test-Path $hashFile) { Get-Content $hashFile -Raw } else { "" }
  $forceInstall = ($reqHash -ne $cachedHash)
}

# ── 依赖安装（仅在 requirements.txt 变更时执行）──────
if ($forceInstall) {
  Write-Host "[IRV] Upgrading pip..."
  & $venvPython -m pip install --upgrade pip --quiet

  Write-Host "[IRV] Installing requirements..."
  & $venvPython -m pip install -r requirements.txt --quiet

  # hyperlpr3 may pull in the CPU onnxruntime package and overwrite DirectML's
  # shared module. Repair that conflict only when the GPU provider is missing.
  $onnxProviders = & $venvPython -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
  if ($onnxProviders -notmatch "DmlExecutionProvider") {
    Write-Host "[IRV] Repairing ONNX Runtime DirectML provider..."
    & $venvPython -m pip uninstall -y onnxruntime --quiet
    & $venvPython -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4 --quiet
  }

  $torchCudaReady = & $venvPython -c "import torch; print('yes' if torch.cuda.is_available() else 'no')"
  if ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and $torchCudaReady -ne "yes") {
    Write-Host "[IRV] Installing CUDA 12.6 PyTorch for traffic gesture recognition..."
    & $venvPython -m pip install --force-reinstall --no-deps torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --quiet
  }

  # 写入 hash 标记，下次启动跳过安装
  Set-Content -Path $hashFile -Value $reqHash -NoNewline
  Write-Host "[IRV] Dependencies ready (hash: $($reqHash.Substring(0,8))...)"
} else {
  Write-Host "[IRV] Dependencies up to date, skipping install"
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

# ── 清理所有残留的 IRV 后端进程 ─────────────────────
$stalePorts = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort (8000..8020) -ErrorAction SilentlyContinue
if ($stalePorts) {
  Write-Host "[IRV] Cleaning up stale backend processes..."
  $killed = @{}
  foreach ($conn in $stalePorts) {
    $owningPid = $conn.OwningProcess
    if (-not $killed[$owningPid]) {
      $proc = Get-Process -Id $owningPid -ErrorAction SilentlyContinue
      if ($proc) {
        Write-Host "[IRV]   Killing $($proc.ProcessName) (PID $owningPid) on port $($conn.LocalPort)"
        Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
        $killed[$owningPid] = $true
      }
    }
  }
  if ($killed.Count -gt 0) {
    Start-Sleep -Seconds 2
  }
}

if (Test-IrvHealth -CheckPort $Port) {
  Write-Host "[IRV] Port $Port still in use, trying next..."
  $Port = 8001
}

Write-Host "[IRV] Starting backend: http://127.0.0.1:$Port"
Write-Host "[IRV] Swagger:          http://127.0.0.1:$Port/docs"
if ($Reload) {
  Write-Host "[IRV] Development reload enabled. Avoid uploading videos while code is reloading."
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
} else {
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
}
