param(
  [int]$Port = 8000,
  [switch]$Reload
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "[IRV] Working directory: $PSScriptRoot"

$venvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path "requirements.txt")) {
  throw "[IRV] requirements.txt not found in $PSScriptRoot"
}

$createdVenv = $false
if (!(Test-Path $venvPython)) {
  Write-Host "[IRV] Creating virtual environment..."
  python -m venv .venv
  if ($LASTEXITCODE -ne 0 -or !(Test-Path $venvPython)) {
    throw "[IRV] Failed to create the Python virtual environment."
  }
  $createdVenv = $true
}

function Invoke-VenvPython {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)

  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    & $venvPython @Arguments
    $pythonExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  if ($pythonExitCode -ne 0) {
    throw "[IRV] Python command failed: python $($Arguments -join ' ')"
  }
}

function Test-PythonImport {
  param([string]$ModuleName)

  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    & $venvPython -c "import importlib, sys; importlib.import_module(sys.argv[1])" $ModuleName *> $null
    $pythonExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  return $pythonExitCode -eq 0
}

function Test-PythonCode {
  param([string]$Code)

  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    & $venvPython -c $Code *> $null
    $pythonExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  return $pythonExitCode -eq 0
}

function Get-PythonText {
  param([string]$Code)

  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $outputLines = & $venvPython -c $Code 2>$null
    $pythonExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  if ($pythonExitCode -ne 0) {
    return $null
  }
  return (($outputLines | Out-String).Trim())
}

$requiredModules = @(
  "fastapi",
  "uvicorn",
  "sqlalchemy",
  "pydantic",
  "dotenv",
  "requests",
  "cv2",
  "mediapipe",
  "numpy",
  "onnxruntime",
  "ultralytics",
  "websockets",
  "filterpy",
  "scipy",
  "torch",
  "torchvision",
  "PIL",
  "tqdm",
  "loguru"
)

if ($createdVenv) {
  Write-Host "[IRV] Upgrading pip in the new virtual environment..."
  Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
}

Write-Host "[IRV] Checking and installing requirements.txt dependencies..."
Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt", "--quiet")

Write-Host "[IRV] Checking HyperLPR3 and its helper dependencies..."
$hyperLprReady = Test-PythonCode -Code "import hyperlpr3"
$missingHyperLprHelpers = @()
foreach ($helperModule in @("tqdm", "loguru")) {
  if (!(Test-PythonImport -ModuleName $helperModule)) {
    $missingHyperLprHelpers += $helperModule
  }
}

if (!$hyperLprReady) {
  Write-Host "[IRV]   MISSING hyperlpr3; installing hyperlpr3 0.1.3..."
  Invoke-VenvPython -Arguments @("-m", "pip", "install", "--force-reinstall", "--no-deps", "hyperlpr3==0.1.3", "--quiet")
} else {
  Write-Host "[IRV]   OK hyperlpr3 is installed"
}

if ($missingHyperLprHelpers.Count -gt 0) {
  Write-Host "[IRV]   MISSING $($missingHyperLprHelpers -join ', '); installing helper dependencies..."
  Invoke-VenvPython -Arguments @("-m", "pip", "install", "tqdm", "loguru", "--quiet")
} else {
  Write-Host "[IRV]   OK tqdm and loguru are installed"
}

# hyperlpr3 declares the CPU onnxruntime package, which conflicts with DirectML.
$onnxProviders = Get-PythonText -Code "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
if (!$onnxProviders -or $onnxProviders -notmatch "DmlExecutionProvider") {
  Write-Host "[IRV] Repairing ONNX Runtime DirectML provider..."
  Invoke-VenvPython -Arguments @("-m", "pip", "uninstall", "-y", "onnxruntime", "--quiet")
  Invoke-VenvPython -Arguments @("-m", "pip", "install", "--force-reinstall", "--no-deps", "onnxruntime-directml==1.24.4", "--quiet")
  $onnxProviders = Get-PythonText -Code "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
}

$hasNvidiaGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$torchCudaReady = Get-PythonText -Code "import torch; print('yes' if torch.cuda.is_available() else 'no')"
if ($hasNvidiaGpu -and $torchCudaReady -ne "yes") {
  Write-Host "[IRV] Installing CUDA 12.6 PyTorch for traffic gesture recognition..."
  Invoke-VenvPython -Arguments @("-m", "pip", "install", "--force-reinstall", "--no-deps", "torch==2.13.0+cu126", "torchvision==0.28.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--quiet")
}

Write-Host "[IRV] Verifying runtime imports..."
$failedImports = @()
foreach ($moduleName in $requiredModules) {
  if (Test-PythonImport -ModuleName $moduleName) {
    Write-Host "[IRV]   OK $moduleName"
  } else {
    Write-Host "[IRV]   MISSING $moduleName"
    $failedImports += $moduleName
  }
}

if ($failedImports.Count -gt 0) {
  throw "[IRV] Dependency verification failed: $($failedImports -join ', ')"
}

$hyperLprVersion = Get-PythonText -Code "import importlib.metadata as m; import hyperlpr3; print(m.version('hyperlpr3'))"
if (!$hyperLprVersion) {
  throw "[IRV] Dependency verification failed: hyperlpr3"
}
Write-Host "[IRV]   OK hyperlpr3 $hyperLprVersion"

if (!(Test-PythonCode -Code "import onnxruntime as ort; assert 'DmlExecutionProvider' in ort.get_available_providers()")) {
  throw "[IRV] Dependency verification failed: ONNX Runtime DirectML provider"
}
Write-Host "[IRV] Dependency check passed. ONNX providers: $onnxProviders"

Write-Host "[IRV] Checking required model files..."
$requiredModelFiles = @(
  "vendor\web_gesture_backend\dgcore\models\hand_landmarker.task",
  "vendor\optimized_traffic\ctpgr\checkpoints\pose_model.pt",
  "vendor\optimized_traffic\models\gesture_bilstm_multi_video.pt"
)
$missingModelFiles = @()
foreach ($relativeModelPath in $requiredModelFiles) {
  $modelPath = Join-Path $PSScriptRoot $relativeModelPath
  if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
    Write-Host "[IRV]   OK $relativeModelPath"
  } else {
    Write-Host "[IRV]   MISSING $relativeModelPath"
    $missingModelFiles += $relativeModelPath
  }
}
if ($missingModelFiles.Count -gt 0) {
  throw "[IRV] Required model files are missing: $($missingModelFiles -join ', ')"
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
if ($Reload) {
  Write-Host "[IRV] Development reload enabled. Avoid uploading videos while code is reloading."
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
} else {
  & $venvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
}
