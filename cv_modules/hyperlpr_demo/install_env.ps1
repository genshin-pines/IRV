$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
}

& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

# hyperlpr3 depends on the CPU onnxruntime package. For this project we keep
# only onnxruntime-directml so gpu_patch.py can see DmlExecutionProvider.
& $Python -m pip uninstall -y onnxruntime
& $Python -m pip install --force-reinstall onnxruntime-directml==1.24.4

& $Python -c "import onnxruntime as ort; print('ONNX Runtime', ort.__version__); print('Providers:', ort.get_available_providers())"
