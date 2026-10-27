# Use exactly the same tracking configuration as the training process.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ProjectMlflow = Join-Path $ProjectRoot '.venv\Scripts\mlflow.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Create the environment first: poetry sync --with dev'
}
$TrackingUri = & $ProjectPython -c 'from src.config import MLFLOW_TRACKING_URI; print(MLFLOW_TRACKING_URI)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot read tracking configuration.' }
$ArtifactUri = & $ProjectPython -c 'from src.config import MLFLOW_ARTIFACTS_URI; print(MLFLOW_ARTIFACTS_URI)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot read artifact configuration.' }
Write-Host 'MLflow UI: http://127.0.0.1:5000 (Ctrl+C to stop)'
& $ProjectMlflow ui --backend-store-uri $TrackingUri --default-artifact-root $ArtifactUri --host 127.0.0.1 --port 5000
if ($LASTEXITCODE -ne 0) { throw 'MLflow UI stopped with an error.' }
