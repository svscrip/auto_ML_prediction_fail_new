# MLflow UI — тот же backend, что и src.train
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Mlflow = Join-Path $Root ".venv\Scripts\mlflow.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Сначала создайте окружение: poetry install (или .\scripts\bootstrap.ps1)"
    exit 1
}

$mlflowUri = & $Python -c "from src.config import MLFLOW_TRACKING_URI; print(MLFLOW_TRACKING_URI)"
$artifactRoot = & $Python -c "from src.config import MLFLOW_ARTIFACTS_DIR; print(MLFLOW_ARTIFACTS_DIR.resolve())"

Write-Host "Backend: $mlflowUri"
Write-Host "Artifacts: $artifactRoot"
Write-Host "MLflow UI: http://127.0.0.1:5000"
Write-Host "Нажмите Ctrl+C для остановки"

& $Mlflow ui `
    --backend-store-uri $mlflowUri `
    --default-artifact-root $artifactRoot `
    --host 127.0.0.1 `
    --port 5000
