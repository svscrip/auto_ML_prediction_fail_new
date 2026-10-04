# Локальный запуск MLOps-пайплайна
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Mlflow = Join-Path $Root ".venv\Scripts\mlflow.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Сначала создайте окружение: poetry install (или .\scripts\bootstrap.ps1)"
    exit 1
}

Write-Host "=== Обучение (smoke) ==="
& $Python -m src.train --smoke

Write-Host "`n=== Инференс ==="
& $Python -m src.predict

$dbPath = (Join-Path $Root "artifacts\mlflow.db").Replace("\", "/")
$artifactRoot = Join-Path $Root "artifacts\mlartifacts"
$mlflowUri = "sqlite:///$dbPath"

Write-Host "`n=== MLflow UI: http://127.0.0.1:5000 ==="
Write-Host "Нажмите Ctrl+C для остановки MLflow"
& $Mlflow ui `
    --backend-store-uri $mlflowUri `
    --default-artifact-root $artifactRoot `
    --host 127.0.0.1 `
    --port 5000
