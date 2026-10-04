# Сброс MLflow store при переносе проекта или битых путях
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Start-Sleep -Seconds 1

$paths = @(
    (Join-Path $Root "artifacts\mlflow.db"),
    (Join-Path $Root "artifacts\mlruns"),
    (Join-Path $Root "mlruns")
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
        Write-Host "Removed: $path"
    }
}

Write-Host "MLflow store reset. Run: poetry run python -m src.train --smoke"
