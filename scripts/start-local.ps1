# Stop immediately if training/inference fails; never serve a stale model by accident.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Create the environment first: poetry sync --with dev'
}
Write-Host 'Training smoke model...'
& $ProjectPython -m src.train --smoke
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inference and UI were not started.' }
Write-Host 'Running inference...'
& $ProjectPython -m src.predict
if ($LASTEXITCODE -ne 0) { throw 'Inference failed; UI was not started.' }
& (Join-Path $ProjectRoot 'scripts\start-mlflow.ps1')
