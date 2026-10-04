# Activate the per-project virtual environment (<repo>\.venv)
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $Activate)) {
    Write-Error "Environment not found: $Activate. Run scripts\bootstrap.ps1 first."
    exit 1
}

& $Activate
Write-Host "Activated: $Activate"
