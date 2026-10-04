# Bootstrap development environment: install Poetry and build a per-project .venv
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "Poetry not found. Installing with pip..."
    pip install --user poetry
}

# Force a per-project virtual environment (<repo>\.venv), ignored by git.
poetry config virtualenvs.in-project true --local

Write-Host "=== poetry install ==="
poetry install

Write-Host ""
Write-Host "Done. Activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1        (PowerShell)"
Write-Host "  or run commands via: poetry run <cmd>"
Write-Host "Install git hooks: poetry run pre-commit install"
