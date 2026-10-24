# Reproduce the locked environment. Install Poetry separately if unavailable.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    throw 'Poetry is required. Install Poetry 2.x, reopen the shell, and retry.'
}
poetry check --lock
if ($LASTEXITCODE -ne 0) { throw 'Poetry configuration/lock check failed.' }
poetry sync --with dev --no-interaction
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
poetry run pre-commit install
if ($LASTEXITCODE -ne 0) { throw 'Git hook installation failed.' }
Write-Host 'Ready. Run commands with poetry run, or activate .venv\Scripts\Activate.ps1.'
