# Archive the default repository-local store. No deletion and no process termination.
# Stop MLflow and training first. Custom/remote stores must be managed separately.
[CmdletBinding(SupportsShouldProcess)]
param()
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))
$RootPrefix = $ProjectRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if ($env:ML_ARTIFACTS_DIR -or $env:MLFLOW_TRACKING_URI -or $env:MLFLOW_ARTIFACTS_URI) {
    throw 'Custom storage is configured. This helper only archives the default local store.'
}
$BackupRoot = Join-Path $ProjectRoot ('artifacts\mlflow-backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$RelativePaths = @('artifacts\mlflow.db', 'artifacts\mlflow.db-wal', 'artifacts\mlflow.db-shm', 'artifacts\mlartifacts', 'artifacts\mlruns', 'mlruns')
$Moves = @()
foreach ($Relative in $RelativePaths) {
    $SourcePath = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Relative))
    $DestinationPath = [IO.Path]::GetFullPath((Join-Path $BackupRoot $Relative))
    foreach ($Candidate in @($SourcePath, $DestinationPath)) {
        if (-not $Candidate.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Path outside project: $Candidate"
        }
        # Refuse junctions/symlinks in every existing ancestor, including artifacts/.
        $Ancestor = $Candidate
        while ($Ancestor -and $Ancestor -ne $ProjectRoot) {
            if (Test-Path -LiteralPath $Ancestor) {
                $Item = Get-Item -LiteralPath $Ancestor -Force
                if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    throw "Reparse point is not supported: $Ancestor"
                }
            }
            $Ancestor = Split-Path -Parent $Ancestor
        }
    }
    if (Test-Path -LiteralPath $SourcePath) {
        $Moves += [pscustomobject]@{ Source = $SourcePath; Destination = $DestinationPath }
    }
}
if ($Moves.Count -eq 0) { Write-Host 'No local store to archive.'; return }
if (-not $PSCmdlet.ShouldProcess($ProjectRoot, "Move stopped MLflow store to $BackupRoot")) { return }
foreach ($Move in $Moves) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Move.Destination) -Force | Out-Null
    Move-Item -LiteralPath $Move.Source -Destination $Move.Destination
    Write-Host "Archived: $($Move.Source) -> $($Move.Destination)"
}
Write-Host 'History is preserved in the backup. The next training run creates a new store.'
