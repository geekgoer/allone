param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Models = 'main18_plain_mlp_h4_w1024_seed0',
    [ValidateSet('cpu','cuda')][string]$Device = 'cpu',
    [switch]$AllMissing
)
$ErrorActionPreference = 'Stop'
$proj = 'F:\FalconReproduction\airsim_control'
$bundle = Split-Path -Parent $PSScriptRoot
$registry = Get-Content -Raw -LiteralPath (Join-Path $bundle 'model_registry.json') | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
Set-Location -LiteralPath $proj
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if ($AllMissing) {
    $selected = @($registry.models | Where-Object { $_.flight_status -eq 'NOT_FOUND' })
} else {
    $ids = @($Models.Split(',') | ForEach-Object { $_.Trim() })
    $selected = @($registry.models | Where-Object { $_.id -in $ids })
    if ($selected.Count -ne $ids.Count) { throw 'Unknown or duplicate model id; inspect model_registry.json.' }
}
Write-Host 'Start UE4 yourself and click Play. This script does not launch a new simulator.'
foreach ($entry in $selected) {
    $checkpoint = $entry.source_checkpoint
    if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.source_checkpoint_sha256) {
        throw "Source checkpoint changed: $checkpoint"
    }
    Write-Host "Flying $($entry.id) at 5m/s from aerial rest; nominal route, no state kick."
    & $Python -m controller_architecture_study.fly_convex_policy --checkpoint $checkpoint --environment-id 48 --route-index 0 --device $Device --keep-gates
    if ($LASTEXITCODE -ne 0) { throw "Flight failed for $($entry.id); preserve the printed report. Remaining models were not started." }
}
