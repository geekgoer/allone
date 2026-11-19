param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [int]$WaitForPlayMinutes = 15
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist' }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$stageRoot = Join-Path $PSScriptRoot 'results\boundary_takeover_20261101_v1'
New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
$boundaryDir = Join-Path $stageRoot 'boundary'
$applicationDir = Join-Path $stageRoot 'application_rates_only'
$needsSampling = -not (Test-Path -LiteralPath (Join-Path $boundaryDir 'report.json'))
if (-not $needsSampling) {
    $boundaryReport = Get-Content -Raw -LiteralPath (Join-Path $boundaryDir 'report.json') | ConvertFrom-Json
    $needsSampling = -not $boundaryReport.complete
}
$needsFlight = -not (Test-Path -LiteralPath (Join-Path $applicationDir 'summary.json'))
if ($needsSampling -or $needsFlight) {
    $waitDeadline = (Get-Date).AddMinutes($WaitForPlayMinutes)
    while ($true) {
        & $Python -c "import socket; s=socket.socket(); s.settimeout(1); result=s.connect_ex(('127.0.0.1',41451)); s.close(); raise SystemExit(0 if result==0 else 1)"
        if ($LASTEXITCODE -eq 0) { break }
        if ((Get-Date) -ge $waitDeadline) { throw 'AirSim RPC is not ready. Click Play and rerun this script; completed data are retained.' }
        Start-Sleep -Seconds 5
    }
}
if ($needsSampling) {
    & $Python -u -m controller_architecture_study.probe_recovery_boundary --output $boundaryDir
    if ($LASTEXITCODE -ne 0) { throw 'Boundary sampling did not complete; all queried trials are retained.' }
}
$rankerDir = Join-Path $stageRoot 'ranker'
if (-not (Test-Path -LiteralPath (Join-Path $rankerDir 'report.json'))) {
    & $Python -u -m controller_architecture_study.repair_controller_effect --source (Join-Path $PSScriptRoot 'results\controller_selector_237d_v2\data') --cadence (Join-Path $PSScriptRoot 'results\controller_effect_20261031_v1\cadence_data') --original (Join-Path $PSScriptRoot 'results\controller_effect_20261031_v1\ranker\ranker.npz') --boundary $boundaryDir --output $rankerDir
    if ($LASTEXITCODE -ne 0) { throw 'Boundary-conditioned ranker training failed; source controllers unchanged.' }
}
if (-not (Test-Path -LiteralPath (Join-Path $applicationDir 'summary.json'))) {
    if (Test-Path -LiteralPath $applicationDir) { throw 'An incomplete application attempt exists; inspect it before choosing a new output directory.' }
    & $Python -u -m controller_architecture_study.run_reviewed_rate_trial --boundary-report (Join-Path $boundaryDir 'report.json') --ranker (Join-Path $rankerDir 'ranker.npz') --seed 6 --output $applicationDir
    if ($LASTEXITCODE -ne 0) { throw 'Application experiment did not complete; inspect retained reports.' }
}
& $Python -m controller_architecture_study.audit_boundary_takeover --root $stageRoot --application application_rates_only
if ($LASTEXITCODE -ne 0) { throw 'Evidence audit failed; keep the raw reports.' }
Write-Host "Completed boundary sampling, frozen ranker fit, and rate-application trials: $stageRoot"
