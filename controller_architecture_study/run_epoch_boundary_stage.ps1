param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'results\epoch_boundary_20261102_v1')
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist' }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$original = Join-Path $PSScriptRoot 'results\boundary_takeover_20261101_v1'
& $Python -u -m controller_architecture_study.replay_obligation_epochs --source (Join-Path $original 'application_rates_only') --output (Join-Path $OutputRoot 'replay')
if ($LASTEXITCODE -ne 0) { throw 'Offline obligation replay failed; no flight was started.' }
$application = Join-Path $OutputRoot 'application'
$boundary = Join-Path $OutputRoot 'boundary_refinement'
$flightDone = Test-Path -LiteralPath (Join-Path $application 'summary.json')
$boundaryDone = $false
if (Test-Path -LiteralPath (Join-Path $boundary 'report.json')) {
    $boundaryDone = (Get-Content -Raw -LiteralPath (Join-Path $boundary 'report.json') | ConvertFrom-Json).complete
}
if (-not ($flightDone -and $boundaryDone)) {
    $connectionProbe = @'
from controller_architecture_study.fly_convex_policy import load_airsim_client
from airsim_interface import require_airsim
load_airsim_client()
c=require_airsim().MultirotorClient(timeout_value=3.)
try:
    vehicles=set(c.listVehicles())
    print('AirSim vehicles:',sorted(vehicles))
    if not {f'Drone{i}' for i in range(1,11)}<=vehicles:
        raise SystemExit('Need Drone1 through Drone10: launch the 12-drone settings, not the old three-drone project.')
finally:
    c.client.close()
'@
    & $Python -c $connectionProbe
    if ($LASTEXITCODE -ne 0) { throw 'AirSim RPC is not ready. Open the 12-drone Blocks project and click Play, then rerun.' }
}
if (-not $flightDone) {
    if (Test-Path -LiteralPath $application) { throw 'Incomplete flight attempt retained. Inspect it, then choose a new -OutputRoot for a new paired attempt.' }
    & $Python -u -m controller_architecture_study.run_reviewed_rate_trial --boundary-report (Join-Path $original 'boundary\report.json') --ranker (Join-Path $original 'ranker\ranker.npz') --seed 6 --output $application
    if ($LASTEXITCODE -ne 0) { throw 'Paired flight failed to complete; partial reports are retained.' }
}
if (-not $boundaryDone) {
    & $Python -u -m controller_architecture_study.refine_recovery_boundary --source (Join-Path $original 'boundary') --output $boundary
    if ($LASTEXITCODE -ne 0) { throw 'Boundary refinement did not complete; valid outcomes and invalid attempts are retained.' }
}
Write-Host "Stage commands completed: $OutputRoot"
Write-Host 'Completion is not a claim of flight benefit or safety. Inspect application/summary.json and boundary_refinement/report.json.'
