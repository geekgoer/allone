param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'results\runtime_service_repair_20261102_v1'),
    [switch]$ReplayOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist' }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$source = Join-Path $PSScriptRoot 'results\epoch_boundary_20261102_v1\application'
if (-not (Test-Path -LiteralPath (Join-Path $OutputRoot 'replay\report.json'))) {
    & $Python -u -m controller_architecture_study.replay_obligation_epochs --source $source --output (Join-Path $OutputRoot 'replay')
    if ($LASTEXITCODE -ne 0) { throw 'Archived obligation audit failed; no flight started.' }
}
if ($ReplayOnly) { Write-Host "Offline binding audit only: $OutputRoot"; return }
if (-not (Test-Path -LiteralPath (Join-Path $OutputRoot 'executor\report.json'))) {
    & $Python -u -m controller_architecture_study.check_cpu_xpu_executor --models plain_mlp_h2_w128_seed0,plain_mlp_h9_w1024_seed0 --output (Join-Path $OutputRoot 'executor')
    if ($LASTEXITCODE -ne 0) { throw 'CPU/XPU process check failed; no flight started.' }
}
$application = Join-Path $OutputRoot 'application'
if (Test-Path -LiteralPath (Join-Path $application 'summary.json')) {
    Write-Host "Existing completed reports retained: $application"
    return
}
if (Test-Path -LiteralPath $application) { throw 'Partial flight results retained. Inspect them and choose a new -OutputRoot.' }
$probe = @'
from controller_architecture_study.fly_convex_policy import load_airsim_client
from airsim_interface import require_airsim
load_airsim_client()
c=require_airsim().MultirotorClient(timeout_value=3.)
try:
    vehicles=set(c.listVehicles())
    print('AirSim vehicles:',sorted(vehicles))
    if not {f'Drone{i}' for i in range(1,11)}<=vehicles:
        raise SystemExit('Need Drone1 through Drone10. Launch the 12-drone settings and click Play.')
finally:
    c.client.close()
'@
& $Python -c $probe
if ($LASTEXITCODE -ne 0) { throw 'AirSim is not ready. Open the 12-drone project, click Play and rerun.' }
$previous = Join-Path $PSScriptRoot 'results\boundary_takeover_20261101_v1'
& $Python -u -m controller_architecture_study.run_reviewed_rate_trial --seed 6 --boundary-report (Join-Path $previous 'boundary\report.json') --ranker (Join-Path $previous 'ranker\ranker.npz') --output $application
if ($LASTEXITCODE -ne 0) { throw 'Paired run interrupted. Partial evidence retained; return the report and terminal output.' }
Write-Host "Completed: $application\summary.json"
Write-Host 'Inspect CPU service gaps, epoch-only refreshes, changed-rate deliveries before collision, collision counts and missing J separately.'
Write-Host 'Command completion does not imply hard deadlines or a demonstrated control benefit.'
