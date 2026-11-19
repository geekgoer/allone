param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable not found' }

if ($Preflight) {
    & $Python -m controller_architecture_study.replay_online_joint --preflight
    exit $LASTEXITCODE
}

Write-Host 'Close AirSim/UE4 and other GPU training first. No training or simulator is started here.'
Write-Host 'Three policies, forward/reverse rounds: 216 seconds measured, plus loading and warmup.'
& $Python -m controller_architecture_study.replay_online_joint --device cuda --clients 3 8 16 32 8 3 --stage-seconds 6 --rounds 2 --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$summaryPath = Join-Path $projectPath 'controller_architecture_study\results\dynamic_complexity_237d_v1\online_joint_pool_v2\summary.json'
$summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
$summary.trials | ForEach-Object {
    [pscustomobject]@{
        case = $_.case
        requests = $_.timing.released
        misses = $_.timing.deadline_misses
        miss_pct = [math]::Round(100 * $_.timing.miss_fraction, 4)
        common50_misses = $_.timing.common_response_limit_misses.'50'
        age_ms = [math]::Round($_.output_age.mean_age_when_served_ms, 3)
        model_switches = $_.model_switches
        T_switches = $_.period_switches
        max_pause_ms = [math]::Round($_.producer_pause_ms.max_observed, 3)
    }
} | Format-Table -AutoSize
Write-Host "Summary: $summaryPath"
