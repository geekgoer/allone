param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [switch]$RunGpuTiming
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }

& $Python -m controller_architecture_study.probe_state_value --preflight
if ($LASTEXITCODE -ne 0) { throw '237D model/data preflight failed.' }
if (-not $RunGpuTiming) {
    & $Python -m controller_architecture_study.probe_state_value --device cpu
    if ($LASTEXITCODE -ne 0) { throw 'Aligned state analysis failed.' }
    Write-Host 'CPU alignment complete. No GPU benchmark or AirSim session was started.'
    exit 0
}

Write-Host 'GPU timing: run only AFTER training has stopped; close AirSim for the replay baseline.'
# The same four weights stay resident in every case. No new model is trained.
$cases = @(
    @{ Clients = 1; Model = 'residual21x1024'; Phase = 'sync' },
    @{ Clients = 3; Model = 'residual21x1024'; Phase = 'sync' },
    @{ Clients = 3; Model = 'residual21x1024'; Phase = 'staggered' },
    @{ Clients = 3; Model = 'cycle'; Phase = 'sync' }
)
for ($round = 0; $round -lt 3; $round++) {
    for ($offset = 0; $offset -lt $cases.Count; $offset++) {
        # Rotate order to reduce thermal/order confounding; retain every report.
        $case = $cases[($offset + $round) % $cases.Count]
        Write-Host "Round $($round + 1): clients=$($case.Clients), model=$($case.Model), phase=$($case.Phase)"
        & $Python -m controller_architecture_study.probe_shared_response --device cuda --clients $case.Clients --model $case.Model --phase $case.Phase --duration 60
        if ($LASTEXITCODE -ne 0) { throw 'Replay failed; previous timestamped reports remain saved.' }
    }
}
Write-Host 'GPU replay complete: 12 one-minute trials plus model loading/warmup. This is NOT a flight result.'
