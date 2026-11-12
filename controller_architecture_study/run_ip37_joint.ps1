param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Bundle = 'F:\FalconReproduction\237D_data\rec\ip37_h4_input8_rc1_20261106',
    [string]$Board = '192.168.77.2',
    [int]$BoardPort = 5238,
    [int]$Drones = 3
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist' }
if (-not (Test-Path -LiteralPath (Join-Path $Bundle 'manifest.json'))) { throw 'Candidate bundle missing' }
if ($Drones -lt 1 -or $Drones -gt 12) { throw 'Drones must be 1..12' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$root = Join-Path $PSScriptRoot "results\ip37_joint_$stamp"
New-Item -ItemType Directory -Path $root | Out-Null
$vehicles = 1..$Drones | ForEach-Object { "Drone$_" }
$common = @('-u', '-m', 'controller_architecture_study.fly_ip37_joint',
    '--bundle', $Bundle, '--board', $Board, '--board-port', "$BoardPort", '--model-id', '11',
    '--period-ms', '50', '--gate-width', '4', '--vehicles') + $vehicles
$summaries = @()
foreach ($policy in @('fixed', 'adaptive')) {
    $out = Join-Path $root $policy
    & $Python @common --policy $policy --output $out
    $exit = $LASTEXITCODE
    $report = Join-Path $out 'result.json'
    if (-not (Test-Path -LiteralPath $report)) { throw "No result file; exit=$exit, directory=$out" }
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $summaries += [pscustomobject]@{ policy=$policy; directory=$out; exit_code=$exit; result=$result }
    if ($result.errors.Count -gt 0) {
        [IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($summaries | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
        $result.errors | ForEach-Object { Write-Warning $_ }
        throw "Runtime/input/planner error; inspect $report"
    }
}
[IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($summaries | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
$summaries | ForEach-Object {
    [pscustomobject]@{policy=$_.policy; flight=$_.result.flight_passed;
        timing=$_.result.timing_passed; clock=$_.result.sim_clock_passed;
        updates=$_.result.period_updates; requests=$_.result.requests;
        response_p99_ms=$_.result.response_ms.p99; decisions=$_.result.decisions | ConvertTo-Json -Compress}
} | Format-Table -AutoSize
Write-Host "Summary: $root\summary.json"
