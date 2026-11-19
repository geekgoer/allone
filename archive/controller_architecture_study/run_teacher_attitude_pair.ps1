param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$CaseFile
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not $CaseFile) { $CaseFile = Join-Path $PSScriptRoot 'results\ip37_safety_design_20261107\seed0_w21\case.json' }
foreach ($requiredPath in @($Python, $CaseFile)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Missing file: $requiredPath" }
}
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$root = Join-Path $PSScriptRoot ('results\teacher_attitude_flight_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $root | Out-Null
$rows = @()
foreach ($variant in @('baseline', 'gain7', 'qp_ff', 'qp_ff_gain7')) {
    $out = Join-Path $root $variant
    Write-Host "Teacher comparison: $variant, T=30ms"
    # The shared driver requires a bundle path; this teacher-only branch never
    # opens that directory or a board socket.
    & $Python -u -m controller_architecture_study.fly_ip37_joint --bundle $PSScriptRoot `
        --reference-teacher --teacher-variant $variant --case $CaseFile `
        --policy fixed --period-ms 30 --vehicles Drone1 --output $out
    $processExit = $LASTEXITCODE
    $report = Join-Path $out 'result.json'
    if (-not (Test-Path -LiteralPath $report)) { throw "No result file; exit=$processExit, directory=$out" }
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $rows += [pscustomobject]@{variant=$variant; directory=$out; exit_code=$processExit; result=$result}
    [IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($rows | ConvertTo-Json -Depth 35), [Text.UTF8Encoding]::new($false))
    $result.errors | ForEach-Object { Write-Warning $_ }
}
$rows | ForEach-Object {
    $vehicle = $_.result.vehicles.Drone1
    $gate4 = $vehicle.crossings | Where-Object { $_.gate -eq 4 }
    [pscustomobject]@{variant=$_.variant; flight=$_.result.flight_passed;
        body_clearance=$_.result.body_clearance_passed; timing=$_.result.timing_before_contact_passed;
        clock=$_.result.sim_clock_passed; gates=@($vehicle.crossings | Where-Object valid).Count;
        contacts=@($vehicle | Where-Object { $null -ne $_.contact }).Count;
        gate4_lateral_m=$gate4.lateral_offset_m; response_p99_ms=$_.result.response_ms.p99;
        query_p99_ms=$_.result.teacher_solve_ms.p99}
} | Format-Table -AutoSize
Write-Host 'All four runs use the CPU teacher at T=30ms, identical route and gates; no board inference or model retraining.'
Write-Host "Summary: $root\summary.json"
