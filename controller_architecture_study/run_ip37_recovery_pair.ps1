param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Bundle = 'F:\FalconReproduction\237D_data\rec\ip37_h4_input8_rc1_20261106',
    [string]$CaseFile,
    [string]$Board = '192.168.77.2',
    [int]$BoardPort = 5238
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not $CaseFile) { $CaseFile = Join-Path $PSScriptRoot 'results\ip37_safety_design_20261107\seed0_w21\case.json' }
foreach ($requiredPath in @($Python, $CaseFile, (Join-Path $Bundle 'manifest.json'))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Missing file: $requiredPath" }
}
$root = Join-Path $PSScriptRoot ('results\ip37_recovery_pair_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $root | Out-Null
$rows = @()
foreach ($period in @(80,30)) {
    $out = Join-Path $root "fixed$period"
    & $Python -u -m controller_architecture_study.fly_ip37_joint --bundle $Bundle `
        --board $Board --board-port $BoardPort --case $CaseFile --policy fixed `
        --period-ms $period --vehicles Drone1 --output $out
    $processExit = $LASTEXITCODE
    $report = Join-Path $out 'result.json'
    if (-not (Test-Path -LiteralPath $report)) { throw "No result file; exit=$processExit, directory=$out" }
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $rows += [pscustomobject]@{period_ms=$period; directory=$out; exit_code=$processExit; result=$result}
    [IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($rows | ConvertTo-Json -Depth 35), [Text.UTF8Encoding]::new($false))
    if ($result.errors.Count -gt 0) {
        $result.errors | ForEach-Object { Write-Warning $_ }
    }
}
$slow = $rows[0].result
$fast = $rows[1].result
$timingOK = $slow.timing_before_contact_passed -and $fast.timing_before_contact_passed -and `
    $slow.sim_clock_passed -and $fast.sim_clock_passed -and `
    $slow.calendar_coverage_passed -and $fast.calendar_coverage_passed
$slowSafe = $slow.flight_passed -and $slow.body_clearance_passed
$fastSafe = $fast.flight_passed -and $fast.body_clearance_passed
$classification = if ($slow.errors.Count -gt 0 -or $fast.errors.Count -gt 0) { 'runtime_or_input_inconclusive' } `
    elseif (-not $timingOK) { 'timing_inconclusive' } `
    elseif (-not $slowSafe -and $fastSafe) { 'candidate_period_recovery_gap' } `
    elseif ($slowSafe -and $fastSafe) { 'both_pass_no_gap_at_this_width' } `
    elseif ($slowSafe -and -not $fastSafe) { 'faster_period_did_not_improve' } `
    else { 'both_failed_no_recovery_demonstrated' }
[IO.File]::WriteAllText((Join-Path $root 'classification.json'), `
    ([pscustomobject]@{classification=$classification; timing_comparable=$timingOK; `
        slow_passed=$slowSafe; fast_passed=$fastSafe; repeated_evidence=$false} | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
$rows | ForEach-Object {
    $margins = $_.result.vehicles.Drone1.crossings | Where-Object crossed | ForEach-Object body_margin_m
    [pscustomobject]@{period_ms=$_.period_ms; flight=$_.result.flight_passed;
        body_clearance=$_.result.body_clearance_passed; timing=$_.result.timing_before_contact_passed;
        contacts=@($_.result.vehicles.PSObject.Properties.Value | Where-Object { $null -ne $_.contact }).Count;
        min_body_margin_m=($margins | Measure-Object -Minimum).Minimum;
        response_p99_ms=$_.result.response_ms.p99; requests=$_.result.requests}
} | Format-Table -AutoSize
Write-Host "Classification: $classification (screening only; repeat before claiming a recovery effect)"
Write-Host "Summary: $root\summary.json"
