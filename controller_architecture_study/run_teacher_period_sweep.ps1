param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$CaseFile,
    [string]$PeriodGrid = '30,50,80,120',
    [ValidateSet('baseline','gain5','gain7','qp_ff','qp_ff_gain7')]
    [string]$TeacherVariant = 'gain7'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$periods = @($PeriodGrid.Split(',') | ForEach-Object {
    [double]::Parse($_.Trim(), [Globalization.CultureInfo]::InvariantCulture)
})
if (@($periods | Where-Object { -not ($_ -gt 0 -and $_ -lt 160) }).Count -gt 0 -or
    @($periods | Select-Object -Unique).Count -ne $periods.Count) {
    throw 'PeriodGrid must contain unique finite periods in (0,160) ms, separated by commas'
}
if (-not $CaseFile) { $CaseFile = Join-Path $PSScriptRoot 'results\ip37_safety_design_20261107\seed0_w21\case.json' }
foreach ($requiredPath in @($Python, $CaseFile)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Missing file: $requiredPath" }
}
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$root = Join-Path $PSScriptRoot ('results\teacher_period_sweep_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $root | Out-Null
$rows = @()
$frozenSignature = $null
foreach ($period in $periods) {
    $out = Join-Path $root "T$period"
    Write-Host "Fixed teacher=$TeacherVariant, T=${period}ms; no injected delay"
    & $Python -u -m controller_architecture_study.fly_ip37_joint --bundle $PSScriptRoot `
        --reference-teacher --teacher-variant $TeacherVariant --case $CaseFile --policy fixed `
        --period-ms $period --periods-ms $periods --vehicles Drone1 --output $out
    $processExit = $LASTEXITCODE
    $report = Join-Path $out 'result.json'
    if (-not (Test-Path -LiteralPath $report)) { throw "No result file; exit=$processExit, directory=$out" }
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $setupPath = Join-Path $out 'setup.json'
    if (Test-Path -LiteralPath $setupPath) {
        $setup = Get-Content -Raw -LiteralPath $setupPath | ConvertFrom-Json
        $signature = [pscustomobject]@{case_sha256=$setup.case_sha256; teacher=$setup.teacher_signature} | ConvertTo-Json -Depth 20 -Compress
        if ($null -eq $frozenSignature) { $frozenSignature = $signature }
        elseif ($signature -ne $frozenSignature) { throw 'Teacher or case changed between periods; preserve all reports' }
    }
    $rows += [pscustomobject]@{teacher_variant=$TeacherVariant; period_ms=$period; directory=$out; exit_code=$processExit; result=$result}
    [IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($rows | ConvertTo-Json -Depth 35), [Text.UTF8Encoding]::new($false))
    $result.errors | ForEach-Object { Write-Warning $_ }
}
$rows | ForEach-Object {
    $vehicle = $_.result.vehicles.Drone1
    $crossed = @($vehicle.crossings | Where-Object crossed)
    $margin = $null
    if ($crossed.Count -eq $vehicle.expected_gates -and $crossed.Count -gt 0) {
        $margin = ($crossed | Measure-Object -Property body_margin_m -Minimum).Minimum
    }
    $rmse = $null
    if ($null -ne $vehicle.tracking_mse_m2) { $rmse = [math]::Sqrt($vehicle.tracking_mse_m2) }
    [pscustomobject]@{T_ms=$_.period_ms; flight=$_.result.flight_passed;
        body=$_.result.body_clearance_passed; timing=$_.result.timing_before_contact_passed;
        clock=$_.result.sim_clock_passed; gates=@($vehicle.crossings | Where-Object valid).Count;
        contacts=@($vehicle | Where-Object { $null -ne $_.contact }).Count;
        RMSE_m=$rmse; min_full_margin_m=$margin; R_mean_ms=$_.result.response_ms.mean;
        R_p99_ms=$_.result.response_ms.p99; outcomes=($_.result.outcomes | ConvertTo-Json -Compress)}
} | Format-Table -AutoSize
Write-Host "Period screening only: identical $TeacherVariant teacher and route; CPU control, not DNN/FPGA. All timing failures remain reported."
Write-Host "Summary: $root\summary.json"
