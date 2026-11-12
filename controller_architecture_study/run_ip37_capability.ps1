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
# Keep the CPU teacher diagnostic single-threaded; this is not a PL timing test.
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$root = Join-Path $PSScriptRoot ('results\ip37_capability_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $root | Out-Null
$rows = @()
foreach ($controller in @('board20', 'teacher20')) {
    $out = Join-Path $root $controller
    $extra = @()
    if ($controller -eq 'teacher20') { $extra = @('--reference-teacher') }
    & $Python -u -m controller_architecture_study.fly_ip37_joint --bundle $Bundle `
        --board $Board --board-port $BoardPort --case $CaseFile --policy fixed `
        --period-ms 20 --vehicles Drone1 --output $out @extra
    $processExit = $LASTEXITCODE
    $report = Join-Path $out 'result.json'
    if (-not (Test-Path -LiteralPath $report)) { throw "No result file; exit=$processExit, directory=$out" }
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $rows += [pscustomobject]@{controller=$controller; directory=$out; exit_code=$processExit; result=$result}
    [IO.File]::WriteAllText((Join-Path $root 'summary.json'), ($rows | ConvertTo-Json -Depth 35), [Text.UTF8Encoding]::new($false))
    $result.errors | ForEach-Object { Write-Warning $_ }
}
$rows | ForEach-Object {
    $vehicle = $_.result.vehicles.Drone1
    [pscustomobject]@{controller=$_.controller; flight=$_.result.flight_passed;
        body_clearance=$_.result.body_clearance_passed; timing=$_.result.timing_before_contact_passed;
        clock=$_.result.sim_clock_passed; contacts=@($vehicle | Where-Object { $null -ne $_.contact }).Count;
        gates=@($vehicle.crossings | Where-Object valid).Count;
        response_p99_ms=$_.result.response_ms.p99; response_max_ms=$_.result.response_ms.max_observed}
} | Format-Table -AutoSize
Write-Host 'Teacher20 uses the original CPU QP teacher. This is a capability diagnostic, not an FPGA or dynamic-policy comparison.'
Write-Host "Summary: $root\summary.json"
