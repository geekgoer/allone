param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in Administrator PowerShell. No trace or flight has started.'
}
Set-Location -LiteralPath $project
$status = (& wpr.exe -status 2>&1 | Out-String)
if ($status -notmatch 'WPR is not recording') { throw 'Cannot confirm WPR is idle. Existing traces were not changed.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $PSScriptRoot "results\ip37_kernel_$stamp"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$started = $false
try {
    & wpr.exe -start GeneralProfile -filemode
    if ($LASTEXITCODE -ne 0) { throw 'WPR failed to start.' }
    $started = $true
    & $Python -u -m controller_architecture_study.fly_ip37_joint `
        --bundle 'F:\FalconReproduction\237D_data\rec\ip37_h4_input8_rc1_20261106' `
        --board 192.168.77.2 --board-port 5238 `
        --case (Join-Path $PSScriptRoot 'results\two_loop_mixed_speed_20261108_v1\case.json') `
        --policy adaptive --period-ms 50 --periods-ms 20 30 40 50 60 80 120 `
        --initial-periods (Join-Path $PSScriptRoot 'results\two_loop_allocation_20261108_v1\both50.json') `
        --vehicles Drone1 Drone4 --evaluation-ms 50 --trace-thread-cycles --output $out
} finally {
    if ($started) {
        & wpr.exe -stop "$out.etl"
        if ($LASTEXITCODE -ne 0) { Write-Warning 'WPR failed to save. Keep the terminal output.' }
    }
}
Write-Host "Flight results: $out"
Write-Host "Kernel trace: $out.etl"
