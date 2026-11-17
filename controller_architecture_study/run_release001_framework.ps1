param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string[]]$Vehicles = @('Drone1', 'Drone2'),
    [ValidateSet('fixed', 'adaptive')][string]$Policy = 'adaptive',
    [int]$ModelId = 11,
    [int[]]$Models = @(5, 10, 13, 8, 11, 6),
    [double]$PeriodMs = 50,
    [double]$ReferenceStartSeconds = 0,
    [double]$AdaptationDelayMs = 0,
    [string]$Case = '',
    [switch]$EnablePlannedSkip
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$delivery = Join-Path $PSScriptRoot 'docs\main18_requant_release001_delivery_20261115'
$bundle = Join-Path $delivery 'board_package_requant18_release001'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$output = Join-Path $PSScriptRoot "results\release001_framework_$stamp"
Set-Location -LiteralPath $project
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$arguments = @(
    '-u', '-m', 'controller_architecture_study.fly_ip37_joint',
    '--scheduler', 'obligation', '--delivery', $delivery, '--bundle', $bundle,
    '--board', '192.168.77.2', '--board-port', '5239',
    '--policy', $Policy, '--model-id', "$ModelId", '--model-ids'
) + @($Models | ForEach-Object { "$_" }) + @(
    '--period-ms', "$PeriodMs", '--periods-ms', '30', '40', '50', '60', '80',
    '--decision-ms', '5', '--reference-start-s', "$ReferenceStartSeconds",
    '--adaptation-delay-ms', "$AdaptationDelayMs", '--vehicles'
) + $Vehicles + @('--output', $output)
if ($Case) { $arguments += @('--case', $Case) }
if ($EnablePlannedSkip) { $arguments += '--enable-planned-skip' }
& $Python @arguments
$code = $LASTEXITCODE
Write-Host "Results: $output"
exit $code
