param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Seeds = '4,5',
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
$library = Join-Path $PSScriptRoot 'native_runtime\build\Release\falcon_native_io.dll'
$source = Join-Path $PSScriptRoot 'native_runtime\native_io.cpp'
if (-not (Test-Path -LiteralPath $library) -or (Get-Item -LiteralPath $source).LastWriteTimeUtc -gt (Get-Item -LiteralPath $library).LastWriteTimeUtc) {
    & (Join-Path $PSScriptRoot 'build_native_runtime.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed.' }
}
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot ('results\joint_runtime_repair\' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmss_fffffffZ'))
}
& $Python -m controller_architecture_study.run_native_runtime --compare-joint --seeds $Seeds --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Paired run failed; completed reports were retained.' }
& $Python -m controller_architecture_study.analyze_joint_runtime $Output
if ($LASTEXITCODE -ne 0) { throw 'Result audit failed.' }
Write-Host ('Audit: ' + (Join-Path $Output 'comparison_audit.json'))
