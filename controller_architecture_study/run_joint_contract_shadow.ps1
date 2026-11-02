param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [int]$Seed = 4,
    [switch]$ShadowOnly,
    [switch]$RecoverAfterMiss
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
$library = Join-Path $PSScriptRoot 'native_runtime\build\Release\falcon_native_io.dll'
$nativeSource = Join-Path $PSScriptRoot 'native_runtime\native_io.cpp'
if (-not (Test-Path -LiteralPath $library) -or (Get-Item -LiteralPath $nativeSource).LastWriteTimeUtc -gt (Get-Item -LiteralPath $library).LastWriteTimeUtc) {
    & (Join-Path $PSScriptRoot 'build_native_runtime.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Native runtime build failed.' }
}
$output = Join-Path $PSScriptRoot ('results\joint_contract_shadow_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$arguments = @('-m', 'controller_architecture_study.run_joint_contract_shadow', '--output', $output, '--seed', "$Seed")
if ($ShadowOnly) { $arguments += '--shadow-only' }
if ($RecoverAfterMiss) { $arguments += '--recover-after-miss' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Shadow pilot failed; reports were retained.' }
Write-Host ('Report: ' + (Join-Path $output 'summary.json'))
