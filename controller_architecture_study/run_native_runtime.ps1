param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Seeds = '4,5',
    [ValidateSet('both','python','cpp')][string]$Backend = 'both',
    [switch]$IoOnly,
    [switch]$CompareRepair
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist' }
$env:PYTHONPATH = $null
$native = Join-Path $PSScriptRoot 'native_runtime\build\Release\falcon_native_io.dll'
if (-not (Test-Path -LiteralPath $native)) {
    & (Join-Path $PSScriptRoot 'build_native_runtime.ps1')
}
$arguments = @('-m','controller_architecture_study.run_native_runtime','--seeds',$Seeds,'--backend',$Backend)
if ($IoOnly) { $arguments += '--io-only' }
if ($CompareRepair) { $arguments += '--compare-repair' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Native runtime trial did not complete; retained reports contain the cause' }
