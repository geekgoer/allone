param(
    [string]$PythonPath = '',
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $PythonPath) {
    $localPython = Join-Path $project '.venv\Scripts\python.exe'
    $originalPython = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) { $PythonPath = $localPython }
    elseif (Test-Path -LiteralPath $originalPython) { $PythonPath = $originalPython }
    else { throw 'No Python found. Supply -PythonPath with the installed experiment python.exe.' }
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python not found: $PythonPath"
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot 'results\moderate_v2' }
$vendor = Join-Path $project 'vendor'
if (Test-Path -LiteralPath $vendor) { $env:PYTHONPATH = $vendor }
Set-Location -LiteralPath $project
Write-Host 'Continuous collection: resume existing results, no per-invocation trial limit.'
Write-Host 'Keep AirSim in Play and the current5ms UDP5008 board service running.'
Write-Host 'Mission failures are recorded; infrastructure failures stop the batch. No automatic retry-until-success.'
& $PythonPath (Join-Path $PSScriptRoot 'collect.py') `
    --config (Join-Path $PSScriptRoot 'collection_config_moderate_v2.json') `
    --enable-fpga-actuation --output-root $OutputRoot
exit $LASTEXITCODE
