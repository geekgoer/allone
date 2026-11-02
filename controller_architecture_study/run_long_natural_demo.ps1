param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectPath
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Write-Host 'Physics demo: fixed80ms versus state-aware control. Not a global FALCON optimum or safety certificate.'
    Write-Host 'A stronger static60ms configuration also passed; this is not evidence that offline-optimal FALCON collides.'
    Write-Host 'Keep the12-vehicle AirSim project in Play. Each flight takes about6min15s.'
    & $Python -u -m controller_architecture_study.long_route_demo --selection '.\controller_architecture_study\results\long_route_extended_prefix_20261027_v9\selection.json'
    if ($LASTEXITCODE -ne 0) { throw 'Demo did not complete; preserve the report and terminal output.' }
} finally {
    Pop-Location
}
