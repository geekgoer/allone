param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [int]$Seed = 0,
    [switch]$Search,
    [switch]$NonInteractive
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectPath
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $taskArgs = @('-u','-m','controller_architecture_study.heterogeneous_motivation',
        '--seed',"$Seed",'--difficulty','0.75','--width','1.7')
    if ($Search) {
        $taskArgs += @('--budget','20')
    } else {
        $selectionPath = Join-Path $PSScriptRoot 'results\heterogeneous_search20_20261028_v2\selection.json'
        if (-not (Test-Path -LiteralPath $selectionPath)) { throw 'The frozen20-candidate selection is not ready.' }
        $taskArgs += @('--selection',$selectionPath)
    }
    if (-not $NonInteractive) { $taskArgs += '--interactive' }
    Write-Host 'Ten heterogeneous loops; equal physical start phase; each measured flight lasts14.7s.'
    Write-Host 'Only a collided vehicle is retired. Other vehicles and simulation time continue.'
    Write-Host 'Cached J is an estimate. This is a finite-search prototype, not full FALCON or certified safety.'
    & $Python @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Experiment did not complete; preserve its output and reports.' }
} finally {
    Pop-Location
}
