param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [int]$Seed = 3,
    [ValidateSet('observe','enforce')][string]$DeadlinePolicy = 'enforce',
    [switch]$Interactive
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$phasePath = Join-Path $PSScriptRoot 'results\anticipatory_repair_20261028_v1'
Push-Location -LiteralPath $projectPath
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist.' }
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $taskArgs = @('-u','-m','controller_architecture_study.heterogeneous_motivation',
        '--selection',(Join-Path $PSScriptRoot 'results\heterogeneous_search20_20261028_v2\selection.json'),
        '--mode-calibration',(Join-Path $phasePath 'capability_other12_calibration0'),
        '--pipeline-costs',(Join-Path $phasePath 'pipeline_costs.json'),
        '--anticipatory','--deadline-policy',$DeadlinePolicy,'--seed',"$Seed",'--difficulty','0.75','--width','1.7')
    if ($Interactive) { $taskArgs += '--interactive' }
    Write-Host "Deadline policy: $DeadlinePolicy. Validation geometry seed: $($Seed+1)."
    Write-Host 'Enforce rejects an inadmissible task set before starting the control mission.'
    Write-Host 'Observe continues after lateness for diagnosis; it is not a hard-real-time success test.'
    & $Python @taskArgs
    if ($LASTEXITCODE -ne 0) { throw 'Infrastructure failure; preserve reports.' }
    Write-Host 'Experiment finished. Read summary.json; process exit0 does not mean flight acceptance.'
} finally {
    Pop-Location
}
