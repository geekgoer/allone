param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [ValidateSet('Pilot','Collect','Train','All')][string]$Stage = 'Pilot',
    [string]$Data = '',
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist.' }
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not $Data) { $Data = Join-Path $PSScriptRoot 'results\controller_selector_237d_v2\data' }
if (-not $Output) { $Output = Join-Path (Split-Path -Parent $Data) 'predictor' }
if (-not (Test-Path -LiteralPath (Join-Path $Data 'plan.json'))) {
    if ($Stage -eq 'Train') { throw 'Collect paired recovery data before training.' }
    & $Python -m controller_architecture_study.controller_selector.collect prepare --output $Data
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
}
if ($Stage -in @('Pilot','Collect','All')) {
    $collectArgs = @('-m','controller_architecture_study.controller_selector.collect','collect','--output',$Data)
    if ($Stage -eq 'Pilot') { $collectArgs += @('--max-groups','2') }
    & $Python @collectArgs
    if ($LASTEXITCODE -ne 0) { throw 'Collection stopped; all completed trials and invalid-attempt diagnostics were retained.' }
}
if ($Stage -in @('Train','All')) {
    $trainArgs = @('-m','controller_architecture_study.controller_selector.train','--data',$Data,'--output',$Output,'--epochs','600')
    if (Test-Path -LiteralPath (Join-Path $Output 'latest_checkpoint.pt')) { $trainArgs += '--resume' }
    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw 'Selector training/export failed; existing controller models are unchanged.' }
}
Write-Host ('Collection: ' + (Join-Path $Data 'collection_summary.json'))
if ($Stage -in @('Train','All')) { Write-Host ('Predictor: ' + (Join-Path $Output 'training_report.json')) }
