param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [string]$Bundle = 'F:\FalconReproduction\237D_data\rec\ip37_h4_input8_rc1_20261106',
    [string]$Output = 'F:\FalconReproduction\airsim_control\controller_architecture_study\results\ip37_campaign_2h_20261108_v2',
    [int[]]$Drones = @(4,7,10,12),
    [double[]]$Widths = @(1.9,2.4,3.0),
    [int[]]$Seeds = @(0),
    [int]$SearchBudget = 12,
    [int]$Repeats = 3,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$campaignArgs = @('-u','-m','controller_architecture_study.ip37_campaign',
    '--bundle',$Bundle,'--output',$Output,'--resume','--search-budget',"$SearchBudget",'--repeats',"$Repeats",'--drones')
$campaignArgs += @($Drones | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) })
$campaignArgs += '--widths'
$campaignArgs += @($Widths | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) })
$campaignArgs += '--seeds'
$campaignArgs += @($Seeds | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) })
if ($PlanOnly) { $campaignArgs += '--plan-only' }
& $Python @campaignArgs
if ($LASTEXITCODE -ne 0) {
    throw "Campaign interrupted; reports retained at $Output. Fix the displayed cause, then rerun the same command to resume."
}
if ($PlanOnly) {
    Write-Host "Offline plan: $Output\plan.json ; cases\ ; no flight started"
} else {
    Write-Host "Reports: $Output\summary.json ; cells.csv ; trials.csv ; pairs.csv"
}
