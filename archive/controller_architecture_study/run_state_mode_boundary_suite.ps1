param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist' }
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:TEMP = Join-Path (Get-Location).Path '.tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'mplconfig'

& $Python -m controller_architecture_study.state_mode_boundary --phase discovery
if ($LASTEXITCODE -ne 0) { throw 'Discovery interrupted; saved trials retained' }
& $Python -m controller_architecture_study.state_mode_boundary --phase confirmation
if ($LASTEXITCODE -ne 0) { throw 'Confirmation interrupted; saved trials retained' }
& $Python -m controller_architecture_study.state_mode_period_boundary
if ($LASTEXITCODE -ne 0) { throw 'Period pilot interrupted; saved trials retained' }
& $Python -m controller_architecture_study.state_mode_period_refinement
if ($LASTEXITCODE -ne 0) { throw 'Bounded midpoint refinement interrupted; saved trials retained' }
& $Python -m controller_architecture_study.analyze_state_mode_boundary
if ($LASTEXITCODE -ne 0) { throw 'Boundary presentation failed; raw flight reports retained' }
& $Python -m controller_architecture_study.analyze_state_mode_period
if ($LASTEXITCODE -ne 0) { throw 'Period presentation failed; raw flight reports retained' }
Write-Host 'Bounded experiment suite complete. Existing finished trials were not rerun.'
