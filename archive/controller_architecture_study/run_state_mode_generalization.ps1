param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python path does not exist' }
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.tmp\mplconfig'
& $Python -m controller_architecture_study.state_mode_generalization --run
if ($LASTEXITCODE -ne 0) { throw 'Generalization interrupted; results retained. Inspect before resuming.' }
& $Python -m controller_architecture_study.state_mode_fast_teacher --run
if ($LASTEXITCODE -ne 0) { throw 'Fast-teacher diagnostics interrupted; results retained.' }
& $Python -m controller_architecture_study.analyze_state_mode_generalization
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed; flight results retained.' }
& $Python -m controller_architecture_study.analyze_state_mode_fast_teacher
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic analysis failed; flight results retained.' }
