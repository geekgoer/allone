param(
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateSet(3,7,8,12)][int]$Vehicles = 8,
    [string]$ModeMap = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist.' }
    $map = $ModeMap
    if ($map -and -not (Test-Path -LiteralPath $map)) { throw 'Explicit mode map does not exist.' }
    if (-not $map) {
        $map = '.\controller_architecture_study\results\framework_runtime_refinement_v1\mode_map.json'
        if (-not (Test-Path -LiteralPath $map)) { $map = '.\controller_architecture_study\results\framework_runtime_calibration_v2\mode_map.json' }
    }
    if (-not (Test-Path -LiteralPath $map)) {
        & $Python -m controller_architecture_study.framework_runtime_calibrate
        if ($LASTEXITCODE -ne 0) { throw 'Empirical recovery calibration failed; retained results must be inspected.' }
    }
    & $Python -m pytest tests/test_framework_cpu_affinity.py tests/test_framework_runtime.py tests/test_safety_comparison.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Runtime regression failed.' }
    & $Python -m controller_architecture_study.run_framework_runtime --stage smoke --vehicles 3 --mode-map $map
    if ($LASTEXITCODE -ne 0) { throw 'Three-loop integration failed; no larger flight was started.' }
    if ($Vehicles -ne 3) {
        & $Python -m controller_architecture_study.run_framework_runtime --stage smoke --vehicles $Vehicles --mode-map $map
        if ($LASTEXITCODE -ne 0) { throw 'Requested-loop nominal integration failed; no safety advantage is claimed.' }
    }
    foreach ($stage in @('calibrate','discover','confirm')) {
        & $Python -m controller_architecture_study.run_framework_runtime --stage $stage --vehicles $Vehicles --mode-map $map
        if ($LASTEXITCODE -ne 0) { throw "Runtime stage $stage failed; all results retained." }
    }
} finally {
    Pop-Location
}
