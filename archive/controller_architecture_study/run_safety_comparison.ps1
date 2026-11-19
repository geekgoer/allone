param(
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateSet(3,8,12)][int]$Vehicles = 8,
    [ValidateSet(2.0,1.5)][double]$NetWidth = 2.0
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable does not exist.' }
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    # Calibration also requires the selected configuration's dynamic nominal run
    # to pass. A fixed small-model smoke is only an optional diagnostic.
    foreach ($stage in @('calibrate','discover','confirm')) {
        & $Python -m controller_architecture_study.run_safety_comparison --stage $stage --vehicles $Vehicles --net-width $NetWidth
        if ($LASTEXITCODE -ne 0) {
            throw "Stage $stage failed. Results are retained; inspect before retrying."
        }
    }
} finally {
    Pop-Location
}
