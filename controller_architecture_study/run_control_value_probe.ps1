param(
    [Parameter(Mandatory=$true)][string]$Python
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable not found.' }
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    & $Python -m controller_architecture_study.control_value_probe --device cuda --resume
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Probe stopped. Completed trials are retained; fix the reported issue and rerun this same command.'
        exit $LASTEXITCODE
    }
    $outputPath = Join-Path $PSScriptRoot 'results\control_value_237d_v1'
    $report = Get-Content -Raw -LiteralPath (Join-Path $outputPath 'summary.json') | ConvertFrom-Json
    $report.aggregates | Select-Object scenario, model, period_ms, extra_delay_ms, trials, comparable, invalid, comparable_mean_mse_m2 | Format-Table -AutoSize
    Write-Host "Completed: $($report.completed_trials); comparable: $($report.comparable_trials); invalid: $($report.failed_or_timing_invalid_trials)"
    Write-Host "Summary: $(Join-Path $outputPath 'summary.json')"
    Write-Host 'Compare paired differences and invalid counts; this pilot is not a safety or supernet-necessity certificate.'
} finally {
    Pop-Location
}
