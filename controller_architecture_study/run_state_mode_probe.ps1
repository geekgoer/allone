param(
    [Parameter(Mandatory=$true)][string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cpu',
    [switch]$PlanOnly,
    [switch]$SummarizeOnly,
    [ValidateRange(0,2147483647)][int]$MaxTrials = 0
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python executable not found.' }
if ($PlanOnly -and $SummarizeOnly) { throw 'Choose PlanOnly or SummarizeOnly.' }
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $probeArgs = @('-m', 'controller_architecture_study.state_mode_probe', '--device', $Device, '--resume')
    if ($PlanOnly) { $probeArgs += '--plan-only' }
    if ($SummarizeOnly) { $probeArgs += '--summarize-only' }
    if ($MaxTrials -gt 0) { $probeArgs += @('--max-trials', "$MaxTrials") }
    & $Python @probeArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'E1 stopped. Completed trials and diagnostics are retained; inspect the error and rerun the same command after resolving it.'
    }
    if (-not $PlanOnly) {
        $summaryPath = Join-Path $PSScriptRoot 'results\state_mode_admissibility_v2\summary.json'
        $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
        $summary.rows | Select-Object case,mode,completed,expected,joint_successes,timing_invalid,status | Format-Table -AutoSize
        Write-Host "Saved: $($summary.completed_trials)/$($summary.expected_trials); joint successes: $($summary.joint_successes)"
        Write-Host "Summary: $summaryPath"
        Write-Host 'Heatmap: controller_architecture_study/results/state_mode_admissibility_v2/recovery_heatmap.svg'
    }
} finally {
    Pop-Location
}
