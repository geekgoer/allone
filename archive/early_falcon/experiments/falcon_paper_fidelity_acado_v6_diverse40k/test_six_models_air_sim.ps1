param(
    [ValidateSet("v6", "v4")]
    [string]$Variant = "v6",

    [ValidateSet(
        "all",
        "depth10_width512",
        "depth10_width1024",
        "depth22_width512",
        "depth22_width1024",
        "depth34_width512",
        "depth34_width1024"
    )]
    [string]$Model = "all",

    [string]$Python = "D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe",

    [ValidateSet(0.0, 0.8, 1.6)]
    [double]$Epsilon = 0.0,

    [int]$InitialStateId = 0,
    [double]$BaseY = 0.0,
    [double]$BaseZ = -2.8,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..\.." )).Path
$runner = Join-Path $projectRoot "run_single_drone_dnn_gates.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path -LiteralPath $runner)) {
    throw "AirSim DNN runner not found: $runner"
}

$listen = Get-NetTCPConnection -State Listen -LocalPort 41451 -ErrorAction SilentlyContinue
if (-not $listen) {
    throw "AirSim RPC port 41451 is not listening. Start UE4Editor, open Blocks.uproject, and click Play before running this script."
}

$modelNames = @(
    "depth10_width512",
    "depth10_width1024",
    "depth22_width512",
    "depth22_width1024",
    "depth34_width512",
    "depth34_width1024"
)
if ($Model -ne "all") {
    $modelNames = @($Model)
}

if ($Variant -eq "v6") {
    $runRoot = Join-Path $scriptDir "runs_full\seed_0"
    $layoutRoot = Join-Path $scriptDir "paper_core_40k_falcon12_v6_diverse"
    $reportRoot = Join-Path $scriptDir "flight_tests_v6"
    $expectedContract = "falcon_paper_gate_relative_euler_12d_v1"
}
else {
    $runRoot = Join-Path $projectRoot "experiments\falcon_paper_fidelity_acado_v4_independent\runs_balanced_120k_seed0\seed_0"
    $layoutRoot = Join-Path $projectRoot "teacher_dataset_full_route_continuous_v2"
    $reportRoot = Join-Path $scriptDir "flight_tests_v4_comparison"
    $expectedContract = "falcon_track_p_gate_relative_12d_v1"
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages"

$results = @()
try {
    foreach ($name in $modelNames) {
        $modelDir = Join-Path $runRoot $name
        $modelPath = Join-Path $modelDir "controller_scripted.pt"
        $normalizationPath = Join-Path $modelDir "normalization.json"
        $specPath = Join-Path $modelDir "model_spec.json"
        $reportPath = Join-Path $reportRoot ("{0}_seed0.json" -f $name)

        foreach ($required in @($modelPath, $normalizationPath, $specPath)) {
            if (-not (Test-Path -LiteralPath $required)) {
                throw "Required model artifact is missing: $required"
            }
        }

        $spec = Get-Content -Raw -LiteralPath $specPath | ConvertFrom-Json
        if ($spec.input_contract -ne $expectedContract) {
            throw "Input contract mismatch for ${name}: expected '$expectedContract', found '$($spec.input_contract)'"
        }

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Testing $Variant / $name" -ForegroundColor Cyan
        Write-Host "Input contract: $($spec.input_contract)" -ForegroundColor Cyan
        Write-Host "Route: epsilon=$Epsilon, initial_state=$InitialStateId, base=($BaseY,$BaseZ)" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan

        & $Python $runner `
            --model $modelPath `
            --normalization $normalizationPath `
            --input-target gate `
            --gate-attitude-encoding auto `
            --controller-backend cpu_dnn `
            --layout-root $layoutRoot `
            --initial-state-id $InitialStateId `
            --vehicle-name Drone1 `
            --device cpu `
            --epsilon $Epsilon `
            --seed 0 `
            --speed 5.0 `
            --base-y $BaseY `
            --base-z $BaseZ `
            --per-gate-timeout 30 `
            --gate-count 5 `
            --x-start 10 `
            --x-spacing 10 `
            --gate-width 4 `
            --gate-height 4 `
            --gate-thickness 0.15 `
            --gate-clearance 0.25 `
            --gate-asset EditorCube `
            --report $reportPath

        $exitCode = $LASTEXITCODE
        $results += [pscustomobject]@{
            variant = $Variant
            model = $name
            exit_code = $exitCode
            report = $reportPath
        }

        if ($exitCode -eq 0) {
            Write-Host "PASS: $name completed its AirSim run." -ForegroundColor Green
        }
        else {
            Write-Warning "FAIL: $name exited with code $exitCode. Continue testing the remaining models; do not treat this model as closed-loop accepted."
        }

        if (-not $NoPause -and $name -ne $modelNames[-1]) {
            Read-Host "Observe the final AirSim state, then press Enter to test the next model"
        }
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}

Write-Host ""
Write-Host "AirSim test summary" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$batchSummary = Join-Path $reportRoot ("batch_summary_{0}_{1}.json" -f $Variant, (Get-Date -Format "yyyyMMdd_HHmmss"))
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $batchSummary -Encoding UTF8
Write-Host "Batch summary: $batchSummary"

if (($results | Where-Object { $_.exit_code -ne 0 }).Count -gt 0) {
    exit 1
}
