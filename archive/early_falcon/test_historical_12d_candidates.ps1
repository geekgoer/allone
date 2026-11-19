param(
    [ValidateSet(
        "robust2ms",
        "paper5ms_d10w512",
        "paper5ms_d10w1024",
        "paper5ms_d22w512",
        "paper5ms_d22w1024",
        "paper5ms_d34w512",
        "paper5ms_d34w1024",
        "paper5ms_all"
    )]
    [string]$Candidate = "robust2ms",
    [string]$Python = "D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe",
    [double]$Epsilon = 0.8,
    [int]$Seed = 0,
    [int]$GateCount = 5,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$catalog = [ordered]@{
    robust2ms = @{
        RelativePath = "dnn_runs\track_p_12d_robust_v1_depth10_width512"
        Speed = 2.0
        Evidence = "historically passed 5 gates at epsilon=0.8 and 1.6, plus an 8-gate test"
    }
    paper5ms_d10w512 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_serial_v2\paper_track_p_12d_5ms_v1_serial_v2_depth10_width512"
        Speed = 5.0
        Evidence = "12-D artifact exists; no stored closed-loop acceptance report"
    }
    paper5ms_d10w1024 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_serial_v2\paper_track_p_12d_5ms_v1_serial_v2_depth10_width1024"
        Speed = 5.0
        Evidence = "12-D artifact exists; no stored closed-loop acceptance report"
    }
    paper5ms_d22w512 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_serial_v2\paper_track_p_12d_5ms_v1_serial_v2_depth22_width512"
        Speed = 5.0
        Evidence = "12-D artifact exists; no stored closed-loop acceptance report"
    }
    paper5ms_d22w1024 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_server\paper_track_p_12d_5ms_v1_server_depth22_width1024"
        Speed = 5.0
        Evidence = "historically passed all 5 gates at epsilon=0.8"
    }
    paper5ms_d34w512 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_server\paper_track_p_12d_5ms_v1_server_depth34_width512"
        Speed = 5.0
        Evidence = "12-D artifact exists; no stored closed-loop acceptance report"
    }
    paper5ms_d34w1024 = @{
        RelativePath = "experiments\paper_track_p_12d_5ms_v1\runs_server\paper_track_p_12d_5ms_v1_server_depth34_width1024"
        Speed = 5.0
        Evidence = "12-D artifact exists; no stored closed-loop acceptance report"
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}

$listener = Get-NetTCPConnection -State Listen -LocalPort 41451 -ErrorAction SilentlyContinue
if (-not $listener) {
    throw "AirSim RPC port 41451 is not listening. Start UE4Editor, open Blocks, and click Play first."
}

$names = if ($Candidate -eq "paper5ms_all") {
    @($catalog.Keys | Where-Object { $_ -like "paper5ms_*" })
} else {
    @($Candidate)
}

$reportRoot = Join-Path $projectRoot "dnn_runs\historical_12d_retests"
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$env:PYTHONPATH = "F:\Airsim\AirSim\PythonClient;F:\Airsim\AirSim\PythonClient\.venv\Lib\site-packages"
$summary = @()

foreach ($name in $names) {
    $entry = $catalog[$name]
    $runRoot = Join-Path $projectRoot $entry.RelativePath
    $model = Join-Path $runRoot "controller_scripted.pt"
    $normalization = Join-Path $runRoot "normalization.json"
    $specPath = Join-Path $runRoot "model_spec.json"
    foreach ($required in @($model, $normalization, $specPath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required artifact is missing: $required"
        }
    }

    $spec = Get-Content -Raw -LiteralPath $specPath | ConvertFrom-Json
    $inputDimensions = if ($null -ne $spec.input_dimensions) {
        [int]$spec.input_dimensions
    } elseif ($null -ne $spec.input_shape) {
        [int]$spec.input_shape[-1]
    } else {
        throw "Cannot determine input dimensions from $specPath"
    }
    if ($inputDimensions -ne 12) {
        throw "Candidate $name is not a 12-D model (found $inputDimensions); refusing to run it."
    }

    $report = Join-Path $reportRoot ("{0}_epsilon{1}_seed{2}.json" -f $name, $Epsilon, $Seed)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Testing historical 12-D candidate: $name"
    Write-Host "Evidence: $($entry.Evidence)"
    Write-Host "Fixed training speed: $($entry.Speed) m/s"
    Write-Host "============================================================"

    $runnerArguments = @(
        ".\run_single_drone_dnn_gates.py",
        "--model", $model,
        "--normalization", $normalization,
        "--input-target", "gate",
        "--gate-attitude-encoding", "rotation_vector",
        "--controller-backend", "cpu_dnn",
        "--layout-root", ".\teacher_dataset_full_route_continuous_v2",
        "--initial-state-id", "0",
        "--vehicle-name", "Drone1",
        "--device", "cpu",
        "--epsilon", ([string]$Epsilon),
        "--seed", ([string]$Seed),
        "--speed", ([string]$entry.Speed),
        "--per-gate-timeout", "30",
        "--gate-count", ([string]$GateCount),
        "--x-start", "10",
        "--x-spacing", "10",
        "--base-z", "-2",
        "--gate-width", "4",
        "--gate-height", "4",
        "--gate-thickness", "0.15",
        "--gate-clearance", "0.25",
        "--gate-asset", "EditorCube",
        "--keep-gates",
        "--report", $report
    )

    & $Python @runnerArguments
    $exitCode = $LASTEXITCODE
    $summary += [pscustomobject]@{
        candidate = $name
        speed_mps = $entry.Speed
        exit_code = $exitCode
        report = $report
    }
    if ($exitCode -ne 0) {
        Write-Warning "Candidate $name failed with exit code $exitCode."
    }
    if (-not $NoPause -and $names.Count -gt 1 -and $name -ne $names[-1]) {
        Read-Host "Press Enter to reset AirSim and test the next historical candidate"
    }
}

Write-Host ""
Write-Host "Historical 12-D retest summary"
$summary | Format-Table -AutoSize
if (@($summary | Where-Object { $_.exit_code -ne 0 }).Count -gt 0) {
    exit 1
}
