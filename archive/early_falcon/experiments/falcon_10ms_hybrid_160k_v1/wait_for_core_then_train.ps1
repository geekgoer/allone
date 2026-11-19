[CmdletBinding()]
param(
    [string]$ProjectRoot = 'F:\FalconReproduction\airsim_control',
    [string]$CollectionPython = 'F:\Airsim\AirSim\PythonClient\.venv\Scripts\python.exe',
    [string]$TrainingPython = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [int]$PollSeconds = 30,
    [string]$Device = 'cuda',
    [bool]$StopAirSimAfterCore = $true
)

$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 5) { throw 'PollSeconds must be at least 5.' }
Set-Location -LiteralPath $ProjectRoot

$experiment = Join-Path $ProjectRoot 'experiments\falcon_10ms_hybrid_160k_v1'
$coreRoot = Join-Path $experiment 'core_40k_airsim_acado_10ms_v2_safe_start'
$coreManifest = Join-Path $coreRoot 'manifest.json'
$coreFinal = Join-Path $coreRoot 'final\falcon_hybrid_core_40k_10ms_v2_safe_start.npz'
$logs = Join-Path $experiment 'pipeline_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$transcript = Join-Path $logs "wait_then_train_$stamp.log"

function Invoke-PythonOrThrow {
    param([string]$Python, [string]$Script, [string[]]$Arguments)
    & $Python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $Script"
    }
}

function Stop-BlocksAirSimIfRequested {
    if (-not $StopAirSimAfterCore) {
        Write-Host 'Keeping AirSim open because StopAirSimAfterCore is false.'
        return
    }
    # Restrict the shutdown to the Blocks project.  Do not terminate every
    # UE4Editor.exe process, since another Unreal project may be open.
    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('UE4Editor.exe', 'UE4Editor-Cmd.exe') -and
        $_.CommandLine -like '*F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject*'
    }
    if ($null -eq $processes -or @($processes).Count -eq 0) {
        Write-Host 'No Blocks UE4Editor process was found; no AirSim process needed to be stopped.'
        return
    }
    foreach ($process in @($processes)) {
        Write-Host "Stopping completed-core AirSim process: pid=$($process.ProcessId), name=$($process.Name)"
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
    }
    Start-Sleep -Seconds 5
}

try {
    Start-Transcript -LiteralPath $transcript -Append | Out-Null
} catch {
    Write-Warning "Could not start transcript: $($_.Exception.Message)"
}

try {
    Write-Host "Waiting for the separately running AirSim core collection..."
    Write-Host "Core manifest: $coreManifest"
    while ($true) {
        if ((Test-Path -LiteralPath $coreManifest) -and (Test-Path -LiteralPath $coreFinal)) {
            try {
                $manifest = Get-Content -Raw -LiteralPath $coreManifest | ConvertFrom-Json
                if ($manifest.status -eq 'completed') {
                    Write-Host 'Detected completed 40k AirSim core.'
                    break
                }
            } catch {
                Write-Host 'Core manifest is being updated; continuing to wait.'
            }
        }
        if (Test-Path -LiteralPath $coreManifest) {
            try {
                $status = (Get-Content -Raw -LiteralPath $coreManifest | ConvertFrom-Json).status
                if ($status -in @('incomplete', 'interrupted')) {
                    throw "Core collection ended with status '$status'. Resume/fix it, then restart this watcher."
                }
            } catch {
                if ($_.Exception.Message -like 'Core collection ended*') { throw }
            }
        }
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] core not complete; waiting $PollSeconds seconds."
        Start-Sleep -Seconds $PollSeconds
    }

    Invoke-PythonOrThrow $CollectionPython '.\validate_teacher_dataset.py' @(
        '--dataset-root', '.\experiments\falcon_10ms_hybrid_160k_v1\core_40k_airsim_acado_10ms_v2_safe_start', '--json'
    )
    Stop-BlocksAirSimIfRequested
    Invoke-PythonOrThrow $CollectionPython '.\experiments\falcon_10ms_hybrid_160k_v1\generate_extensions.py' @('--resume')
    Invoke-PythonOrThrow $CollectionPython '.\experiments\falcon_10ms_hybrid_160k_v1\prepare_hybrid_dataset.py' @()
    # Omit --json here: the validator then turns missing launch/near-phase
    # coverage into a failing pipeline condition instead of a mere report.
    Invoke-PythonOrThrow $CollectionPython '.\experiments\falcon_10ms_hybrid_160k_v1\validate_hybrid_dataset.py' @()
    Invoke-PythonOrThrow $TrainingPython '.\experiments\falcon_10ms_hybrid_160k_v1\train_six_models_8000_safe.py' @('--device', $Device)
    Write-Host 'Pipeline complete: core, extensions, preparation, audit, and all six models finished.'
} finally {
    try { Stop-Transcript | Out-Null } catch {}
}
