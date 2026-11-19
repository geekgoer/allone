param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$BatchSize = 512
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$config = Join-Path $PSScriptRoot 'configs\incremental_supernet_237d_m2_w8a16_v1.json'
& $Python -m controller_architecture_study.incremental_supernet.quantize preflight --config $config
if ($LASTEXITCODE -ne 0) { throw 'Source or dataset contract check failed; no training started.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize train --config $config --device $Device --batch-size $BatchSize --resume
if ($LASTEXITCODE -ne 0) { throw 'QAT interrupted; completed epochs retained. Rerun this command to resume.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize export --config $config --device $Device --golden-count 32 --integer-batch-size 32
if ($LASTEXITCODE -eq 2) {
    throw 'Integer arithmetic passed, but a quality goal was not met. Artifacts retained; inspect export_report.json before FPGA work.'
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Integer export failed. QAT checkpoints retained; inspect the numerical error.'
}
Write-Host 'M2 complete: one W8A16 package, 136 configurations, true integer validation and switch-consistency report. FPGA/flight not yet verified.'
