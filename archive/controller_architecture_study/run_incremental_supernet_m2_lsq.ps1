param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$BatchSize = 512
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$config = Join-Path $PSScriptRoot 'configs\incremental_supernet_237d_m2_lsq_v3.json'
& $Python -m controller_architecture_study.incremental_supernet.quantize preflight --config $config
if ($LASTEXITCODE -ne 0) { throw 'Frozen FP32, v2 initialization, or dataset check failed.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize train --config $config --device $Device --batch-size $BatchSize --resume
if ($LASTEXITCODE -ne 0) { throw 'QAT interrupted; checkpoints retained. Rerun to resume the last complete epoch.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize export --config $config --device $Device --golden-count 32 --integer-batch-size 32
if ($LASTEXITCODE -eq 2) {
    throw 'Integer numerics passed, but original quality goals failed. Inspect export_report.json; artifacts retained.'
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Integer export failed; checkpoints retained. Inspect the numerical error.'
}
Write-Host 'M2 v3 complete: shared learned dyadic scales frozen for deployment; all 136 configurations retained.'
