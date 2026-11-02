param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$BatchSize = 512
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$config = Join-Path $PSScriptRoot 'configs\incremental_supernet_237d_m2_mixed_v2.json'
& $Python -m controller_architecture_study.incremental_supernet.quantize preflight --config $config
if ($LASTEXITCODE -ne 0) { throw 'Source or dataset contract check failed.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize train --config $config --device $Device --batch-size $BatchSize --resume
if ($LASTEXITCODE -ne 0) { throw 'QAT interrupted; rerun this command to resume the last completed epoch.' }
& $Python -m controller_architecture_study.incremental_supernet.quantize export --config $config --device $Device --golden-count 32 --integer-batch-size 32
if ($LASTEXITCODE -eq 2) {
    throw 'Integer numerics passed, but quality goals failed. Artifacts retained; inspect export_report.json.'
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Integer export failed; checkpoints retained. Inspect the numerical error.'
}
Write-Host 'M2 mixed precision complete: backbone W8, heads W16, A16; all 136 configurations retained.'
