param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [int]$BatchSize = 512
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$config = Join-Path $PSScriptRoot 'configs\incremental_supernet_237d_v3_recovery600.json'
$run = Join-Path $PSScriptRoot 'results\incremental_supernet_237d_v3_recovery600'
& $Python -m controller_architecture_study.incremental_supernet.train --config $config --output-dir $run --device $Device --batch-size $BatchSize --resume
if ($LASTEXITCODE -ne 0) { throw 'Training interrupted or failed. Checkpoints retained; rerun this command to resume.' }
& $Python -m controller_architecture_study.incremental_supernet.export --checkpoint (Join-Path $run 'best_model.pt') --output-dir (Join-Path $run 'export') --device $Device --golden-count 32
if ($LASTEXITCODE -ne 0) { throw 'Training checkpoints retained. Inspect the export error before retrying.' }
Write-Host "600 additional epochs completed; source run unchanged. Artifacts: $run\export"
