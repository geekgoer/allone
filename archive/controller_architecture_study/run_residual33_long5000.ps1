param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }

$configPath = Join-Path $PSScriptRoot 'configs\residual33x1024_long5000_v1.json'
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$output = Join-Path $PSScriptRoot 'results\decision_convex_residual33_long5000_v1\residual_mlp_h33_w1024_seed0'
$checks = @(@{ Path = $config.init_checkpoint; Hash = $config.source_checkpoint_sha256 })
foreach ($item in $config.dataset_file_sha256.PSObject.Properties) {
    $checks += @{ Path = Join-Path $config.data_dir $item.Name; Hash = $item.Value }
}
foreach ($item in $checks) {
    if (-not (Test-Path -LiteralPath $item.Path)) { throw "Required file missing: $($item.Path)" }
    if ((Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Hash) {
        throw "Frozen source/data changed: $($item.Path)"
    }
}

$arguments = @('-m', 'controller_architecture_study.train_controller', '--config', $configPath,
    '--output-dir', $output, '--device', 'cuda')
if (Test-Path -LiteralPath (Join-Path $output 'config.json')) { $arguments += '--resume' }
Write-Host 'One model only: residual 33x1024, input=237, FP32, +5000 epochs, NO early stopping.'
Write-Host 'Adam moments restored; cosine LR 1e-5 -> 1e-6; batch=512. Source/data remain read-only.'
Write-Host 'Latest recovery snapshot every 10 epochs (and on improvements); rerun to resume.'
Write-Host "Output: $output"
if ($PlanOnly) {
    Write-Host "Validated plan: $Python $($arguments -join ' ')"
    exit 0
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Training interrupted/failed. Saved checkpoints remain; rerun this same command to resume.' }
Write-Host '5000-epoch run completed. best_model.pt is validation-selected; last_model.pt is the final training state.'
