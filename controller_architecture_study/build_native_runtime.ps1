param([string]$AirSimRoot = 'F:\Airsim\AirSim')
$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'native_runtime'
$build = Join-Path $source 'build'
& cmake -S $source -B $build -G 'Visual Studio 17 2022' -A x64 "-DAIRSIM_ROOT=$AirSimRoot"
if ($LASTEXITCODE -ne 0) { throw 'Native runtime CMake configuration failed' }
& cmake --build $build --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Native runtime compilation failed' }
Write-Host "Native runtime: $build\Release\falcon_native_io.dll"
