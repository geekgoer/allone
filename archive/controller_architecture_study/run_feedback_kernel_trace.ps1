param(
    [string]$Python = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe',
    [switch]$ResidentWorker
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Windows拒绝了内核性能采样。请从“以管理员身份运行”的PowerShell执行本脚本。'
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python路径不存在' }
Set-Location -LiteralPath $project
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$output = Join-Path $project "controller_architecture_study\results\feedback_kernel_trace_$stamp"
$trace = "$output.etl"
$status = (& wpr.exe -status 2>&1 | Out-String)
if ($status -notmatch 'WPR is not recording') {
    throw '已有WPR记录或无法确认记录器空闲；本脚本不会停止其他记录。'
}
$started = $false
$trialExit = 1
try {
    & wpr.exe -start GeneralProfile -filemode
    if ($LASTEXITCODE -ne 0) { throw '无法启动Windows内核采样，请保留WPR错误。' }
    $started = $true
    $probeArgs = @('-u', '-m', 'controller_architecture_study.probe_feedback_handoff', '--normal-only', '--output', $output)
    if ($ResidentWorker) { $probeArgs += '--resident-worker' }
    & $Python @probeArgs
    $trialExit = $LASTEXITCODE
} finally {
    if ($started) {
        & wpr.exe -stop $trace
        if ($LASTEXITCODE -ne 0) { Write-Warning 'WPR停止或保存失败，请保留此窗口输出。' }
    }
}
Write-Host "诊断目录：$output"
Write-Host "内核跟踪：$trace"
if ($trialExit -ne 0) { throw '诊断试飞异常，记录已保留。' }
