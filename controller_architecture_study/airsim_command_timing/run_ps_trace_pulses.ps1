param(
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$BoardKit,
    [Parameter(Mandatory = $true)][int]$DiagnosticPid,
    [ValidateSet('natural_retime20', 'ping', 'receipt', 'event_log', 'event_schedule', 'pin_release')][string]$Arm = 'natural_retime20'
)

$ErrorActionPreference = 'Stop'
if ($BoardKit -notmatch '^(/home/root|/run)/tcs_command_trace_20261124_[a-zA-Z0-9]+$') { throw 'Expected the dedicated diagnostic kit' }
$diag = Get-Process -Id $DiagnosticPid
if ($diag.ProcessName -ne 'UE4Editor-Cmd' -or -not (netstat -ano -p tcp |
        Select-String "192\.168\.77\.1:41591\s+.*LISTENING\s+$DiagnosticPid\s*$")) { throw 'Unexpected diagnostic port owner' }
$saved = $diag.ProcessorAffinity
$mask = 15L
if (($saved.ToInt64() -band $mask) -ne $mask -or $saved.ToInt64() -eq $mask) { throw 'Expected four logical CPUs and a wider original allocation' }
$label = if ($Arm -eq 'natural_retime20') { 'run_pulse4' } else { "run_$Arm" }
$seed = (Get-Random -Minimum 500000 -Maximum 2000000000) * 32768L
$line = 'ssh -o BatchMode=yes axu3egb "sh {0}/run_ps_trace.sh {0} {0}/{1} config_{2}.json {3}"' -f $BoardKit, $label, $Arm, $seed
$marker = 'ssh -o BatchMode=yes axu3egb "test -s {0}/{1}/trial/execution_identity.json"' -f $BoardKit, $label
$job = Start-Job -ArgumentList $line -ScriptBlock {
    param($remote)
    ssh -o BatchMode=yes vitis-vm $remote
    if ($LASTEXITCODE -ne 0) { throw 'Board trial failed' }
}
$switches = [System.Collections.Generic.List[object]]::new()
try {
    $ready = $false
    foreach ($i in 1..105) {
        if ($job.State -notin 'Running', 'NotStarted') { break }
        ssh -o BatchMode=yes vitis-vm $marker
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        Start-Sleep -Seconds 1
    }
    if (!$ready) { Receive-Job $job | Out-Host; throw 'Board did not start its measured window' }
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    foreach ($target in 3000, 9000, 15000) {
        Start-Sleep -Milliseconds ([Math]::Max(0, $target - $timer.ElapsedMilliseconds))
        if ($job.State -ne 'Running') { break }
        $diag.ProcessorAffinity = [IntPtr]$mask
        $start = $timer.Elapsed.TotalMilliseconds
        Start-Sleep -Milliseconds 150
        $diag.ProcessorAffinity = $saved
        $switches.Add(@{ start_ms_after_marker = $start; end_ms_after_marker = $timer.Elapsed.TotalMilliseconds })
    }
    Receive-Job $job -Wait | Tee-Object -FilePath (Join-Path $Output "$label.log")
    if ($job.State -ne 'Completed') { throw 'Board trial did not complete' }
} finally {
    $diag.ProcessorAffinity = $saved
    @{ name = $label; seed = $seed; affinity = $mask; original = $saved.ToInt64(); switches = $switches } |
        ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii -Path (Join-Path $Output "$label.json")
}
