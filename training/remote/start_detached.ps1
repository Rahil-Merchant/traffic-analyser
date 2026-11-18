<#
Start setup_box.ps1 as a process that does not belong to the ssh session, so a dropped connection does not
kill it (Win32_Process.Create runs it outside sshd's job object). Output goes to <Root>\logs\setup_<time>.log;
the last line of that log is "EXIT <code>".
    powershell -NoProfile -ExecutionPolicy Bypass -File start_detached.ps1 -Root D:\training -Steps envs,weights
#>
param([Parameter(Mandatory = $true)][string]$Root, [Parameter(Mandatory = $true)][string]$Steps)
$logs = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ("setup_{0:yyMMdd_HHmmss}.log" -f (Get-Date))
$setup = Join-Path $PSScriptRoot 'setup_box.ps1'
$inner = "& '$setup' -Root '$Root' -Steps '$Steps' *>> '$log'; Add-Content '$log' ('EXIT ' + [int](-not `$?))"
$enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $enc"; CurrentDirectory = $Root }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"started pid $($r.ProcessId), log $log"
