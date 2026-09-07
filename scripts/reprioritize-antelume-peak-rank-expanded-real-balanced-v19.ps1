param(
    [string]$RemoteHost = "100.54.137.149",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [switch]$RecoverExisting,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
$expectedRunnerSha256 = "57ef88c6756848e83cf8932f2bdee7065633410407acda9bedd16daa0969d248"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-balanced-v19-priority"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19"
$remoteRunner = "$remoteRoot/run.sh"
$remoteTemporary = "$remoteRoot/run.sh.priority-v2.partial"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-balanced-v19-priority"
    $Payload["runner_sha256"] = $expectedRunnerSha256
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Invoke-External([string]$Program, [string[]]$Arguments) {
    $output = & $Program @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE`: $output"
    }
    return ($output -join "`n")
}

$actualRunnerSha256 = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualRunnerSha256 -ne $expectedRunnerSha256) {
    throw "Priority v19 runner hash changed: $actualRunnerSha256"
}
& bash -n "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
if ($LASTEXITCODE -ne 0) { throw "Priority v19 runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_balanced_v19_priority" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Priority v19 deployment already reached a terminal state"
}

$sshArgs = @(
    "-i", $SshKey,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=30"
)
$remote = "${RemoteUser}@${RemoteHost}"

try {
    if ($RecoverExisting) {
        $recoverTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
mapfile -t pids < <(pgrep -f '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\.sh$')
test "${#pids[@]}" -eq 1
test ! -e "$run_root/training.log"
test ! -e "$run_root/results"
test ! -e /home/ubuntu/biohub-peak-rank-expanded-real-balanced-v19-results.tar.gz
echo '__RUNNER_SHA__  '"$run_root/run.sh" | sha256sum -c - >/dev/null
printf '%s\n' "${pids[0]}"
'@
        $recover = $recoverTemplate.Replace("__RUNNER_SHA__", $expectedRunnerSha256)
        $newPid = (Invoke-External "ssh.exe" ($sshArgs + @($remote, $recover))).Trim()
        if ($newPid -notmatch "^[0-9]+$") { throw "Unexpected recovered v19 PID: $newPid" }
        Write-Terminal @{
            status = "reprioritized_waiting_for_verified_v2"
            remote_host = $RemoteHost
            remote_scope = $remoteRoot
            replacement_waiter_pid = [int64]$newPid
            receipt_recovered_after_successful_remote_launch = $true
            gpu_policy = "start_after_verified_v2_and_yield_to_all_active_gpu_clients"
            untouched_workloads = @("active_depth_pu_v2", "nucverse3d_waiter", "rsna_and_all_unrelated_processes")
        }
        Get-Content -Raw -LiteralPath $terminalPath
        exit 0
    }

    $inspect = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
mapfile -t pids < <(pgrep -f '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\.sh$')
test "${#pids[@]}" -eq 1
old_pid="${pids[0]}"
test ! -e "$run_root/training.log"
test ! -e "$run_root/results"
test ! -e /home/ubuntu/biohub-peak-rank-expanded-real-balanced-v19-results.tar.gz
printf '%s\n' "$old_pid"
'@
    $oldPid = (Invoke-External "ssh.exe" ($sshArgs + @($remote, $inspect))).Trim()
    if ($oldPid -notmatch "^[0-9]+$") { throw "Unexpected old v19 PID: $oldPid" }

    $stop = "set -euo pipefail; kill -0 '$oldPid'; test ! -e '$remoteRoot/training.log'; test ! -e '$remoteRoot/results'; kill -TERM '$oldPid'; for poll in `$(seq 1 10); do if ! kill -0 '$oldPid' 2>/dev/null; then exit 0; fi; sleep 1; done; exit 11"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $stop)) | Out-Null
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteTemporary")) | Out-Null

    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
temporary="$run_root/run.sh.priority-v2.partial"
runner="$run_root/run.sh"
echo '__RUNNER_SHA__  '"$temporary" | sha256sum -c -
mv -- "$temporary" "$runner"
chmod +x "$runner"
bash -n "$runner"
for poll in $(seq 1 45); do
  if flock -n "$run_root/run.lock" -c true; then break; fi
  sleep 1
done
flock -n "$run_root/run.lock" -c true
nohup bash "$runner" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf '%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__RUNNER_SHA__", $expectedRunnerSha256)
    $launchOutput = Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    $newPid = (($launchOutput -split "`r?`n") | Where-Object { $_ -match "^[0-9]+$" } | Select-Object -Last 1).Trim()
    if ($newPid -notmatch "^[0-9]+$") { throw "Unexpected new v19 PID: $newPid" }
    Write-Terminal @{
        status = "reprioritized_waiting_for_verified_v2"
        remote_host = $RemoteHost
        remote_scope = $remoteRoot
        previous_waiter_pid = [int64]$oldPid
        replacement_waiter_pid = [int64]$newPid
        gpu_policy = "start_after_verified_v2_and_yield_to_all_active_gpu_clients"
        untouched_workloads = @("active_depth_pu_v2", "nucverse3d_waiter", "rsna_and_all_unrelated_processes")
    }
    Get-Content -Raw -LiteralPath $terminalPath
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
