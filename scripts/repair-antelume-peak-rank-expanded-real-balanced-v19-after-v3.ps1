param(
    [string]$RemoteHost = "100.54.137.149",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
$runnerSha256 = "90e7d2efc982274cb53d37635ed258e2ac39fda773bd0c35001dda73b6b010a6"
$previousRunnerSha256 = "57ef88c6756848e83cf8932f2bdee7065633410407acda9bedd16daa0969d248"
$priorityReceipt = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-balanced-v19-priority/deployment-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-balanced-v19-after-v3"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19"
$remoteRunner = "$remoteRoot/run.sh"
$remoteTemporary = "$remoteRoot/run.sh.after-v3.partial"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-balanced-v19-after-v3"
    $Payload["runner_sha256"] = $runnerSha256
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

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $runner).Hash.ToLowerInvariant() -ne $runnerSha256) {
    throw "V19 after-V3 runner hash changed"
}
& bash -n "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
if ($LASTEXITCODE -ne 0) { throw "V19 after-V3 runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_balanced_v19_after_v3" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) { throw "V19 after-V3 repair already reached a terminal state" }
if (-not (Test-Path -LiteralPath $priorityReceipt -PathType Leaf)) { throw "Prior V19 priority receipt is missing" }
$prior = Get-Content -Raw -LiteralPath $priorityReceipt | ConvertFrom-Json
if (-not (
    $prior.status -eq "reprioritized_waiting_for_verified_v2" -and
    $prior.runner_sha256 -eq $previousRunnerSha256 -and
    $prior.competition_submission_performed -eq $false
)) { throw "Prior V19 priority receipt changed" }

$sshArgs = @(
    "-i", $SshKey,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=30"
)
$remote = "${RemoteUser}@${RemoteHost}"
try {
    $inspectTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
mapfile -t pids < <(pgrep -f '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\.sh$' || true)
test "${#pids[@]}" -eq 0
test ! -e "$run_root/training.log"
test ! -e "$run_root/results"
test ! -e /home/ubuntu/biohub-peak-rank-expanded-real-balanced-v19-results.tar.gz
echo '__PREVIOUS_SHA__  '"$run_root/run.sh" | sha256sum -c -
'@
    $inspect = $inspectTemplate.Replace("__PREVIOUS_SHA__", $previousRunnerSha256)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $inspect)) | Out-Null
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteTemporary")) | Out-Null
    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
temporary="$run_root/run.sh.after-v3.partial"
runner="$run_root/run.sh"
echo '__RUNNER_SHA__  '"$temporary" | sha256sum -c -
mv -- "$temporary" "$runner"
chmod +x "$runner"
bash -n "$runner"
nohup bash "$runner" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf '%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__RUNNER_SHA__", $runnerSha256)
    $launchOutput = Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    $newPid = (($launchOutput -split "`r?`n") | Where-Object { $_ -match "^[0-9]+$" } | Select-Object -Last 1).Trim()
    if ($newPid -notmatch "^[0-9]+$") { throw "Unexpected repaired V19 PID: $newPid" }
    Write-Terminal @{
        status = "repaired_waiting_for_verified_v3"
        remote_host = $RemoteHost
        remote_scope = $remoteRoot
        replacement_waiter_pid = [int64]$newPid
        previous_failed_waiter_pid = [int64]$prior.replacement_waiter_pid
        priority_receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $priorityReceipt).Hash.ToLowerInvariant()
        gpu_policy = "start_after_verified_v3_and_yield_to_all_active_gpu_clients"
        active_v3_interrupted = $false
        rsna_or_unrelated_process_interrupted = $false
    }
    Get-Content -Raw -LiteralPath $terminalPath
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
