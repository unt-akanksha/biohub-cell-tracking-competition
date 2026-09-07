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
$v21Runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-xl-balanced-v21.sh"
$v27Runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-hard-mined-temporal-snr-v27.sh"
$v21PreviousSha256 = "d5e0b93e082a9f42f271fa752b8121b90caee546c6ebfc0bc1414355df02d0ae"
$v27PreviousSha256 = "e3a881ca4be87bbffb0f2772ec4aa844fd46a624c921845359310402070c4490"
$v21RunnerSha256 = "8d2e1c215597e13b03c5457b6b19ef0bacf83bd3348eba47bb36e6ab9600a300"
$v27RunnerSha256 = "3761235e14f594dfd5a747f9b77f7ca073c3349d2e42186d2ea1eab007a4ccea"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-cost-aware-detector-queue-v1"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-cost-aware-detector-queue-v1"
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

function Invoke-RemoteScript([string]$Script) {
    $output = $Script | & ssh.exe @sshArgs $remote "bash -s" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Remote Bash failed with exit code $LASTEXITCODE`: $output"
    }
    return ($output -join "`n")
}

foreach ($entry in @(
    @{ path = $v21Runner; shell_path = "scripts/run-antelume-peak-rank-expanded-real-xl-balanced-v21.sh"; sha256 = $v21RunnerSha256 },
    @{ path = $v27Runner; shell_path = "scripts/run-antelume-peak-rank-hard-mined-temporal-snr-v27.sh"; sha256 = $v27RunnerSha256 }
)) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.path).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) { throw "Cost-aware runner hash changed: $($entry.path)" }
    & bash -n $entry.shell_path
    if ($LASTEXITCODE -ne 0) { throw "Cost-aware runner does not parse: $($entry.path)" }
}
if ($ValidateOnly) {
    @{
        status = "validated"
        automatic_chain = @("capacity-pu-v3", "expanded-real-xl-balanced-v21", "hard-mined-temporal-snr-v27")
        maximum_gpu_hours_removed = 45
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    $priorTerminal = Get-Content -Raw -LiteralPath $terminalPath | ConvertFrom-Json
    if ($priorTerminal.status -ne "failed") {
        throw "Cost-aware queue repair already reached a terminal state"
    }
    $previousFailure = Join-Path $stateRoot "deployment-terminal.previous-failed.json"
    if (Test-Path -LiteralPath $previousFailure) {
        throw "A prior cost-aware retry failure is already preserved"
    }
    Move-Item -LiteralPath $terminalPath -Destination $previousFailure
}

$sshArgs = @(
    "-i", $SshKey,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=30"
)
$remote = "${RemoteUser}@${RemoteHost}"
$v21Remote = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh"
$v27Remote = "/home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh"
$v21Temporary = "$v21Remote.cost-aware.partial"
$v27Temporary = "$v27Remote.cost-aware.partial"

try {
    $preflightTemplate = @'
set -euo pipefail
exact_waiter() {
  local pattern="$1"
  mapfile -t rows < <(pgrep -f "$pattern" || true)
  test "${#rows[@]}" -eq 1
  printf '%s\n' "${rows[0]}"
}
v4=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/faint-pu-v4/input/run-antelume-peak-rank-faint-pu-v4\.sh$')
v9=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9/run\.sh$')
v11=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-blob-v11/run\.sh$')
v13=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-global-v13/run\.sh$')
v15=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-multiscale-v15/run\.sh$')
v17=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-safe-rank-v17/run\.sh$')
v19=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\.sh$')
v21=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run\.sh$')
v23=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/temporal-min-local-snr-balanced-v23/run\.sh$')
v27=$(exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run\.sh$')
graph=$(exact_waiter '^bash /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run\.sh$')
for root in \
  /home/ubuntu/biohub-peak-rank-detector-v1/faint-pu-v4 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-blob-v11 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-global-v13 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-multiscale-v15 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-safe-rank-v17 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19 \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21 \
  /home/ubuntu/biohub-peak-rank-detector-v1/temporal-min-local-snr-balanced-v23 \
  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27; do
  test ! -e "$root/results"
  test ! -e "$root/training.log"
  test ! -e "$root/run.complete"
done
test ! -e /home/ubuntu/biohub-results/competition-graph-context-division-frozen-ensemble-v2
test ! -e /home/ubuntu/biohub-graph-context-frozen-ensemble-v2-results.tar.gz
echo '__V21_OLD__  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh' | sha256sum -c -
echo '__V27_OLD__  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh' | sha256sum -c -
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | grep -E '^[0-9]+$')
test "${#gpu_pids[@]}" -eq 1
gpu_pid="${gpu_pids[0]}"
tr '\0' ' ' < "/proc/$gpu_pid/cmdline" | grep -q 'capacity-pu-v3'
printf 'gpu_pid=%s v4=%s v9=%s v11=%s v13=%s v15=%s v17=%s v19=%s v21=%s v23=%s v27=%s graph=%s\n' \
  "$gpu_pid" "$v4" "$v9" "$v11" "$v13" "$v15" "$v17" "$v19" "$v21" "$v23" "$v27" "$graph"
'@
    $preflight = $preflightTemplate.Replace("__V21_OLD__", $v21PreviousSha256).Replace("__V27_OLD__", $v27PreviousSha256)
    $preflightOutput = Invoke-RemoteScript $preflight
    Invoke-External "scp.exe" ($sshArgs + @($v21Runner, "${remote}:$v21Temporary")) | Out-Null
    Invoke-External "scp.exe" ($sshArgs + @($v27Runner, "${remote}:$v27Temporary")) | Out-Null

    $rewireTemplate = @'
set -euo pipefail
stop_exact_waiter() {
  local pattern="$1"
  mapfile -t rows < <(pgrep -f "$pattern" || true)
  test "${#rows[@]}" -eq 1
  local pid="${rows[0]}"
  mapfile -t children < <(pgrep -P "$pid" || true)
  kill -TERM "$pid"
  if test "${#children[@]}" -gt 0; then kill -TERM "${children[@]}" 2>/dev/null || true; fi
  for _attempt in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  ! kill -0 "$pid" 2>/dev/null
  printf '%s\n' "$pid"
}
echo '__V21_NEW__  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh.cost-aware.partial' | sha256sum -c -
echo '__V27_NEW__  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh.cost-aware.partial' | sha256sum -c -
mapfile -t gpu_before < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | grep -E '^[0-9]+$')
test "${#gpu_before[@]}" -eq 1
tr '\0' ' ' < "/proc/${gpu_before[0]}/cmdline" | grep -q 'capacity-pu-v3'
stopped_v4=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/faint-pu-v4/input/run-antelume-peak-rank-faint-pu-v4\.sh$')
stopped_v9=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9/run\.sh$')
stopped_v11=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-blob-v11/run\.sh$')
stopped_v13=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-global-v13/run\.sh$')
stopped_v15=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-multiscale-v15/run\.sh$')
stopped_v17=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-safe-rank-v17/run\.sh$')
stopped_v19=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19/run\.sh$')
stopped_v21=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run\.sh$')
stopped_v23=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/temporal-min-local-snr-balanced-v23/run\.sh$')
stopped_v27=$(stop_exact_waiter '^bash /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run\.sh$')
stopped_graph=$(stop_exact_waiter '^bash /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run\.sh$')
mv -- /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh.cost-aware.partial \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh
mv -- /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh.cost-aware.partial \
  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh
chmod +x \
  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh \
  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh
bash -n /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh
bash -n /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh
nohup bash /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh \
  >/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/controller.cost-aware.log 2>&1 < /dev/null &
new_v21=$!
nohup bash /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh \
  >/home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/controller.cost-aware.log 2>&1 < /dev/null &
new_v27=$!
sleep 3
kill -0 "$new_v21"
kill -0 "$new_v27"
mapfile -t gpu_after < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | grep -E '^[0-9]+$')
test "${#gpu_after[@]}" -eq 1
test "${gpu_after[0]}" = "${gpu_before[0]}"
tr '\0' ' ' < "/proc/${gpu_after[0]}/cmdline" | grep -q 'capacity-pu-v3'
echo '__V21_NEW__  /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-xl-balanced-v21/run.sh' | sha256sum -c -
echo '__V27_NEW__  /home/ubuntu/biohub-peak-rank-detector-v1/hard-mined-temporal-snr-v27/run.sh' | sha256sum -c -
printf 'gpu_pid=%s new_v21=%s new_v27=%s stopped_v4=%s stopped_v9=%s stopped_v11=%s stopped_v13=%s stopped_v15=%s stopped_v17=%s stopped_v19=%s stopped_v21=%s stopped_v23=%s stopped_v27=%s stopped_graph=%s\n' \
  "${gpu_after[0]}" "$new_v21" "$new_v27" "$stopped_v4" "$stopped_v9" "$stopped_v11" "$stopped_v13" "$stopped_v15" "$stopped_v17" "$stopped_v19" "$stopped_v21" "$stopped_v23" "$stopped_v27" "$stopped_graph"
'@
    $rewire = $rewireTemplate.Replace("__V21_NEW__", $v21RunnerSha256).Replace("__V27_NEW__", $v27RunnerSha256)
    $rewireOutput = Invoke-RemoteScript $rewire
    $gpuMatch = [regex]::Match($rewireOutput, "gpu_pid=([0-9]+)")
    $v21Match = [regex]::Match($rewireOutput, "new_v21=([0-9]+)")
    $v27Match = [regex]::Match($rewireOutput, "new_v27=([0-9]+)")
    if (-not ($gpuMatch.Success -and $v21Match.Success -and $v27Match.Success)) {
        throw "Cost-aware queue returned unexpected evidence: $rewireOutput"
    }
    Write-Terminal @{
        status = "rewired_cost_aware_waiting_for_v3"
        remote_host = $RemoteHost
        active_v3_gpu_pid = [int64]$gpuMatch.Groups[1].Value
        replacement_v21_waiter_pid = [int64]$v21Match.Groups[1].Value
        replacement_v27_waiter_pid = [int64]$v27Match.Groups[1].Value
        preflight = $preflightOutput
        mutation_evidence = $rewireOutput
        automatic_chain = @("capacity-pu-v3", "expanded-real-xl-balanced-v21", "hard-mined-temporal-snr-v27")
        skipped_automatic_runs = @("faint-pu-v4", "expanded-real-local-shape-v9", "expanded-real-blob-v11", "expanded-real-global-v13", "expanded-real-multiscale-v15", "expanded-real-safe-rank-v17", "expanded-real-balanced-v19", "temporal-min-local-snr-v23", "graph-context-frozen-ensemble-v2")
        maximum_gpu_hours_removed = 45
        active_v3_interrupted = $false
        rsna_or_unrelated_process_interrupted = $false
    }
    Get-Content -Raw -LiteralPath $terminalPath
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
