param(
    [string]$RemoteHost = "100.54.137.149",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$ExpandedKernelVersion = 3,
    [int]$PollSeconds = 120,
    [double]$MaximumWaitHours = 72.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($ExpandedKernelVersion -lt 1 -or $PollSeconds -lt 60 -or $MaximumWaitHours -le 0) {
    throw "Invalid expanded-real faint deployment bounds"
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-faint-v7-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$expandedState = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-expanded-replay-v2"
$expandedTerminalPath = Join-Path $expandedState "harvest-v$ExpandedKernelVersion-terminal.json"
$expandedDownloadRoot = Join-Path $expandedState "kernel-v$ExpandedKernelVersion"
$expandedArchive = Join-Path $expandedDownloadRoot "biohub-real-localization-expanded-shards-v2.tar"
$expandedSums = "$expandedArchive.sha256"
$v4State = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-faint-pu-v4"
$v4TerminalPath = Join-Path $v4State "harvest-terminal.json"
$v4Archive = Join-Path $v4State "peak-rank-faint-pu-v4-results.tar.gz"
$trainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_faint_detector.py"
$faintTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_faint_cell_pu_detector.py"
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-faint-v7.sh"
$guard = Join-Path $RepositoryRoot "scripts/run-antelume-biohub-yield-guard-v1.sh"
$expectedHashes = @{
    trainer = "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe"
    faint_trainer = "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c"
    runner = "7f8478c770b5d08f95b310f232714e8ba35a9ca2736d648e1c830ccdba48c5af"
    guard = "3ff07adde246f118581503616d43a7c5587c2033d55ff0d93dae7dee6e641728"
}
$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-faint-v7-deploy"
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Assert-Hash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label is missing" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) { throw "$Label hash changed: $actual" }
}

function Invoke-External([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
Assert-Hash $trainer $expectedHashes.trainer "Expanded-real trainer"
Assert-Hash $faintTrainer $expectedHashes.faint_trainer "Faint-cell trainer"
Assert-Hash $runner $expectedHashes.runner "Expanded-real runner"
Assert-Hash $guard $expectedHashes.guard "GPU yield guard"
& python -m py_compile $trainer
if ($LASTEXITCODE -ne 0) { throw "Expanded-real trainer does not compile" }
& bash -n "scripts/run-antelume-peak-rank-expanded-real-faint-v7.sh"
if ($LASTEXITCODE -ne 0) { throw "Expanded-real runner does not parse" }
& bash -n "scripts/run-antelume-biohub-yield-guard-v1.sh"
if ($LASTEXITCODE -ne 0) { throw "GPU yield guard does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_faint_v7_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Expanded-real faint deployment already reached a terminal state"
}

try {
    while (
        (-not (Test-Path -LiteralPath $expandedTerminalPath -PathType Leaf) -or
         -not (Test-Path -LiteralPath $v4TerminalPath -PathType Leaf)) -and
        [DateTimeOffset]::UtcNow -lt $deadline
    ) {
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $expandedTerminalPath -PathType Leaf)) {
        throw "Timed out waiting for expanded replay harvest"
    }
    if (-not (Test-Path -LiteralPath $v4TerminalPath -PathType Leaf)) {
        throw "Timed out waiting for v4 detector harvest"
    }
    $expanded = Get-Content -Raw -LiteralPath $expandedTerminalPath | ConvertFrom-Json
    $v4 = Get-Content -Raw -LiteralPath $v4TerminalPath | ConvertFrom-Json
    if ($expanded.status -ne "verified") { throw "Expanded replay harvest did not verify" }
    if ($v4.status -ne "harvest_verified") { throw "v4 detector harvest did not verify" }
    if (-not (Test-Path -LiteralPath $expandedArchive -PathType Leaf) -or
        -not (Test-Path -LiteralPath $expandedSums -PathType Leaf)) {
        throw "Expanded replay archive receipt is incomplete"
    }
    if (-not (Test-Path -LiteralPath $v4Archive -PathType Leaf)) {
        throw "v4 detector archive is missing"
    }
    $expandedSha = (Get-FileHash -LiteralPath $expandedArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $v4Sha = (Get-FileHash -LiteralPath $v4Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expandedSha -ne [string]$expanded.verification.archive_sha256) {
        throw "Expanded replay archive changed after verification"
    }
    if ($v4Sha -ne [string]$v4.archive_sha256) {
        throw "v4 detector archive changed after verification"
    }
    $expandedSumFields = ((Get-Content -Raw -LiteralPath $expandedSums).Trim() -split '\s+')
    if ($expandedSumFields.Count -ne 2 -or $expandedSumFields[0] -ne $expandedSha) {
        throw "Expanded replay SHA receipt changed"
    }

    $sshArgs = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30"
    )
    $remote = "${RemoteUser}@${RemoteHost}"
    $prepare = "set -euo pipefail; mkdir -p /home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/input /home/ubuntu/biohub-gpu-yield-guard-v1"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $prepare))
    Invoke-External "scp.exe" ($sshArgs + @($trainer, "${remote}:/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/input/train_expanded_real_faint_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($faintTrainer, "${remote}:/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/input/train_faint_cell_pu_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/run.sh"))
    Invoke-External "scp.exe" ($sshArgs + @($guard, "${remote}:/home/ubuntu/biohub-gpu-yield-guard-v1/run.sh.new"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedArchive, "${remote}:/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/input/biohub-real-localization-expanded-shards-v2.tar"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedSums, "${remote}:/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/input/biohub-real-localization-expanded-shards-v2.tar.sha256"))

    $v4Bytes = (Get-Item -LiteralPath $v4Archive).Length
    $remoteCommandTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7
guard_root=/home/ubuntu/biohub-gpu-yield-guard-v1
v4_archive=/home/ubuntu/biohub-peak-rank-faint-pu-v4-results.tar.gz
v4_ack=/home/ubuntu/biohub-peak-rank-detector-v1/faint-pu-v4/harvest.verified
test "$(sha256sum "$v4_archive" | awk '{print $1}')" = '__V4_SHA__'
printf '%s %s\n' '__V4_SHA__' '__V4_BYTES__' > "$v4_ack.partial"
mv "$v4_ack.partial" "$v4_ack"
chmod +x "$run_root/run.sh" "$guard_root/run.sh.new"
bash -n "$run_root/run.sh"
bash -n "$guard_root/run.sh.new"
for process_dir in /proc/[0-9]*; do
  pid=${process_dir##*/}
  test -r "$process_dir/cmdline" || continue
  command=$(tr '\0' ' ' < "$process_dir/cmdline" 2>/dev/null || true)
  case "$command" in
    "bash /home/ubuntu/biohub-gpu-yield-guard-v1/run.sh "|"bash /home/ubuntu/biohub/scripts/run-antelume-biohub-yield-guard-v1.sh ")
      kill -TERM "$pid"
      ;;
  esac
done
sleep 2
install -m 0755 "$guard_root/run.sh.new" "$guard_root/run.sh"
rm -f "$guard_root/run.sh.new"
nohup bash "$guard_root/run.sh" >"$guard_root/controller.log" 2>&1 < /dev/null &
guard_pid=$!
sleep 2
kill -0 "$guard_pid"
test ! -e /home/ubuntu/biohub-peak-rank-expanded-real-faint-v7-results.tar.gz
nohup bash "$run_root/run.sh" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
printf 'GUARD_PID=%s\nRUNNER_PID=%s\n' "$guard_pid" "$runner_pid"
'@
    $remoteCommand = $remoteCommandTemplate.Replace("__V4_SHA__", $v4Sha).Replace("__V4_BYTES__", [string]$v4Bytes)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $remoteCommand))
    Write-Terminal @{
        status = "deployed"
        remote_host = $RemoteHost
        expanded_archive_sha256 = $expandedSha
        v4_archive_sha256 = $v4Sha
        gpu_policy = "sequential_and_yield_to_unrelated_clients"
        remote_scope = "/home/ubuntu/biohub-peak-rank-detector-v1 and Biohub GPU guard only"
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
