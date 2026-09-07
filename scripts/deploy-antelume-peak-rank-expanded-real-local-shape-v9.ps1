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
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-local-shape-v9-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$trainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_local_shape_detector.py"
$expandedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_faint_detector.py"
$faintTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_faint_cell_pu_detector.py"
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-local-shape-v9.sh"
$expectedHashes = @{
    trainer = "1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a"
    expanded_trainer = "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe"
    faint_trainer = "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c"
    runner = "16f36d6772df38b75e82c53585ee2379d9aa5b55d2b5f774828875684cccda7e"
}

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-local-shape-v9-deploy"
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
Assert-Hash $trainer $expectedHashes.trainer "Local-shape trainer"
Assert-Hash $expandedTrainer $expectedHashes.expanded_trainer "Expanded-real base"
Assert-Hash $faintTrainer $expectedHashes.faint_trainer "Faint-cell base"
Assert-Hash $runner $expectedHashes.runner "Local-shape runner"
& python -m py_compile $trainer $expandedTrainer $faintTrainer
if ($LASTEXITCODE -ne 0) { throw "Local-shape sources do not compile" }
& bash -n "scripts/run-antelume-peak-rank-expanded-real-local-shape-v9.sh"
if ($LASTEXITCODE -ne 0) { throw "Local-shape runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_local_shape_v9_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Local-shape deployment already reached a terminal state"
}

try {
    $sshArgs = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30"
    )
    $remote = "${RemoteUser}@${RemoteHost}"
    $remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9"
    $remoteInput = "$remoteRoot/input"
    $prepare = "set -euo pipefail; test ! -e '$remoteRoot/run.sh'; test ! -e '/home/ubuntu/biohub-peak-rank-expanded-real-local-shape-v9-results.tar.gz'; mkdir -p '$remoteInput'"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $prepare))
    # The executable keeps a basename recognized by the already-installed
    # Biohub-only GPU yield guard. Its content is still bound to the distinct
    # local-shape hash; support modules cannot shadow it.
    Invoke-External "scp.exe" ($sshArgs + @($trainer, "${remote}:$remoteInput/train_expanded_real_faint_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedTrainer, "${remote}:$remoteInput/expanded_real_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($faintTrainer, "${remote}:$remoteInput/train_faint_cell_pu_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteRoot/run.sh"))

    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-local-shape-v9
input_root="$run_root/input"
echo '__TRAINER_SHA__  '"$input_root/train_expanded_real_faint_detector.py" | sha256sum -c -
echo '__EXPANDED_SHA__  '"$input_root/expanded_real_base.py" | sha256sum -c -
echo '__FAINT_SHA__  '"$input_root/train_faint_cell_pu_detector.py" | sha256sum -c -
echo '__RUNNER_SHA__  '"$run_root/run.sh" | sha256sum -c -
chmod +x "$run_root/run.sh"
bash -n "$run_root/run.sh"
nohup bash "$run_root/run.sh" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf 'RUNNER_PID=%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__TRAINER_SHA__", $expectedHashes.trainer).Replace("__EXPANDED_SHA__", $expectedHashes.expanded_trainer).Replace("__FAINT_SHA__", $expectedHashes.faint_trainer).Replace("__RUNNER_SHA__", $expectedHashes.runner)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    Write-Terminal @{
        status = "deployed_waiting_for_v7"
        remote_host = $RemoteHost
        runner_sha256 = $expectedHashes.runner
        trainer_sha256 = $expectedHashes.trainer
        gpu_policy = "sequential_after_verified_v7_and_yield_to_unrelated_clients"
        remote_scope = $remoteRoot
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
