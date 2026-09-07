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
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-blob-v11-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$trainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_blob_detector.py"
$blobModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_blob.py"
$localShapeTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_local_shape_detector.py"
$expandedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_faint_detector.py"
$faintTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_faint_cell_pu_detector.py"
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-blob-v11.sh"
$expectedHashes = @{
    trainer = "381ca892ee0baf0a3485d238b8981ea32706708a52cfbb4d31569b14aedb52c0"
    blob_model = "6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d"
    local_shape_trainer = "1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a"
    expanded_trainer = "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe"
    faint_trainer = "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c"
    runner = "17cdc28e609c0d5ff5d67723bc49657df2fa0a869207fd172af4d56770c384fb"
}

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-blob-v11-deploy"
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
Assert-Hash $trainer $expectedHashes.trainer "Blob trainer"
Assert-Hash $blobModel $expectedHashes.blob_model "Blob model"
Assert-Hash $localShapeTrainer $expectedHashes.local_shape_trainer "Local-shape base"
Assert-Hash $expandedTrainer $expectedHashes.expanded_trainer "Expanded-real base"
Assert-Hash $faintTrainer $expectedHashes.faint_trainer "Faint-cell base"
Assert-Hash $runner $expectedHashes.runner "Blob runner"
& python -m py_compile $trainer $blobModel $localShapeTrainer $expandedTrainer $faintTrainer
if ($LASTEXITCODE -ne 0) { throw "Blob lane sources do not compile" }
& bash -n "scripts/run-antelume-peak-rank-expanded-real-blob-v11.sh"
if ($LASTEXITCODE -ne 0) { throw "Blob runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_blob_v11_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Blob deployment already reached a terminal state"
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
    $remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-blob-v11"
    $remoteInput = "$remoteRoot/input"
    $prepare = "set -euo pipefail; test ! -e '$remoteRoot/run.sh'; test ! -e '/home/ubuntu/biohub-peak-rank-expanded-real-blob-v11-results.tar.gz'; mkdir -p '$remoteInput'"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $prepare))
    Invoke-External "scp.exe" ($sshArgs + @($trainer, "${remote}:$remoteInput/train_expanded_real_faint_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($blobModel, "${remote}:$remoteInput/model_blob.py"))
    Invoke-External "scp.exe" ($sshArgs + @($localShapeTrainer, "${remote}:$remoteInput/local_shape_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedTrainer, "${remote}:$remoteInput/expanded_real_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($faintTrainer, "${remote}:$remoteInput/train_faint_cell_pu_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteRoot/run.sh"))

    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-blob-v11
input_root="$run_root/input"
echo '__TRAINER_SHA__  '"$input_root/train_expanded_real_faint_detector.py" | sha256sum -c -
echo '__BLOB_SHA__  '"$input_root/model_blob.py" | sha256sum -c -
echo '__LOCAL_SHAPE_SHA__  '"$input_root/local_shape_base.py" | sha256sum -c -
echo '__EXPANDED_SHA__  '"$input_root/expanded_real_base.py" | sha256sum -c -
echo '__FAINT_SHA__  '"$input_root/train_faint_cell_pu_detector.py" | sha256sum -c -
echo '__RUNNER_SHA__  '"$run_root/run.sh" | sha256sum -c -
chmod +x "$run_root/run.sh"
bash -n "$run_root/run.sh"
cd "$input_root"
PYTHONPATH="$input_root:/home/ubuntu/biohub" /home/ubuntu/venv/bin/python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location('blob_v11', 'train_expanded_real_faint_detector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.RUN_ID.endswith('blob-peak-rank-v11')
assert module.MODEL_FAMILY == 'blob_aware_temporal_peak_rank_v11'
PY
nohup bash "$run_root/run.sh" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf 'RUNNER_PID=%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__TRAINER_SHA__", $expectedHashes.trainer).Replace("__BLOB_SHA__", $expectedHashes.blob_model).Replace("__LOCAL_SHAPE_SHA__", $expectedHashes.local_shape_trainer).Replace("__EXPANDED_SHA__", $expectedHashes.expanded_trainer).Replace("__FAINT_SHA__", $expectedHashes.faint_trainer).Replace("__RUNNER_SHA__", $expectedHashes.runner)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    Write-Terminal @{
        status = "deployed_waiting_for_v9"
        remote_host = $RemoteHost
        runner_sha256 = $expectedHashes.runner
        trainer_sha256 = $expectedHashes.trainer
        model_family = "blob_aware_temporal_peak_rank_v11"
        gpu_policy = "sequential_after_verified_v9_and_yield_to_unrelated_clients"
        remote_scope = $remoteRoot
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
