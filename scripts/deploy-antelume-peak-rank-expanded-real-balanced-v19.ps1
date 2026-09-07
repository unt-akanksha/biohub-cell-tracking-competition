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
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-balanced-v19-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$trainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_balanced_safe_rank_detector.py"
$safeTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_safe_rank_detector.py"
$safeModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_safe_rank.py"
$multiscaleModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_multiscale.py"
$globalModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_global.py"
$blobModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_blob.py"
$localShapeTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_local_shape_detector.py"
$expandedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_faint_detector.py"
$faintTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_faint_cell_pu_detector.py"
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
$expectedHashes = @{
    trainer = "854a305143803d2b10116f83d7cf75483d6f91f5930bda8931d95323cdffc802"
    safe_trainer = "5570ffcb9734623f28db006b5d85452cc0ff7372eee05a6a65231b0104540f93"
    safe_model = "89a31ed36d172347b1e0f4e13979bbaa60701f7a86defdc2b4140f191cbbfdd8"
    multiscale_model = "56cff859acd3c6e3554fb91bb9fb3b4f7c5dca64dbab00bc2a22e8aa5483b9bf"
    global_model = "ac3f861d7693e760a4da14264ce22cfd55f964ecf5404d16f5ed44bc43acfa82"
    blob_model = "6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d"
    local_shape_trainer = "1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a"
    expanded_trainer = "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe"
    faint_trainer = "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c"
    runner = "f3453af801d13550f04e60ea632b78d2f913194abcd213aced0d259e32ac0e2d"
}

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-balanced-v19-deploy"
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
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
Assert-Hash $trainer $expectedHashes.trainer "Balanced trainer"
Assert-Hash $safeTrainer $expectedHashes.safe_trainer "Safe-rank base"
Assert-Hash $safeModel $expectedHashes.safe_model "Safe-rank model"
Assert-Hash $multiscaleModel $expectedHashes.multiscale_model "Multiscale model"
Assert-Hash $globalModel $expectedHashes.global_model "Global model"
Assert-Hash $blobModel $expectedHashes.blob_model "Blob model"
Assert-Hash $localShapeTrainer $expectedHashes.local_shape_trainer "Local-shape base"
Assert-Hash $expandedTrainer $expectedHashes.expanded_trainer "Expanded-real base"
Assert-Hash $faintTrainer $expectedHashes.faint_trainer "Faint-cell base"
Assert-Hash $runner $expectedHashes.runner "Balanced runner"
& python -m py_compile $trainer $safeTrainer $safeModel $multiscaleModel $globalModel $blobModel $localShapeTrainer $expandedTrainer $faintTrainer
if ($LASTEXITCODE -ne 0) { throw "Balanced safe-rank sources do not compile" }
& bash -n "scripts/run-antelume-peak-rank-expanded-real-balanced-v19.sh"
if ($LASTEXITCODE -ne 0) { throw "Balanced safe-rank runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_balanced_v19_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Balanced safe-rank deployment already reached a terminal state"
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
    $remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19"
    $remoteInput = "$remoteRoot/input"
    $prepare = "set -euo pipefail; test ! -e '$remoteRoot/run.sh'; test ! -e '/home/ubuntu/biohub-peak-rank-expanded-real-balanced-v19-results.tar.gz'; mkdir -p '$remoteInput'"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $prepare))
    Invoke-External "scp.exe" ($sshArgs + @($trainer, "${remote}:$remoteInput/train_expanded_real_faint_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($safeTrainer, "${remote}:$remoteInput/safe_rank_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($safeModel, "${remote}:$remoteInput/model_safe_rank.py"))
    Invoke-External "scp.exe" ($sshArgs + @($multiscaleModel, "${remote}:$remoteInput/model_multiscale.py"))
    Invoke-External "scp.exe" ($sshArgs + @($globalModel, "${remote}:$remoteInput/model_global.py"))
    Invoke-External "scp.exe" ($sshArgs + @($blobModel, "${remote}:$remoteInput/model_blob.py"))
    Invoke-External "scp.exe" ($sshArgs + @($localShapeTrainer, "${remote}:$remoteInput/local_shape_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedTrainer, "${remote}:$remoteInput/expanded_real_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($faintTrainer, "${remote}:$remoteInput/train_faint_cell_pu_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteRoot/run.sh"))

    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-balanced-v19
input_root="$run_root/input"
echo '__TRAINER_SHA__  '"$input_root/train_expanded_real_faint_detector.py" | sha256sum -c -
echo '__SAFE_TRAINER_SHA__  '"$input_root/safe_rank_base.py" | sha256sum -c -
echo '__SAFE_SHA__  '"$input_root/model_safe_rank.py" | sha256sum -c -
echo '__MULTISCALE_SHA__  '"$input_root/model_multiscale.py" | sha256sum -c -
echo '__GLOBAL_SHA__  '"$input_root/model_global.py" | sha256sum -c -
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
spec = importlib.util.spec_from_file_location('balanced_v19', 'train_expanded_real_faint_detector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.RUN_ID.endswith('safe-rank-peak-rank-v19')
assert module.safe.MODEL_FAMILY == 'safe_rank_multiscale_blob_global_temporal_peak_rank_v17'
assert module.ORIGINAL_OPTIMIZATION_COUNTS == {'44b6': 150, '6bba': 330}
assert module.BALANCED_OPTIMIZATION_COUNTS == {'44b6': 300, '6bba': 330}
PY
nohup bash "$run_root/run.sh" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf 'RUNNER_PID=%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__TRAINER_SHA__", $expectedHashes.trainer).Replace("__SAFE_TRAINER_SHA__", $expectedHashes.safe_trainer).Replace("__SAFE_SHA__", $expectedHashes.safe_model).Replace("__MULTISCALE_SHA__", $expectedHashes.multiscale_model).Replace("__GLOBAL_SHA__", $expectedHashes.global_model).Replace("__BLOB_SHA__", $expectedHashes.blob_model).Replace("__LOCAL_SHAPE_SHA__", $expectedHashes.local_shape_trainer).Replace("__EXPANDED_SHA__", $expectedHashes.expanded_trainer).Replace("__FAINT_SHA__", $expectedHashes.faint_trainer).Replace("__RUNNER_SHA__", $expectedHashes.runner)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    Write-Terminal @{
        status = "deployed_waiting_for_v17"
        remote_host = $RemoteHost
        runner_sha256 = $expectedHashes.runner
        trainer_sha256 = $expectedHashes.trainer
        model_family = "safe_rank_multiscale_blob_global_temporal_peak_rank_v17"
        optimization_sampling_policy = "duplicate_44b6_once_balance_embryo_crops"
        gpu_policy = "sequential_after_verified_v17_and_yield_to_unrelated_clients"
        remote_scope = $remoteRoot
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
