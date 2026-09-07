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
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-xl-hard-mined-temporal-snr-v31-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$v21ValidationPath = Join-Path $RepositoryRoot ".biohub/automation/peak-rank-expanded-real-xl-balanced-validation-controller-v21.json"
$v27ValidationPath = Join-Path $RepositoryRoot ".biohub/automation/peak-rank-hard-mined-temporal-snr-validation-controller-v27.json"
$graphHarvestPath = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2/harvest-terminal.json"
$trainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_xl_temporal_stable_hard_mined_safe_rank_detector.py"
$hardMinedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_temporal_stable_hard_mined_safe_rank_detector.py"
$temporalTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_temporal_stable_balanced_safe_rank_detector.py"
$stableModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_temporal_stable.py"
$balancedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_balanced_safe_rank_detector.py"
$safeTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_safe_rank_detector.py"
$safeModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_safe_rank.py"
$multiscaleModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_multiscale.py"
$globalModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_global.py"
$blobModel = Join-Path $RepositoryRoot "research/peak_rank_detection/model_blob.py"
$localShapeTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_local_shape_detector.py"
$expandedTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_expanded_real_faint_detector.py"
$faintTrainer = Join-Path $RepositoryRoot "research/peak_rank_detection/train_faint_cell_pu_detector.py"
$samplingManifest = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-expanded-replay-v2/hard-mining-manifest-v27.json"
$runner = Join-Path $RepositoryRoot "scripts/run-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.sh"
$expectedHashes = @{
    trainer = "475ca05d1d321c43f5b711f94c55ebf16bbf125d83d8306ead1c3c0d5af8c891"
    hard_mined_trainer = "1832e8dc7738ca778b3b4f07159e68741d8d6fef2e30e2a14b1075748851e7c8"
    temporal_trainer = "57479c473929aad7f2fdaca5c1322ea88e5f051006faf32c96d066ee4f365a79"
    stable_model = "04f2c5d580279b7d035007ef0d5c422ad6ee43fec430f8f8a809d69858db172e"
    balanced_trainer = "854a305143803d2b10116f83d7cf75483d6f91f5930bda8931d95323cdffc802"
    safe_trainer = "5570ffcb9734623f28db006b5d85452cc0ff7372eee05a6a65231b0104540f93"
    safe_model = "89a31ed36d172347b1e0f4e13979bbaa60701f7a86defdc2b4140f191cbbfdd8"
    multiscale_model = "56cff859acd3c6e3554fb91bb9fb3b4f7c5dca64dbab00bc2a22e8aa5483b9bf"
    global_model = "ac3f861d7693e760a4da14264ce22cfd55f964ecf5404d16f5ed44bc43acfa82"
    blob_model = "6728c620a1fd10d4e192f9d1e6c9859971bc335b1bedb50ef54e3d85bc684b3d"
    local_shape_trainer = "1c3f4fd526aa2126f7b14403d91b4493efbc3dc0d48903992c84bb3aecd8654a"
    expanded_trainer = "6eb0f506204c1f30fbee3ad9859215826a07fe7c8d78a2be868371233fbd2ebe"
    faint_trainer = "6ef8092a89c1c01c536c690da573a10b49ce99bf83d4d3ddd69403741f808b0c"
    sampling_manifest = "9967efa25021453b4f043a23e47e744153da678f17f5c00180ecd6ebd4dca900"
    runner = "056683ce3a02f295ec245b534a4654d0ce74426b1cbe1d76b83805965ce8e419"
}

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-xl-hard-mined-temporal-snr-v31-deploy"
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

function Read-CleanValidation([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label evidence is missing" }
    $payload = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if (-not (
        $payload.status -eq "completed" -and
        $payload.promotion_passed -eq $true -and
        $payload.accepted_for_candidate_integration -eq $true -and
        $payload.competition_submission_performed -eq $false
    )) { throw "$Label did not independently pass clean validation" }
    return $payload
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
Assert-Hash $trainer $expectedHashes.trainer "V31 trainer"
Assert-Hash $hardMinedTrainer $expectedHashes.hard_mined_trainer "V27 hard-mined base"
Assert-Hash $temporalTrainer $expectedHashes.temporal_trainer "V23 temporal base"
Assert-Hash $stableModel $expectedHashes.stable_model "Temporal-SNR model"
Assert-Hash $balancedTrainer $expectedHashes.balanced_trainer "Balanced base"
Assert-Hash $safeTrainer $expectedHashes.safe_trainer "Safe-rank base"
Assert-Hash $safeModel $expectedHashes.safe_model "Safe-rank model"
Assert-Hash $multiscaleModel $expectedHashes.multiscale_model "Multiscale model"
Assert-Hash $globalModel $expectedHashes.global_model "Global model"
Assert-Hash $blobModel $expectedHashes.blob_model "Blob model"
Assert-Hash $localShapeTrainer $expectedHashes.local_shape_trainer "Local-shape base"
Assert-Hash $expandedTrainer $expectedHashes.expanded_trainer "Expanded-real base"
Assert-Hash $faintTrainer $expectedHashes.faint_trainer "Faint-cell base"
Assert-Hash $samplingManifest $expectedHashes.sampling_manifest "Hard-mining manifest"
Assert-Hash $runner $expectedHashes.runner "V31 runner"
& python -m py_compile $trainer $hardMinedTrainer $temporalTrainer $stableModel $balancedTrainer $safeTrainer $safeModel $multiscaleModel $globalModel $blobModel $localShapeTrainer $expandedTrainer $faintTrainer
if ($LASTEXITCODE -ne 0) { throw "V31 sources do not compile" }
& bash -n "scripts/run-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.sh"
if ($LASTEXITCODE -ne 0) { throw "V31 runner does not parse" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "antelume_peak_rank_xl_hard_mined_temporal_snr_v31_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) { throw "V31 deployment already reached a terminal state" }

try {
    $v21Validation = Read-CleanValidation $v21ValidationPath "V21 XL detector"
    $v27Validation = Read-CleanValidation $v27ValidationPath "V27 hard-mined detector"
    if (-not (Test-Path -LiteralPath $graphHarvestPath -PathType Leaf)) {
        throw "Graph v2 harvest evidence is missing"
    }
    $graphHarvest = Get-Content -Raw -LiteralPath $graphHarvestPath | ConvertFrom-Json
    if (-not (
        @("accepted", "scientifically_rejected") -contains $graphHarvest.status -and
        $graphHarvest.competition_submission_performed -eq $false -and
        $graphHarvest.remote_harvest_acknowledged -eq $true
    )) { throw "Graph v2 archive was not independently verified and acknowledged" }

    $sshArgs = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30"
    )
    $remote = "${RemoteUser}@${RemoteHost}"
    $remoteRoot = "/home/ubuntu/biohub-peak-rank-detector-v1/xl-hard-mined-temporal-snr-v31"
    $remoteInput = "$remoteRoot/input"
    $prepare = "set -euo pipefail; test ! -e '$remoteRoot/run.sh'; test ! -e '/home/ubuntu/biohub-peak-rank-xl-hard-mined-temporal-snr-v31-results.tar.gz'; mkdir -p '$remoteInput'; readlink -f '/home/ubuntu/biohub/model.py' | grep -Fx '/home/ubuntu/biohub/research/peak_rank_detection/model.py'"
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $prepare))
    Invoke-External "scp.exe" ($sshArgs + @($trainer, "${remote}:$remoteInput/train_expanded_real_faint_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($hardMinedTrainer, "${remote}:$remoteInput/hard_mined_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($temporalTrainer, "${remote}:$remoteInput/temporal_stable_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($stableModel, "${remote}:$remoteInput/model_temporal_stable.py"))
    Invoke-External "scp.exe" ($sshArgs + @($balancedTrainer, "${remote}:$remoteInput/balanced_safe_rank_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($safeTrainer, "${remote}:$remoteInput/safe_rank_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($safeModel, "${remote}:$remoteInput/model_safe_rank.py"))
    Invoke-External "scp.exe" ($sshArgs + @($multiscaleModel, "${remote}:$remoteInput/model_multiscale.py"))
    Invoke-External "scp.exe" ($sshArgs + @($globalModel, "${remote}:$remoteInput/model_global.py"))
    Invoke-External "scp.exe" ($sshArgs + @($blobModel, "${remote}:$remoteInput/model_blob.py"))
    Invoke-External "scp.exe" ($sshArgs + @($localShapeTrainer, "${remote}:$remoteInput/local_shape_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($expandedTrainer, "${remote}:$remoteInput/expanded_real_base.py"))
    Invoke-External "scp.exe" ($sshArgs + @($faintTrainer, "${remote}:$remoteInput/train_faint_cell_pu_detector.py"))
    Invoke-External "scp.exe" ($sshArgs + @($samplingManifest, "${remote}:$remoteInput/hard-mining-manifest-v27.json"))
    Invoke-External "scp.exe" ($sshArgs + @($runner, "${remote}:$remoteRoot/run.sh"))

    $launchTemplate = @'
set -euo pipefail
run_root=/home/ubuntu/biohub-peak-rank-detector-v1/xl-hard-mined-temporal-snr-v31
input_root="$run_root/input"
echo '__TRAINER_SHA__  '"$input_root/train_expanded_real_faint_detector.py" | sha256sum -c -
echo '__HARD_SHA__  '"$input_root/hard_mined_base.py" | sha256sum -c -
echo '__TEMPORAL_SHA__  '"$input_root/temporal_stable_base.py" | sha256sum -c -
echo '__STABLE_SHA__  '"$input_root/model_temporal_stable.py" | sha256sum -c -
echo '__BALANCED_SHA__  '"$input_root/balanced_safe_rank_base.py" | sha256sum -c -
echo '__SAFE_TRAINER_SHA__  '"$input_root/safe_rank_base.py" | sha256sum -c -
echo '__SAFE_MODEL_SHA__  '"$input_root/model_safe_rank.py" | sha256sum -c -
echo '__MULTISCALE_SHA__  '"$input_root/model_multiscale.py" | sha256sum -c -
echo '__GLOBAL_SHA__  '"$input_root/model_global.py" | sha256sum -c -
echo '__BLOB_SHA__  '"$input_root/model_blob.py" | sha256sum -c -
echo '__LOCAL_SHA__  '"$input_root/local_shape_base.py" | sha256sum -c -
echo '__EXPANDED_SHA__  '"$input_root/expanded_real_base.py" | sha256sum -c -
echo '__FAINT_SHA__  '"$input_root/train_faint_cell_pu_detector.py" | sha256sum -c -
echo '__MANIFEST_SHA__  '"$input_root/hard-mining-manifest-v27.json" | sha256sum -c -
echo '__RUNNER_SHA__  '"$run_root/run.sh" | sha256sum -c -
chmod +x "$run_root/run.sh"
bash -n "$run_root/run.sh"
cd "$input_root"
PYTHONPATH="$input_root:/home/ubuntu/biohub" /home/ubuntu/venv/bin/python - <<'PY'
import importlib.util, json
from model_temporal_stable import LOCAL_SNR_BANDS, TemporalMinimumLocalSnrSafeRankDetector
spec = importlib.util.spec_from_file_location('xl_hard_v31', 'train_expanded_real_faint_detector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = json.load(open('hard-mining-manifest-v27.json'))
assert module.RUN_ID.endswith('safe-rank-peak-rank-v31')
assert module.hard_mined.SAMPLING_MANIFEST_SHA256 == '__MANIFEST_SHA__'
assert manifest['effective_counts'] == {'44b6': 495, '6bba': 495}
assert manifest['selection_data_read'] is False
assert manifest['sealed_audit_data_read'] is False
assert LOCAL_SNR_BANDS == ((3, 9), (3, 11), (5, 13))
model = TemporalMinimumLocalSnrSafeRankDetector(widths=(160, 320, 640, 1280), depths=(3, 3, 9, 3))
assert sum(parameter.numel() for parameter in model.parameters()) == 129761606
PY
nohup bash "$run_root/run.sh" >"$run_root/controller.log" 2>&1 < /dev/null &
runner_pid=$!
sleep 2
kill -0 "$runner_pid"
printf 'RUNNER_PID=%s\n' "$runner_pid"
'@
    $launch = $launchTemplate.Replace("__TRAINER_SHA__", $expectedHashes.trainer).Replace("__HARD_SHA__", $expectedHashes.hard_mined_trainer).Replace("__TEMPORAL_SHA__", $expectedHashes.temporal_trainer).Replace("__STABLE_SHA__", $expectedHashes.stable_model).Replace("__BALANCED_SHA__", $expectedHashes.balanced_trainer).Replace("__SAFE_TRAINER_SHA__", $expectedHashes.safe_trainer).Replace("__SAFE_MODEL_SHA__", $expectedHashes.safe_model).Replace("__MULTISCALE_SHA__", $expectedHashes.multiscale_model).Replace("__GLOBAL_SHA__", $expectedHashes.global_model).Replace("__BLOB_SHA__", $expectedHashes.blob_model).Replace("__LOCAL_SHA__", $expectedHashes.local_shape_trainer).Replace("__EXPANDED_SHA__", $expectedHashes.expanded_trainer).Replace("__FAINT_SHA__", $expectedHashes.faint_trainer).Replace("__MANIFEST_SHA__", $expectedHashes.sampling_manifest).Replace("__RUNNER_SHA__", $expectedHashes.runner)
    Invoke-External "ssh.exe" ($sshArgs + @($remote, $launch))
    Write-Terminal @{
        status = "deployed_waiting_for_verified_graph_harvest"
        remote_host = $RemoteHost
        remote_scope = $remoteRoot
        runner_sha256 = $expectedHashes.runner
        trainer_sha256 = $expectedHashes.trainer
        hard_mining_manifest_sha256 = $expectedHashes.sampling_manifest
        v21_validation_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $v21ValidationPath).Hash.ToLowerInvariant()
        v27_validation_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $v27ValidationPath).Hash.ToLowerInvariant()
        graph_harvest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $graphHarvestPath).Hash.ToLowerInvariant()
        parameter_count = 129761606
        model_family = "temporal_min_local_snr_safe_rank_multiscale_global_peak_rank_v23"
        optimization_sampling_policy = "embryo_balanced_495_each_v23_local_snr_hardest_first"
        effective_optimization_counts = @{ "44b6" = 495; "6bba" = 495 }
        gpu_policy = "conditional_after_clean_v21_v27_and_acknowledged_graph_harvest"
        steps = 6000
        seed = 14790551
        maximum_wall_seconds = 108000
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
