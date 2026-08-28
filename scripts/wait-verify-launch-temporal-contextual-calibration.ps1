param(
    [double]$MinimumQuotaHours = 6.0,
    [double]$MaximumWaitHours = 24.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$transferLaunchTerminal = Join-Path $automationDir 'temporal-contextual-transfer-launch.json'
$logPath = Join-Path $automationDir 'temporal-contextual-calibration-launch.log'
$terminalPath = Join-Path $automationDir 'temporal-contextual-calibration-launch.json'
$transferRef = 'indarkarhana/biohub-temporal-contextual-transfer-v3'
$calibrationRef = 'indarkarhana/biohub-temporal-contextual-calibration-v3'
$calibrationDir = Join-Path $projectRoot 'kaggle/biohub-temporal-contextual-calibration-v3'
$calibrationMetadata = Join-Path $calibrationDir 'kernel-metadata.json'
$calibrationNotebook = Join-Path $calibrationDir 'biohub-temporal-contextual-calibration-v3.ipynb'
$runtimeRoot = Join-Path $projectRoot '.biohub/cache/dataset-redownloads/biohub-temporal-contextual-transfer-runtime-v1-version4'
$verifier = Join-Path $runtimeRoot 'verify_appearance_output.py'
$downloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain'
$appearanceRoot = Join-Path $downloadRoot 'temporal_contextual_transfer_v3'
$expectedRuntimeManifestSha256 = 'cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d'
$expectedMetadataSha256 = '46f0b73396f50d1e3bebb40ce2f811b20d791347dbf51b0b512e2e71e492b955'
$expectedNotebookSha256 = '4596cd9fd7211c27f6b437268c8e719847f0e7cce91438cc86195e79aba497e1'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'temporal-contextual-pair-fusion-blend-v3'
        status = $Status
        transfer_ref = $transferRef
        calibration_ref = $calibrationRef
        minimum_quota_hours = $MinimumQuotaHours
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        runtime_manifest_sha256 = $expectedRuntimeManifestSha256
        calibration_metadata_sha256 = $expectedMetadataSha256
        calibration_notebook_sha256 = $expectedNotebookSha256
        processed_acceptance_ground_truth_read = $false
        public_leaderboard_used_for_selection = $false
        submission_created = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

try {
    $runtimeManifest = Join-Path $runtimeRoot 'SOURCE_MANIFEST.json'
    $runtimeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
    if ($runtimeHash -ne $expectedRuntimeManifestSha256) {
        throw "Contextual runtime changed: $runtimeHash"
    }
    $metadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $calibrationMetadata).Hash.ToLowerInvariant()
    $notebookHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $calibrationNotebook).Hash.ToLowerInvariant()
    if ($metadataHash -ne $expectedMetadataSha256) {
        throw "Calibration metadata changed: $metadataHash"
    }
    if ($notebookHash -ne $expectedNotebookSha256) {
        throw "Calibration notebook changed: $notebookHash"
    }
    $metadata = Get-Content -Raw -LiteralPath $calibrationMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $calibrationRef -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.kernel_sources -notcontains $transferRef
    ) {
        throw 'Calibration metadata contract is invalid'
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $transferLaunchTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the transfer launch terminal'
        }
        Start-Sleep -Seconds 60
    }
    $transferLaunch = Get-Content -Raw -LiteralPath $transferLaunchTerminal | ConvertFrom-Json
    if ($transferLaunch.status -ne 'launched' -or $transferLaunch.kernel_ref -ne $transferRef) {
        throw 'Transfer autolaunch did not produce valid launch evidence'
    }

    $externalPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($externalPoll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $externalPoll += 1
        $statusOutput = (& kaggle kernels status $transferRef 2>&1) -join "`n"
        Write-ChainLog "transfer_status poll=$externalPoll output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Contextual transfer did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for contextual transfer completion'
    }

    if (Test-Path -LiteralPath $downloadRoot) {
        if (-not (Test-Path -LiteralPath $appearanceRoot)) {
            throw "Transfer download target already exists without expected output: $downloadRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $downloadRoot | Out-Null
        $downloadOutput = (& kaggle kernels output $transferRef -p $downloadRoot --force 2>&1) -join "`n"
        Write-ChainLog "transfer_download output=$downloadOutput"
    }
    $launcherPath = Join-Path $downloadRoot 'launcher_terminal.json'
    $trainingPath = Join-Path $appearanceRoot 'training_terminal.json'
    if (-not (Test-Path -LiteralPath $launcherPath) -or -not (Test-Path -LiteralPath $trainingPath)) {
        throw 'Downloaded transfer output is missing launcher or training evidence'
    }
    $launcher = Get-Content -Raw -LiteralPath $launcherPath | ConvertFrom-Json
    $trainingSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $trainingPath).Hash.ToLowerInvariant()
    if (
        $launcher.schema_version -ne 1 -or
        $launcher.run_id -ne 'temporal-contextual-pair-fusion-v3' -or
        $launcher.status -ne 'completed' -or
        $launcher.training_terminal_exists -ne $true -or
        $launcher.training_terminal_sha256 -ne $trainingSha256 -or
        $launcher.gpu_count_required -ne 2 -or
        $launcher.public_predictions_copied -ne $false -or
        $launcher.public_leaderboard_used_for_selection -ne $false -or
        $launcher.submission_created -ne $false
    ) {
        throw 'Downloaded transfer launcher evidence is invalid'
    }

    $verifyOutput = Join-Path $automationDir 'temporal-contextual-transfer-verification.json'
    $verifyError = Join-Path $automationDir 'temporal-contextual-transfer-verification.stderr.log'
    & python $verifier --root $appearanceRoot --expected-family temporal_contextual_pair_fusion_v3 --strict-checkpoint 1> $verifyOutput 2> $verifyError
    if ($LASTEXITCODE -ne 0) {
        throw "Strict transfer verifier failed; see $verifyError"
    }
    $verified = Get-Content -Raw -LiteralPath $verifyOutput | ConvertFrom-Json
    if (
        $verified.status -ne 'verified' -or
        $verified.run_id -ne 'temporal-contextual-pair-fusion-v3' -or
        $verified.appearance_family -ne 'temporal_contextual_pair_fusion_v3' -or
        $verified.gpu_count -ne 2 -or
        $verified.strict_checkpoint_loaded -ne $true -or
        $verified.competition_artifacts_found -ne $false -or
        $verified.authorized_for_calibration -ne $true -or
        $verified.authorized_for_submission -ne $false -or
        $verified.training_terminal_sha256 -ne $trainingSha256
    ) {
        throw 'Strict transfer verification evidence is invalid'
    }

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-ChainLog "calibration_quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $calibrationDir 2>&1) -join "`n"
        Write-ChainLog "calibration_push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $calibrationStatus = (& kaggle kernels status $calibrationRef 2>&1) -join "`n"
            if ($calibrationStatus -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Calibration push returned success but status is invalid: $calibrationStatus"
            }
            Write-ChainTerminal 'launched' @{
                quota_before_hours = $remaining
                transfer_launcher_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $launcherPath).Hash.ToLowerInvariant()
                transfer_training_terminal_sha256 = $trainingSha256
                transfer_verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verifyOutput).Hash.ToLowerInvariant()
                kaggle_status = $calibrationStatus
                push_output = $pushOutput
            }
            Write-ChainLog "calibration_launched status=$calibrationStatus"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Calibration push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch contextual calibration'
} catch {
    Write-ChainTerminal 'failed' @{ error = $_.Exception.Message }
    Write-ChainLog "failed error=$($_.Exception.Message)"
    exit 1
}
