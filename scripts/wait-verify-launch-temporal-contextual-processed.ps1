param(
    [double]$MinimumQuotaHours = 6.0,
    [double]$MaximumWaitHours = 48.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$calibrationLaunchTerminal = Join-Path $automationDir 'temporal-contextual-calibration-launch.json'
$logPath = Join-Path $automationDir 'temporal-contextual-processed-launch.log'
$terminalPath = Join-Path $automationDir 'temporal-contextual-processed-launch.json'
$calibrationRef = 'indarkarhana/biohub-temporal-contextual-calibration-v3'
$processedRef = 'indarkarhana/biohub-temporal-contextual-processed-acceptance-v3'
$processedDir = Join-Path $projectRoot 'kaggle/biohub-temporal-contextual-processed-acceptance-v3'
$processedMetadata = Join-Path $processedDir 'kernel-metadata.json'
$processedNotebook = Join-Path $processedDir 'biohub-temporal-contextual-processed-acceptance-v3.ipynb'
$verifier = Join-Path $projectRoot 'research/temporal_contrastive/verify_dual_fold_calibration_output.py'
$calibrationDownloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-calibration-v3-autochain'
$calibrationRoot = Join-Path $calibrationDownloadRoot 'temporal_contextual_calibration_v3'
$appearanceRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain/temporal_contextual_transfer_v3'
$trackastraRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/trackastra-dual-fold-synthetic-v1/trackastra_dual_fold_v1'
$expectedMetadataSha256 = '624a14a202cde15a1bfc4cf942241df1f80801094e60e8886182c8dffc826125'
$expectedNotebookSha256 = '08c0316da23940e21490d56909d3bb88c690b9103dfb0d2cf57c40de96e8665b'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'temporal-contextual-pair-fusion-processed-acceptance-v3'
        status = $Status
        calibration_ref = $calibrationRef
        processed_ref = $processedRef
        minimum_quota_hours = $MinimumQuotaHours
        quota_reserve_hours = 0.0
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        processed_metadata_sha256 = $expectedMetadataSha256
        processed_notebook_sha256 = $expectedNotebookSha256
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
    $metadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $processedMetadata).Hash.ToLowerInvariant()
    $notebookHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $processedNotebook).Hash.ToLowerInvariant()
    if ($metadataHash -ne $expectedMetadataSha256) {
        throw "Processed metadata changed: $metadataHash"
    }
    if ($notebookHash -ne $expectedNotebookSha256) {
        throw "Processed notebook changed: $notebookHash"
    }
    $metadata = Get-Content -Raw -LiteralPath $processedMetadata | ConvertFrom-Json
    $requiredKernelSources = @(
        'indarkarhana/biohub-temporal-contextual-transfer-v3',
        'indarkarhana/biohub-temporal-contextual-calibration-v3',
        'indarkarhana/biohub-trackastra-dual-fold-synthetic-v1'
    )
    $requiredDatasetSources = @(
        'indarkarhana/biohub-temporal-contextual-transfer-runtime-v1',
        'indarkarhana/biohub-trackastra-graph-runtime-v1',
        'indarkarhana/biohub-hoct-processed-validation-v1'
    )
    if (
        $metadata.id -ne $processedRef -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        @($requiredKernelSources | Where-Object { $metadata.kernel_sources -notcontains $_ }).Count -ne 0 -or
        @($requiredDatasetSources | Where-Object { $metadata.dataset_sources -notcontains $_ }).Count -ne 0 -or
        $metadata.competition_sources -notcontains 'biohub-cell-tracking-during-development'
    ) {
        throw 'Processed metadata contract is invalid'
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $calibrationLaunchTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the calibration launch terminal'
        }
        Start-Sleep -Seconds 60
    }
    $calibrationLaunch = Get-Content -Raw -LiteralPath $calibrationLaunchTerminal | ConvertFrom-Json
    if (
        $calibrationLaunch.status -ne 'launched' -or
        $calibrationLaunch.calibration_ref -ne $calibrationRef -or
        $calibrationLaunch.expected_gpu_count -ne 2 -or
        $calibrationLaunch.public_leaderboard_used_for_selection -ne $false -or
        $calibrationLaunch.submission_created -ne $false
    ) {
        throw 'Calibration autolaunch did not produce valid launch evidence'
    }

    $externalPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($externalPoll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $externalPoll += 1
        $statusOutput = (& kaggle kernels status $calibrationRef 2>&1) -join "`n"
        Write-ChainLog "calibration_status poll=$externalPoll output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Contextual calibration did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for contextual calibration completion'
    }

    if (Test-Path -LiteralPath $calibrationDownloadRoot) {
        if (
            -not (Test-Path -LiteralPath $calibrationRoot) -or
            -not (Test-Path -LiteralPath (Join-Path $calibrationDownloadRoot 'calibration_launcher_terminal.json'))
        ) {
            throw "Calibration download target exists without expected output: $calibrationDownloadRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $calibrationDownloadRoot | Out-Null
        $downloadOutput = (& kaggle kernels output $calibrationRef -p $calibrationDownloadRoot --force 2>&1) -join "`n"
        Write-ChainLog "calibration_download output=$downloadOutput"
    }

    $verificationPath = Join-Path $automationDir 'temporal-contextual-calibration-verification.json'
    $verificationError = Join-Path $automationDir 'temporal-contextual-calibration-verification.stderr.log'
    & python $verifier --root $calibrationDownloadRoot --appearance-root $appearanceRoot --trackastra-root $trackastraRoot 1> $verificationPath 2> $verificationError
    if ($LASTEXITCODE -ne 0) {
        throw "Strict calibration verifier failed; see $verificationError"
    }
    $verified = Get-Content -Raw -LiteralPath $verificationPath | ConvertFrom-Json
    $calibrationTerminal = Join-Path $calibrationRoot 'calibration_terminal.json'
    $calibrationSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $calibrationTerminal).Hash.ToLowerInvariant()
    if (
        $verified.status -ne 'verified' -or
        $verified.run_id -ne 'temporal-contextual-pair-fusion-blend-v3' -or
        $verified.appearance_family -ne 'temporal_contextual_pair_fusion_v3' -or
        $verified.gpu_count -ne 2 -or
        $verified.calibration_terminal_sha256 -ne $calibrationSha256 -or
        $verified.competition_artifacts_found -ne $false -or
        $verified.authorized_for_processed_materialization -ne $true -or
        $verified.authorized_for_submission -ne $false -or
        @($verified.selected.PSObject.Properties).Count -ne 2
    ) {
        throw 'Strict calibration verification evidence is invalid'
    }

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-ChainLog "processed_quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $processedDir 2>&1) -join "`n"
        Write-ChainLog "processed_push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $processedStatus = (& kaggle kernels status $processedRef 2>&1) -join "`n"
            if ($processedStatus -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Processed push returned success but status is invalid: $processedStatus"
            }
            Write-ChainTerminal 'launched' @{
                quota_before_hours = $remaining
                calibration_launcher_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $calibrationDownloadRoot 'calibration_launcher_terminal.json')).Hash.ToLowerInvariant()
                calibration_terminal_sha256 = $calibrationSha256
                calibration_verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verificationPath).Hash.ToLowerInvariant()
                selected = $verified.selected
                kaggle_status = $processedStatus
                push_output = $pushOutput
            }
            Write-ChainLog "processed_launched status=$processedStatus"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Processed push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch contextual processed materialization'
} catch {
    Write-ChainTerminal 'failed' @{ error = $_.Exception.Message }
    Write-ChainLog "failed error=$($_.Exception.Message)"
    exit 1
}
