param(
    [double]$MinimumQuotaHours = 7.0,
    [double]$MaximumWaitHours = 336.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$v3SubmissionTerminal = Join-Path $automationDir 'temporal-contextual-submission.json'
$v3SubmissionReceipt = Join-Path $automationDir 'temporal-contextual-submission-receipt.json'
$logPath = Join-Path $automationDir 'zebrahub-multiscale-pretrain-launch.log'
$terminalPath = Join-Path $automationDir 'zebrahub-multiscale-pretrain-launch.json'
$kernelRef = 'indarkarhana/biohub-zebrahub-multiscale-pretrain-v1'
$kernelDir = Join-Path $projectRoot 'kaggle/biohub-zebrahub-multiscale-pretrain-v1'
$metadataPath = Join-Path $kernelDir 'kernel-metadata.json'
$notebookPath = Join-Path $kernelDir 'biohub-zebrahub-multiscale-pretrain-v1.ipynb'
$runtimeRef = 'indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4'
$shardsRef = 'indarkarhana/biohub-zebrahub-contextual-shards-v1'
$v3PretrainRef = 'indarkarhana/biohub-zebrahub-contextual-pretrain-v1'
$v3AcceptanceRef = 'indarkarhana/biohub-zsns001-contextual-gate-v1'
$kernelStateScript = Join-Path $projectRoot 'scripts/get-kaggle-kernel-state.py'
$preflightScript = Join-Path $projectRoot 'scripts/build-zebrahub-multiscale-contextual-pretrain-preflight.py'
$preflightReport = Join-Path $projectRoot 'artifacts/preflights/zebrahub-multiscale-contextual-pretrain-v1-post-v3-autolaunch-v1.json'
$configPath = Join-Path $projectRoot 'config/experiments/temporal-multiscale-contextual-pair-fusion-v4.json'
$expectedMetadataSha256 = 'd7edfb27e13d11c7cc3d9e85d3683417193e6acafc4ed3c0eb13362ef39504c6'
$expectedNotebookSha256 = 'c15f7e1d9080067bec78e7567b1aba8c95132b135854cedd56951954083ed4ec'
$expectedPreflightScriptSha256 = '955a589c7f56b14bf8655c195ab835f8b30a77eb5f73c16b4abcd366806d2a64'
$expectedPreflightReportSha256 = '1e36b0bd4f3db8a7fe128ab5ac146e7da1de43c7242d415d6c1f464bfca0c104'
$expectedConfigSha256 = '55f0b775a890b1004a14a4b25d6d0608dd541cd8c760a5652e0e2fd59634e7e0'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-LaunchLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-LaunchTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'zebrahub-multiscale-contextual-pretrain-v1'
        status = $Status
        kernel_ref = $kernelRef
        prerequisite_submission_terminal = $v3SubmissionTerminal
        minimum_quota_hours = $MinimumQuotaHours
        quota_reserve_hours = 0.0
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        runtime_dataset_ref = $runtimeRef
        runtime_dataset_version = 3
        shards_dataset_ref = $shardsRef
        shards_dataset_version = 4
        metadata_sha256 = $expectedMetadataSha256
        notebook_sha256 = $expectedNotebookSha256
        preflight_report_sha256 = $expectedPreflightReportSha256
        competition_data_read = $false
        public_leaderboard_used_for_selection = $false
        submission_created = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label changed: $actual"
    }
}

if (Test-Path -LiteralPath $terminalPath) {
    Write-LaunchLog "refused_existing_terminal path=$terminalPath"
    exit 2
}

try {
    Require-Hash $metadataPath $expectedMetadataSha256 'Multiscale metadata'
    Require-Hash $notebookPath $expectedNotebookSha256 'Multiscale notebook'
    Require-Hash $preflightScript $expectedPreflightScriptSha256 'Multiscale preflight builder'
    Require-Hash $preflightReport $expectedPreflightReportSha256 'Multiscale preflight report'
    Require-Hash $configPath $expectedConfigSha256 'Multiscale experiment config'
    $preflight = Get-Content -Raw -LiteralPath $preflightReport | ConvertFrom-Json
    if (
        $preflight.run_id -ne 'zebrahub-multiscale-contextual-pretrain-v1' -or
        $preflight.checks.Count -lt 7 -or
        ($preflight.checks | Where-Object status -ne 'passed').Count -ne 0
    ) {
        throw 'Multiscale preflight evidence is invalid'
    }
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.dataset_sources.Count -ne 2 -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.dataset_sources -notcontains $shardsRef -or
        $metadata.kernel_sources.Count -ne 2 -or
        $metadata.kernel_sources -notcontains $v3PretrainRef -or
        $metadata.kernel_sources -notcontains $v3AcceptanceRef -or
        $metadata.competition_sources.Count -ne 0
    ) {
        throw 'Multiscale pretraining metadata contract is invalid'
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    Write-LaunchLog 'waiting_for_verified_v3_submission'
    while (-not (Test-Path -LiteralPath $v3SubmissionTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the verified contextual-v3 submission'
        }
        Start-Sleep -Seconds 60
    }
    if (-not (Test-Path -LiteralPath $v3SubmissionReceipt -PathType Leaf)) {
        throw 'Contextual-v3 submission terminal exists without its immutable receipt'
    }
    $submitted = Get-Content -Raw -LiteralPath $v3SubmissionTerminal | ConvertFrom-Json
    $receipt = Get-Content -Raw -LiteralPath $v3SubmissionReceipt | ConvertFrom-Json
    $receiptSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $v3SubmissionReceipt).Hash.ToLowerInvariant()
    if (
        $submitted.status -ne 'submitted' -or
        $submitted.competition -ne 'biohub-cell-tracking-during-development' -or
        $submitted.kernel_ref -ne 'indarkarhana/biohub-temporal-contextual-submission-candidate-v3' -or
        $submitted.competition_submission_performed -ne $true -or
        $submitted.receipt_sha256 -ne $receiptSha256 -or
        $receipt.status -ne 'submitted' -or
        $receipt.competition_submission_performed -ne $true -or
        $receipt.candidate_sha256 -ne $submitted.candidate_sha256
    ) {
        throw 'Contextual-v3 verified submission prerequisite is invalid'
    }

    foreach ($sourceRef in @($v3PretrainRef, $v3AcceptanceRef)) {
        $sourceStatus = (& kaggle kernels status $sourceRef 2>&1) -join "`n"
        Write-LaunchLog "source_status ref=$sourceRef output=$sourceStatus"
        if ($sourceStatus -notmatch 'COMPLETE') {
            throw "Required accepted source is not complete: $sourceRef $sourceStatus"
        }
    }
    foreach ($datasetRequirement in @(
        @{ Ref = $runtimeRef; Version = 3 },
        @{ Ref = $shardsRef; Version = 4 }
    )) {
        $datasetText = (& kaggle datasets status $datasetRequirement.Ref --format json 2>&1) -join "`n"
        $dataset = $datasetText | ConvertFrom-Json
        Write-LaunchLog "dataset_status ref=$($datasetRequirement.Ref) output=$datasetText"
        if (
            $dataset.status -ne 'ready' -or
            [int]$dataset.current_version_number -ne [int]$datasetRequirement.Version
        ) {
            throw "Required private dataset version changed: $($datasetRequirement.Ref)"
        }
    }

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-LaunchLog "quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $kernelDir 2>&1) -join "`n"
        Write-LaunchLog "push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
            if ($statusOutput -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Multiscale push succeeded but remote status is invalid: $statusOutput"
            }
            $stateText = (& python $kernelStateScript --kernel-slug $kernelRef 2>&1) -join "`n"
            $state = $stateText | ConvertFrom-Json
            $version = [int]$state.current_version_number
            if (
                $state.kernel_slug -ne $kernelRef -or
                $state.present -ne $true -or
                $version -lt 1 -or
                $state.is_private -ne $true -or
                $state.enable_gpu -ne $true -or
                $state.enable_tpu -ne $false -or
                $state.enable_internet -ne $false -or
                $state.dataset_sources.Count -ne 2 -or
                $state.dataset_sources -notcontains $runtimeRef -or
                $state.dataset_sources -notcontains $shardsRef -or
                $state.kernel_sources.Count -ne 2 -or
                $state.kernel_sources -notcontains $v3PretrainRef -or
                $state.kernel_sources -notcontains $v3AcceptanceRef -or
                $state.competition_sources.Count -ne 0
            ) {
                throw "Remote multiscale kernel state is invalid: $stateText"
            }
            Write-LaunchTerminal 'launched' @{
                prerequisite_v3_candidate_sha256 = $receipt.candidate_sha256
                prerequisite_v3_receipt_sha256 = $receiptSha256
                quota_before_hours = $remaining
                kernel_version = $version
                kaggle_status = $statusOutput
                push_output = $pushOutput
            }
            Write-LaunchLog "launched version=$version status=$statusOutput"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Multiscale pretraining push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch multiscale pretraining'
} catch {
    Write-LaunchTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-LaunchLog "failed error=$($_.Exception.Message)"
    exit 1
}
