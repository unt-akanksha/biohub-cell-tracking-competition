param(
    [double]$MinimumQuotaHours = 6.5,
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$parentTerminal = Join-Path $automationDir 'temporal-multiscale-submission.json'
$logPath = Join-Path $automationDir 'multiscale-division-localization-launch.log'
$terminalPath = Join-Path $automationDir 'multiscale-division-localization-launch.json'
$kernelDir = Join-Path $projectRoot 'kaggle/biohub-multiscale-division-localization-v1'
$metadataPath = Join-Path $kernelDir 'kernel-metadata.json'
$notebookPath = Join-Path $kernelDir 'biohub-multiscale-division-localization-v1.ipynb'
$kernelStateScript = Join-Path $projectRoot 'scripts/get-kaggle-kernel-state.py'
$kernelRef = 'indarkarhana/biohub-multiscale-division-localization-v1'
$parentRef = 'indarkarhana/biohub-temporal-multiscale-transfer-v4'
$runtimeRef = 'indarkarhana/biohub-division-localization-runtime-v1'
$trainDataRef = 'indarkarhana/biohub-zebrahub-contextual-shards-v1'
$localizationDataRef = 'indarkarhana/biohub-division-localization-shards-v1'
$expectedMetadataSha256 = 'd8618610e8abd84fbb5eb396da805e987a11912311329347a6c91bb7c6db6d07'
$expectedNotebookSha256 = 'bb57cb73d0701ca2b8486b501523b5e2eb9d9e092d1679a78e752b8b5dc16817'
$expectedStateScriptSha256 = 'ad4f9621122edbf4801ce350db2b8723d70964cb1258c19685f13ef5cdb60d60'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'multiscale-division-localization-v1'
        status = $Status
        kernel_ref = $kernelRef
        parent_ref = $parentRef
        minimum_quota_hours = $MinimumQuotaHours
        quota_reserve_hours = 0.0
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        metadata_sha256 = $expectedMetadataSha256
        notebook_sha256 = $expectedNotebookSha256
        competition_data_attached = $false
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
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

try {
    Require-Hash $metadataPath $expectedMetadataSha256 'Localization kernel metadata'
    Require-Hash $notebookPath $expectedNotebookSha256 'Localization notebook'
    Require-Hash $kernelStateScript $expectedStateScriptSha256 'Kaggle state reader'
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.dataset_sources -notcontains $trainDataRef -or
        $metadata.dataset_sources -notcontains $localizationDataRef -or
        $metadata.kernel_sources -notcontains $parentRef -or
        $metadata.competition_sources.Count -ne 0
    ) {
        throw 'Localization kernel metadata contract is invalid'
    }
    if ($ValidateOnly) {
        [pscustomobject]@{
            status = 'validated'
            stage = 'multiscale_division_localization_launch'
            expected_gpu_count = 2
            quota_reserve_hours = 0.0
        } | ConvertTo-Json
        exit 0
    }
    if (Test-Path -LiteralPath $terminalPath) {
        throw "Localization launch terminal already exists: $terminalPath"
    }
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $parentTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the verified v4 submission terminal'
        }
        Start-Sleep -Seconds 60
    }
    $parent = Get-Content -Raw -LiteralPath $parentTerminal | ConvertFrom-Json
    if (
        $parent.status -ne 'submitted' -or
        $parent.run_id -ne 'temporal-multiscale-contextual-pair-fusion-candidate-v4' -or
        $parent.kernel_ref -ne 'indarkarhana/biohub-multiscale-submission-candidate-v4' -or
        $parent.public_leaderboard_used_for_selection -ne $false -or
        $parent.competition_submission_performed -ne $true
    ) {
        throw 'Verified v4 submission terminal is invalid'
    }
    foreach ($dataset in @(
        @{ Ref = $runtimeRef; Version = 1 },
        @{ Ref = $localizationDataRef; Version = 1 }
    )) {
        $statusText = (& kaggle datasets status $dataset.Ref --format json 2>&1) -join "`n"
        $status = $statusText | ConvertFrom-Json
        if ($status.status -ne 'ready' -or [int]$status.current_version_number -ne $dataset.Version) {
            throw "Localization dataset is not ready: $statusText"
        }
    }
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-ChainLog "quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $kernelDir 2>&1) -join "`n"
        Write-ChainLog "kernel_push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $stateText = (& python $kernelStateScript --kernel-slug $kernelRef 2>&1) -join "`n"
            $state = $stateText | ConvertFrom-Json
            $version = [int]$state.current_version_number
            if (
                $state.present -ne $true -or
                $state.kernel_slug -ne $kernelRef -or
                $version -lt 1 -or
                $state.is_private -ne $true -or
                $state.enable_gpu -ne $true -or
                $state.enable_tpu -ne $false -or
                $state.enable_internet -ne $false -or
                $state.dataset_sources -notcontains $runtimeRef -or
                $state.dataset_sources -notcontains $trainDataRef -or
                $state.dataset_sources -notcontains $localizationDataRef -or
                $state.kernel_sources -notcontains $parentRef -or
                $state.competition_sources.Count -ne 0
            ) {
                throw "Remote localization kernel state is invalid: $stateText"
            }
            Write-ChainTerminal 'launched' @{
                kernel_version = $version
                quota_before_hours = $remaining
                parent_receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $parentTerminal).Hash.ToLowerInvariant()
                push_output = $pushOutput
            }
            Write-ChainLog "localization_launched version=$version"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Localization kernel push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch division localization'
} catch {
    if (-not $ValidateOnly) {
        Write-ChainTerminal 'failed' @{
            error = $_.Exception.Message
        }
        Write-ChainLog "failed error=$($_.Exception.Message)"
    }
    throw
}

