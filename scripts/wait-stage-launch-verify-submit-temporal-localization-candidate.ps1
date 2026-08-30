param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 72.0,
    [int]$PollSeconds = 120,
    [int]$KernelPollSeconds = 300,
    [double]$MinimumGpuReserveHours = 8.0,
    [int]$DeclaredCandidateBudgetSeconds = 39600,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if (
    $MaximumWaitHours -le 0 -or $PollSeconds -lt 60 -or
    $KernelPollSeconds -lt 60 -or $MinimumGpuReserveHours -lt 0 -or
    $DeclaredCandidateBudgetSeconds -ne 39600
) {
    throw "Invalid temporal-localization controller bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$competitionRef = "biohub-cell-tracking-during-development"
$runtimeRef = "indarkarhana/biohub-temporal-localization-consensus-v1"
$kernelRef = "indarkarhana/biohub-ema-temporal-localization-v1"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/aws-4gpu-temporal-localizer-v1"
$harvestTerminal = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "biohub-synthetic256-temporal-node-localizer-v1-results.tar.gz"
$extractRoot = Join-Path $stateRoot "harvested-results"
$resultsRoot = Join-Path $extractRoot "synthetic256-temporal-node-localizer-v1"
$runtimeRoot = Join-Path $RepositoryRoot ".biohub/cache/runtime-datasets/biohub-temporal-localization-consensus-v1"
$runtimeManifest = Join-Path $runtimeRoot "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json"
$candidateDir = Join-Path $RepositoryRoot "kaggle/biohub-ema-temporal-localization-v1"
$candidateNotebook = Join-Path $candidateDir "biohub-ema-temporal-localization-v1.ipynb"
$candidateMetadata = Join-Path $candidateDir "kernel-metadata.json"
$datasetBuilder = Join-Path $RepositoryRoot "scripts/build-temporal-localization-candidate-dataset.py"
$candidateBuilder = Join-Path $RepositoryRoot "scripts/build-temporal-localization-submission-candidate.py"
$verifier = Join-Path $RepositoryRoot "scripts/verify-temporal-localization-submission-candidate.py"
$submitter = Join-Path $RepositoryRoot "scripts/submit-temporal-localization-candidate.py"
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$baselineValidator = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv"
$expectedBaselineValidatorSha256 = "4dbf2079c1efc1108370f33104a6e80882a851d45dbfaa0540d40e88e4941f4b"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "temporal-localization-candidate-controller-v1.json"
$logPath = Join-Path $automationRoot "temporal-localization-candidate-controller-v1.log"
$quotaPath = Join-Path $automationRoot "temporal-localization-candidate-quota-before.json"
$receiptPath = Join-Path $automationRoot "temporal-localization-candidate-submission-receipt.json"
$promotionPath = Join-Path $automationRoot "temporal-localization-candidate-promotion.json"
$requiredOutputPattern = "^(candidate_evidence\.json|run_stats\.csv|submission\.csv|validator_results\.csv|watchdog-terminal\.json)$"

function Invoke-NativeOutput([scriptblock]$Command) {
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $lines = & $Command 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = ($lines -join "`n") }
}

function Write-ControllerLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Write-ControllerTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "temporal-localization-candidate-controller-v1"
        status = $Status
        target_public_score = 0.945
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        public_leaderboard_used_for_selection = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Assert-SafeArchive([string]$Path) {
    $native = Invoke-NativeOutput { & tar -tzf $Path }
    if ($native.ExitCode -ne 0) { throw "Could not inspect harvested archive: $($native.Output)" }
    $entries = @($native.Output -split "`r?`n" | Where-Object { $_ })
    if ($entries.Count -lt 1) { throw "Harvested archive is empty" }
    foreach ($rawEntry in $entries) {
        $entry = $rawEntry.Replace("\", "/")
        $segments = @($entry -split "/" | Where-Object { $_ })
        if (
            $entry.StartsWith("/") -or [IO.Path]::IsPathRooted($entry) -or
            $segments -contains ".." -or $segments.Count -lt 1 -or
            $segments[0] -ne "synthetic256-temporal-node-localizer-v1"
        ) {
            throw "Unsafe path in harvested archive: $rawEntry"
        }
    }
}

New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
foreach ($required in @(
    $datasetBuilder, $candidateBuilder, $verifier, $submitter,
    $kernelStateScript, $evaluationPython, $baselineValidator
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required temporal-localization controller input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle", "tar")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required temporal-localization command is missing: $command"
    }
}
$baselineValidatorSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $baselineValidator
).Hash.ToLowerInvariant()
if ($baselineValidatorSha256 -ne $expectedBaselineValidatorSha256) {
    throw "Pinned temporal-localization baseline validator changed"
}

if ($ValidateOnly) {
    & python -m py_compile $datasetBuilder $candidateBuilder $verifier $submitter $kernelStateScript
    if ($LASTEXITCODE -ne 0) { throw "Temporal-localization Python validation failed" }
    @{
        schema_version = 1
        status = "validated"
        run_id = "temporal-localization-candidate-controller-v1"
        waits_for_aws_harvest = $true
        minimum_gpu_reserve_hours = $MinimumGpuReserveHours
        declared_candidate_budget_seconds = $DeclaredCandidateBudgetSeconds
        candidate_submission_requires_external_promotion = $true
        public_leaderboard_used_for_selection = $false
    } | ConvertTo-Json
    exit 0
}

foreach ($forbidden in @(
    $terminalPath, $extractRoot, $runtimeRoot, $candidateDir,
    $promotionPath, $receiptPath, $quotaPath
)) {
    if (Test-Path -LiteralPath $forbidden) {
        throw "Refusing to reuse temporal-localization controller state: $forbidden"
    }
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    while (-not (Test-Path -LiteralPath $harvestTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Timed out waiting for the AWS temporal-localizer harvest"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $harvest = Get-Content -Raw -LiteralPath $harvestTerminal | ConvertFrom-Json
    if ($harvest.status -ne "harvested") {
        Write-ControllerTerminal "skipped_after_aws_failure" @{
            aws_harvest_status = $harvest.status
            dataset_uploaded = $false
            kernel_launched = $false
            competition_submission_performed = $false
        }
        exit 0
    }
    if (
        $harvest.run_id -ne "aws-4gpu-temporal-localizer-harvest-v1" -or
        $harvest.authorized_for_submission -ne $false -or
        -not (Test-Path -LiteralPath $archivePath -PathType Leaf)
    ) {
        throw "AWS temporal-localizer harvest evidence is invalid"
    }
    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    $archiveBytes = (Get-Item -LiteralPath $archivePath).Length
    if ($archiveHash -ne [string]$harvest.result_sha256 -or $archiveBytes -ne [int64]$harvest.result_bytes) {
        throw "AWS temporal-localizer archive changed after hash-bound harvest"
    }
    Assert-SafeArchive $archivePath
    New-Item -ItemType Directory -Path $extractRoot | Out-Null
    $native = Invoke-NativeOutput { & tar -xzf $archivePath -C $extractRoot }
    if ($native.ExitCode -ne 0) { throw "AWS temporal-localizer extraction failed: $($native.Output)" }
    if (-not (Test-Path -LiteralPath $resultsRoot -PathType Container)) {
        throw "AWS temporal-localizer archive has an unexpected root"
    }

    $native = Invoke-NativeOutput {
        & python $datasetBuilder --results-root $resultsRoot --output-root $runtimeRoot
    }
    if ($native.ExitCode -ne 0) {
        Write-ControllerTerminal "skipped_after_scientific_rejection" @{
            scientific_gate_error = $native.Output
            dataset_uploaded = $false
            kernel_launched = $false
            competition_submission_performed = $false
        }
        exit 0
    }
    if (-not (Test-Path -LiteralPath $runtimeManifest -PathType Leaf)) {
        throw "Temporal-localization runtime manifest was not produced"
    }
    $runtimeManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeRoot "dataset-metadata.json") | ConvertFrom-Json
    if ($runtimeMetadata.id -ne $runtimeRef -or $runtimeMetadata.isPrivate -ne $true) {
        throw "Temporal-localization private dataset metadata is invalid"
    }

    $native = Invoke-NativeOutput {
        & kaggle datasets list --mine -s "biohub-temporal-localization-consensus-v1" --format json
    }
    if ($native.ExitCode -ne 0) { throw "Kaggle dataset inventory failed: $($native.Output)" }
    if ($native.Output -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput {
            & kaggle datasets version -p $runtimeRoot -m "Sealed-audit and real-gated temporal localization consensus v1"
        }
        $datasetOperation = "versioned"
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOperation = "created"
        if ($native.ExitCode -ne 0 -and $native.Output -match "(?i)already exists|conflict") {
            $native = Invoke-NativeOutput {
                & kaggle datasets version -p $runtimeRoot -m "Sealed-audit and real-gated temporal localization consensus v1"
            }
            $datasetOperation = "versioned_after_create_conflict"
        }
    }
    if ($native.ExitCode -ne 0) { throw "Kaggle runtime upload failed: $($native.Output)" }
    Write-ControllerLog "runtime_uploaded operation=$datasetOperation output=$($native.Output)"

    $datasetReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $runtimeRef --format json }
        Write-ControllerLog "runtime_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)ready|complete") {
            $datasetReady = $true
            break
        }
        if ($native.Output -match "(?i)error|failed") {
            throw "Kaggle runtime processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $datasetReady) { throw "Timed out waiting for the Kaggle runtime dataset" }

    $native = Invoke-NativeOutput { & kaggle quota --format json }
    if ($native.ExitCode -ne 0) { throw "Could not verify Kaggle GPU quota: $($native.Output)" }
    $quota = $native.Output | ConvertFrom-Json
    $gpuQuota = @($quota | Where-Object { $_.resource -eq "GPU" })
    if ($gpuQuota.Count -ne 1) { throw "Kaggle GPU quota response is invalid" }
    $remainingGpuHours = [double](([string]$gpuQuota[0].remaining) -replace "h$", "")
    $requiredGpuHours = $MinimumGpuReserveHours + ($DeclaredCandidateBudgetSeconds / 3600.0)
    if ($remainingGpuHours -lt $requiredGpuHours) {
        throw (
            "Kaggle GPU reserve gate failed: remaining=$remainingGpuHours, " +
            "required=$requiredGpuHours"
        )
    }
    $native.Output | Set-Content -LiteralPath $quotaPath -Encoding utf8

    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Pre-launch kernel state lookup failed: $($native.Output)" }
    $beforeState = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$beforeState.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid next temporal-localization kernel version" }

    $native = Invoke-NativeOutput { & python $candidateBuilder --runtime-root $runtimeRoot }
    if ($native.ExitCode -ne 0) { throw "Temporal-localization candidate build failed: $($native.Output)" }
    $metadata = Get-Content -Raw -LiteralPath $candidateMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or $metadata.machine_shape -ne "NvidiaTeslaT4" -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.competition_sources -notcontains $competitionRef -or
        $metadata.kernel_sources.Count -ne 0
    ) {
        throw "Built temporal-localization kernel metadata is invalid"
    }
    $notebookText = Get-Content -Raw -LiteralPath $candidateNotebook
    foreach ($requiredPattern in @(
        "Exactly two T4 GPUs are required", "cuda:0", "cuda:1", "ThreadPoolExecutor",
        '"declared_budget_seconds": 39600', "Timer(38400, _biohub_budget_expired)"
    )) {
        if ($notebookText -notmatch [regex]::Escape($requiredPattern)) {
            throw "Built temporal-localization notebook lost a required contract: $requiredPattern"
        }
    }
    if ($notebookText -match "kaggle competitions submit") {
        throw "Built temporal-localization notebook contains a submission command"
    }

    $launched = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels push -p $candidateDir }
        Write-ControllerLog "kernel_push expected_version=$expectedVersion output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)successfully pushed") {
            $launched = $true
            break
        }
        if ($native.Output -notmatch "(?i)quota|accelerator|maximum batch GPU session count|temporarily") {
            throw "Temporal-localization kernel push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for a Kaggle dual-T4 launch slot" }

    Start-Sleep -Seconds 10
    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Post-launch kernel state lookup failed: $($native.Output)" }
    $state = $native.Output | ConvertFrom-Json
    if (
        $state.present -ne $true -or [int]$state.current_version_number -ne $expectedVersion -or
        $state.is_private -ne $true -or $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or $state.enable_internet -ne $false -or
        $state.dataset_sources -notcontains $runtimeRef -or
        $state.competition_sources -notcontains $competitionRef
    ) {
        throw "Remote temporal-localization kernel state is invalid: $($native.Output)"
    }

    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-ControllerLog "candidate_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)COMPLETE") {
            $complete = $true
            break
        }
        if ($native.Output -match "(?i)ERROR|CANCEL") {
            throw "Temporal-localization candidate failed: $($native.Output)"
        }
        Start-Sleep -Seconds $KernelPollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for temporal-localization candidate completion" }

    $downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/kernel-outputs/biohub-ema-temporal-localization-v1-version$expectedVersion"
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Refusing to overwrite temporal-localization kernel output: $downloadRoot"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput {
        & kaggle kernels output "$kernelRef/$expectedVersion" -p $downloadRoot --force `
            --file-pattern $requiredOutputPattern --page-size 200
    }
    if ($native.ExitCode -ne 0) { throw "Candidate output download failed: $($native.Output)" }

    $native = Invoke-NativeOutput {
        & $evaluationPython $verifier --output-root $downloadRoot `
            --baseline-validator $baselineValidator --runtime-manifest $runtimeManifest `
            --report $promotionPath
    }
    if ($native.ExitCode -ne 0) {
        Write-ControllerTerminal "candidate_rejected" @{
            kernel_version = $expectedVersion
            download_root = $downloadRoot
            verification_error = $native.Output
            runtime_manifest_sha256 = $runtimeManifestHash
            competition_submission_performed = $false
        }
        exit 0
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne "eligible_for_submission" -or $promotion.authorized_for_submission -ne $true) {
        throw "Temporal-localization promotion report is invalid"
    }

    $native = Invoke-NativeOutput {
        & $evaluationPython $submitter --promotion $promotionPath --receipt $receiptPath `
            --kernel-ref $kernelRef --kernel-version $expectedVersion --execute
    }
    if ($native.ExitCode -ne 0) { throw "Promoted candidate submission failed: $($native.Output)" }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne "submitted" -or $receipt.competition_submission_performed -ne $true) {
        throw "Temporal-localization submission receipt is invalid"
    }
    Write-ControllerTerminal "submitted" @{
        kernel_version = $expectedVersion
        dataset_operation = $datasetOperation
        runtime_manifest_sha256 = $runtimeManifestHash
        submission_sha256 = $promotion.submission_sha256
        proxy_gain = $promotion.proxy_gain
        missed_gt_node_gain = $promotion.missed_gt_node_gain
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        remaining_gpu_hours_before = $remainingGpuHours
        minimum_gpu_reserve_hours = $MinimumGpuReserveHours
        competition_submission_performed = $true
    }
    exit 0
}
catch {
    Write-ControllerTerminal "failed" @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-ControllerLog "failed error=$($_.Exception.Message)"
    exit 1
}
