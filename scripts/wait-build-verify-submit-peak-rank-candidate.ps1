param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 60.0,
    [int]$PollSeconds = 120,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid peak-ranking candidate wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$runtimeRef = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
$kernelRef = "indarkarhana/biohub-peak-rank-tracking-candidate-v1"
$competitionRef = "biohub-cell-tracking-during-development"
$validationTerminal = Join-Path $RepositoryRoot ".biohub/automation/peak-rank-validation-controller-v1.json"
$runtimeRoot = Join-Path $RepositoryRoot ".biohub/staging/biohub-peak-rank-validation-runtime-v1"
$runtimeManifest = Join-Path $runtimeRoot "SOURCE_MANIFEST.json"
$promoter = Join-Path $RepositoryRoot "scripts/promote-peak-rank-validation-runtime.py"
$builder = Join-Path $RepositoryRoot "scripts/build-peak-rank-submission-candidate.py"
$verifier = Join-Path $RepositoryRoot "scripts/verify-peak-rank-submission-candidate.py"
$submitter = Join-Path $RepositoryRoot "scripts/submit-peak-rank-candidate.py"
$kernelState = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$candidateRoot = Join-Path $RepositoryRoot "kaggle/biohub-peak-rank-tracking-candidate-v1"
$metadataPath = Join-Path $candidateRoot "kernel-metadata.json"
$notebookPath = Join-Path $candidateRoot "biohub-peak-rank-tracking-candidate-v1.ipynb"
$baselineValidator = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "peak-rank-candidate-controller-v1.json"
$logPath = Join-Path $automationRoot "peak-rank-candidate-controller-v1.log"
$promotionPath = Join-Path $automationRoot "peak-rank-candidate-promotion-v1.json"
$receiptPath = Join-Path $automationRoot "peak-rank-candidate-submission-receipt-v1.json"
$requiredOutputPattern = '(^|.*/)(candidate_evidence\.json|run_stats\.csv|submission\.csv|validator_results\.csv|launcher_terminal\.json|worker-[01]\.json)$'

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "peak-rank-candidate-controller-v1"
        status = $Status
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        target_public_score = 0.945
        public_leaderboard_used_for_selection = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Invoke-NativeOutput([scriptblock]$Command) {
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $lines = & $Command 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    [pscustomobject]@{ ExitCode = $exitCode; Output = ($lines -join "`n") }
}

New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
foreach ($required in @(
    $promoter, $builder, $verifier, $submitter, $kernelState,
    $evaluationPython, $baselineValidator
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required peak-ranking candidate input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required peak-ranking candidate command is missing: $command"
    }
}
if ($ValidateOnly) {
    $errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile(
        $PSCommandPath, [ref]$null, [ref]$errors
    )
    if ($errors.Count -ne 0) { throw "Peak-ranking candidate controller syntax is invalid" }
    & $evaluationPython -m py_compile $promoter $builder $verifier $submitter $kernelState
    if ($LASTEXITCODE -ne 0) { throw "Peak-ranking candidate Python preflight failed" }
    @{
        status = "validated"
        validation_terminal = $validationTerminal
        exact_gpu_count = 2
        candidate_submission_requires_external_promotion = $true
    } | ConvertTo-Json
    exit 0
}
foreach ($forbidden in @($terminalPath, $promotionPath, $receiptPath, $candidateRoot)) {
    if (Test-Path -LiteralPath $forbidden) {
        throw "Refusing to reuse peak-ranking candidate state: $forbidden"
    }
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    while (-not (Test-Path -LiteralPath $validationTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Timed out waiting for peak-ranking clean validation"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $validation = Get-Content -Raw -LiteralPath $validationTerminal | ConvertFrom-Json
    if ($validation.status -ne "completed" -or $validation.accepted_for_candidate_integration -ne $true) {
        Write-Terminal "skipped_after_clean_rejection" @{
            validation_status = $validation.status
            selection_passed = $validation.selection_passed
            acceptance_opened = $validation.acceptance_opened
            promotion_passed = $validation.promotion_passed
            runtime_versioned = $false
            kernel_launched = $false
            competition_submission_performed = $false
        }
        exit 0
    }
    if ($validation.competition_submission_performed -ne $false) {
        throw "Peak-ranking validation crossed the submission boundary"
    }
    $native = Invoke-NativeOutput { & $evaluationPython $promoter }
    if ($native.ExitCode -ne 0) { throw "Runtime promotion binding failed: $($native.Output)" }
    $manifest = Get-Content -Raw -LiteralPath $runtimeManifest | ConvertFrom-Json
    if (
        $manifest.clean_validation_promotion_passed -ne $true -or
        $manifest.checkpoint_sha256 -ne $validation.checkpoint_sha256
    ) {
        throw "Promoted runtime differs from clean validation"
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
    $native = Invoke-NativeOutput {
        & kaggle datasets version -p $runtimeRoot -m "Clean-promoted peak-rank detector and production bridge v1"
    }
    if ($native.ExitCode -ne 0) { throw "Promoted runtime version failed: $($native.Output)" }
    Write-Log "runtime_versioned output=$($native.Output)"
    $runtimeReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $runtimeRef --format json }
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)ready|complete") {
            $runtimeReady = $true
            break
        }
        if ($native.Output -match "(?i)error|failed") {
            throw "Promoted runtime processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $runtimeReady) { throw "Timed out waiting for promoted runtime" }

    $native = Invoke-NativeOutput { & $evaluationPython $kernelState --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Candidate kernel state lookup failed: $($native.Output)" }
    $before = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$before.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid next peak-ranking kernel version" }
    $native = Invoke-NativeOutput { & $evaluationPython $builder --runtime-root $runtimeRoot }
    if ($native.ExitCode -ne 0) { throw "Candidate build failed: $($native.Output)" }
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne "NvidiaTeslaT4" -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.competition_sources -notcontains $competitionRef -or
        $metadata.kernel_sources.Count -ne 0
    ) {
        throw "Built peak-ranking candidate metadata is invalid"
    }
    $notebook = Get-Content -Raw -LiteralPath $notebookPath
    foreach ($requiredPattern in @(
        "predict_with_official_linker.py",
        "device_count() != 2",
        "CUDA_VISIBLE_DEVICES",
        "peak_worker_manifests",
        "SEC_EDGE_TTA_ACTIVE",
        "redoctopusk/biohub-948tta2",
        "completed_pending_external_promotion_gate"
    )) {
        if ($notebook -notmatch [regex]::Escape($requiredPattern)) {
            throw "Built peak-ranking candidate lost contract: $requiredPattern"
        }
    }
    if ($notebook -match "kaggle competitions submit") {
        throw "Built candidate notebook contains a submission command"
    }

    $launched = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels push -p $candidateRoot }
        Write-Log "kernel_push expected_version=$expectedVersion output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)successfully pushed") {
            $launched = $true
            break
        }
        if ($native.Output -notmatch "(?i)quota|accelerator|maximum batch GPU session count|temporarily") {
            throw "Peak-ranking candidate push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for a two-GPU Kaggle slot" }
    Start-Sleep -Seconds 15
    $native = Invoke-NativeOutput { & $evaluationPython $kernelState --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Post-launch kernel lookup failed: $($native.Output)" }
    $state = $native.Output | ConvertFrom-Json
    if (
        $state.present -ne $true -or
        [int]$state.current_version_number -ne $expectedVersion -or
        $state.is_private -ne $true -or
        $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or
        $state.enable_internet -ne $false -or
        $state.dataset_sources -notcontains $runtimeRef -or
        $state.competition_sources -notcontains $competitionRef
    ) {
        throw "Remote peak-ranking candidate state is invalid"
    }
    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-Log "candidate_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)COMPLETE") {
            $complete = $true
            break
        }
        if ($native.Output -match "(?i)ERROR|CANCEL") {
            throw "Peak-ranking candidate kernel failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for peak-ranking candidate completion" }
    $downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/kernel-outputs/peak-rank-tracking-candidate-v1-version$expectedVersion"
    if (Test-Path -LiteralPath $downloadRoot) { throw "Refusing to reuse candidate output" }
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
        Write-Terminal "candidate_rejected" @{
            kernel_version = $expectedVersion
            download_root = $downloadRoot
            verification_error = $native.Output
            runtime_manifest_sha256 = $manifestHash
            competition_submission_performed = $false
        }
        exit 0
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne "eligible_for_submission" -or $promotion.authorized_for_submission -ne $true) {
        throw "Peak-ranking promotion report is invalid"
    }
    $native = Invoke-NativeOutput {
        & $evaluationPython $submitter --promotion $promotionPath --receipt $receiptPath `
            --kernel-ref $kernelRef --kernel-version $expectedVersion --execute
    }
    if ($native.ExitCode -ne 0) { throw "Promoted peak-ranking submission failed: $($native.Output)" }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne "submitted" -or $receipt.competition_submission_performed -ne $true) {
        throw "Peak-ranking submission receipt is invalid"
    }
    Write-Terminal "submitted" @{
        kernel_version = $expectedVersion
        download_root = $downloadRoot
        runtime_manifest_sha256 = $manifestHash
        submission_sha256 = $promotion.submission_sha256
        proxy_gain = $promotion.proxy_gain
        adjusted_edge_delta = $promotion.adjusted_edge_delta
        worst_movie_proxy_delta = $promotion.worst_movie_proxy_delta
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        competition_submission_performed = $true
    }
    exit 0
}
catch {
    Write-Terminal "failed" @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-Log "failed error=$($_.Exception.Message)"
    exit 1
}
