param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 60.0,
    [int]$PollSeconds = 120,
    [ValidateSet("v1", "depth-pu-v2", "capacity-pu-v3", "faint-pu-v4", "expanded-real-faint-v7", "expanded-real-local-shape-v9", "expanded-real-blob-v11", "capacity-faint-ensemble-v5", "capacity-faint-confidence-v6", "capacity-faint-expanded-v8", "expanded-local-shape-ensemble-v10", "expanded-blob-ensemble-v12", "logit-ensemble-v4")]
    [string]$Variant = "v1",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid peak-ranking candidate wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$competitionRef = "biohub-cell-tracking-during-development"
$variantConfig = if ($Variant -eq "v1") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
        kernel_ref = "indarkarhana/biohub-peak-rank-tracking-candidate-v1"
        validation_terminal = ".biohub/automation/peak-rank-validation-controller-v1.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-validation-runtime-v1"
        builder = "scripts/build-peak-rank-submission-candidate.py"
        candidate_root = "kaggle/biohub-peak-rank-tracking-candidate-v1"
        notebook_name = "biohub-peak-rank-tracking-candidate-v1.ipynb"
        controller_id = "peak-rank-candidate-controller-v1"
        promotion_name = "peak-rank-candidate-promotion-v1.json"
        receipt_name = "peak-rank-candidate-submission-receipt-v1.json"
        expected_run_id = "peak-rank-tracking-candidate-v1"
        output_slug = "peak-rank-tracking-candidate-v1"
    }
}
elseif ($Variant -eq "depth-pu-v2") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-depth-pu-validation-runtime-v2"
        kernel_ref = "indarkarhana/biohub-peak-rank-depth-pu-tracking-candidate-v2"
        validation_terminal = ".biohub/automation/peak-rank-depth-pu-validation-controller-v2.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-depth-pu-validation-runtime-v2"
        builder = "scripts/build-peak-rank-depth-pu-submission-candidate.py"
        candidate_root = "kaggle/biohub-peak-rank-depth-pu-tracking-candidate-v2"
        notebook_name = "biohub-peak-rank-depth-pu-tracking-candidate-v2.ipynb"
        controller_id = "peak-rank-depth-pu-candidate-controller-v2"
        promotion_name = "peak-rank-depth-pu-candidate-promotion-v2.json"
        receipt_name = "peak-rank-depth-pu-candidate-submission-receipt-v2.json"
        expected_run_id = "peak-rank-depth-pu-tracking-candidate-v2"
        output_slug = "peak-rank-depth-pu-tracking-candidate-v2"
    }
}
elseif ($Variant -eq "capacity-pu-v3") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-pu-validation-runtime-v3"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-pu-tracking-candidate-v3"
        validation_terminal = ".biohub/automation/peak-rank-capacity-pu-validation-controller-v3.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-pu-validation-runtime-v3"
        builder = "scripts/build-peak-rank-capacity-pu-submission-candidate.py"
        candidate_root = "kaggle/biohub-peak-rank-capacity-pu-tracking-candidate-v3"
        notebook_name = "biohub-peak-rank-capacity-pu-tracking-candidate-v3.ipynb"
        controller_id = "peak-rank-capacity-pu-candidate-controller-v3"
        promotion_name = "peak-rank-capacity-pu-candidate-promotion-v3.json"
        receipt_name = "peak-rank-capacity-pu-candidate-submission-receipt-v3.json"
        expected_run_id = "peak-rank-capacity-pu-tracking-candidate-v3"
        output_slug = "peak-rank-capacity-pu-tracking-candidate-v3"
    }
}
elseif ($Variant -eq "faint-pu-v4") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-faint-pu-validation-runtime-v4"
        kernel_ref = "indarkarhana/biohub-peak-rank-faint-pu-tracking-candidate-v4"
        validation_terminal = ".biohub/automation/peak-rank-faint-pu-validation-controller-v4.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-faint-pu-validation-runtime-v4"
        builder = "scripts/build-peak-rank-faint-pu-submission-candidate.py"
        candidate_root = "kaggle/biohub-peak-rank-faint-pu-tracking-candidate-v4"
        notebook_name = "biohub-peak-rank-faint-pu-tracking-candidate-v4.ipynb"
        controller_id = "peak-rank-faint-pu-candidate-controller-v4"
        promotion_name = "peak-rank-faint-pu-candidate-promotion-v4.json"
        receipt_name = "peak-rank-faint-pu-candidate-submission-receipt-v4.json"
        expected_run_id = "peak-rank-faint-pu-tracking-candidate-v4"
        output_slug = "peak-rank-faint-pu-tracking-candidate-v4"
    }
}
elseif ($Variant -eq "expanded-real-faint-v7") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-faint-validation-runtime-v7"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-faint-tracking-candidate-v7"
        validation_terminal = ".biohub/automation/peak-rank-expanded-real-faint-validation-controller-v7.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-faint-validation-runtime-v7"
        builder = "scripts/build-peak-rank-expanded-real-faint-submission-candidate-v7.py"
        candidate_root = "kaggle/biohub-peak-rank-expanded-real-faint-tracking-candidate-v7"
        notebook_name = "biohub-peak-rank-expanded-real-faint-tracking-candidate-v7.ipynb"
        controller_id = "peak-rank-expanded-real-faint-candidate-controller-v7"
        promotion_name = "peak-rank-expanded-real-faint-candidate-promotion-v7.json"
        receipt_name = "peak-rank-expanded-real-faint-candidate-submission-receipt-v7.json"
        expected_run_id = "peak-rank-expanded-real-faint-tracking-candidate-v7"
        output_slug = "peak-rank-expanded-real-faint-tracking-candidate-v7"
    }
}
elseif ($Variant -eq "expanded-real-local-shape-v9") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-local-shape-tracking-candidate-v9"
        validation_terminal = ".biohub/automation/peak-rank-expanded-real-local-shape-validation-controller-v9.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
        builder = "scripts/build-peak-rank-expanded-real-local-shape-submission-candidate-v9.py"
        candidate_root = "kaggle/biohub-peak-rank-expanded-real-local-shape-tracking-candidate-v9"
        notebook_name = "biohub-peak-rank-expanded-real-local-shape-tracking-candidate-v9.ipynb"
        controller_id = "peak-rank-expanded-real-local-shape-candidate-controller-v9"
        promotion_name = "peak-rank-expanded-real-local-shape-candidate-promotion-v9.json"
        receipt_name = "peak-rank-expanded-real-local-shape-candidate-submission-receipt-v9.json"
        expected_run_id = "peak-rank-expanded-real-local-shape-tracking-candidate-v9"
        output_slug = "peak-rank-expanded-real-local-shape-tracking-candidate-v9"
    }
}
elseif ($Variant -eq "expanded-real-blob-v11") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-blob-validation-runtime-v11"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-blob-tracking-candidate-v11"
        validation_terminal = ".biohub/automation/peak-rank-expanded-real-blob-validation-controller-v11.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-blob-validation-runtime-v11"
        builder = "scripts/build-peak-rank-expanded-real-blob-submission-candidate-v11.py"
        candidate_root = "kaggle/biohub-peak-rank-expanded-real-blob-tracking-candidate-v11"
        notebook_name = "biohub-peak-rank-expanded-real-blob-tracking-candidate-v11.ipynb"
        controller_id = "peak-rank-expanded-real-blob-candidate-controller-v11"
        promotion_name = "peak-rank-expanded-real-blob-candidate-promotion-v11.json"
        receipt_name = "peak-rank-expanded-real-blob-candidate-submission-receipt-v11.json"
        expected_run_id = "peak-rank-expanded-real-blob-tracking-candidate-v11"
        output_slug = "peak-rank-expanded-real-blob-tracking-candidate-v11"
    }
}
elseif ($Variant -eq "capacity-faint-ensemble-v5") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-faint-ensemble-validation-runtime-v5"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-faint-ensemble-tracking-candidate-v5"
        validation_terminal = ".biohub/automation/peak-rank-capacity-faint-ensemble-validation-controller-v5.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-faint-ensemble-validation-runtime-v5"
        builder = "scripts/build-peak-rank-capacity-faint-ensemble-submission-candidate-v5.py"
        candidate_root = "kaggle/biohub-peak-rank-capacity-faint-ensemble-tracking-candidate-v5"
        notebook_name = "biohub-peak-rank-capacity-faint-ensemble-tracking-candidate-v5.ipynb"
        controller_id = "peak-rank-capacity-faint-ensemble-candidate-controller-v5"
        promotion_name = "peak-rank-capacity-faint-ensemble-candidate-promotion-v5.json"
        receipt_name = "peak-rank-capacity-faint-ensemble-candidate-submission-receipt-v5.json"
        expected_run_id = "peak-rank-capacity-faint-ensemble-tracking-candidate-v5"
        output_slug = "peak-rank-capacity-faint-ensemble-tracking-candidate-v5"
    }
}
elseif ($Variant -eq "capacity-faint-confidence-v6") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-faint-confidence-tracking-candidate-v6"
        validation_terminal = ".biohub/automation/peak-rank-capacity-faint-confidence-validation-controller-v6.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
        builder = "scripts/build-peak-rank-capacity-faint-confidence-ensemble-submission-candidate-v6.py"
        candidate_root = "kaggle/biohub-peak-rank-capacity-faint-confidence-tracking-candidate-v6"
        notebook_name = "biohub-peak-rank-capacity-faint-confidence-tracking-candidate-v6.ipynb"
        controller_id = "peak-rank-capacity-faint-confidence-candidate-controller-v6"
        promotion_name = "peak-rank-capacity-faint-confidence-candidate-promotion-v6.json"
        receipt_name = "peak-rank-capacity-faint-confidence-candidate-submission-receipt-v6.json"
        expected_run_id = "peak-rank-capacity-faint-confidence-tracking-candidate-v6"
        output_slug = "peak-rank-capacity-faint-confidence-tracking-candidate-v6"
    }
}
elseif ($Variant -eq "capacity-faint-expanded-v8") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
        kernel_ref = "indarkarhana/biohub-peak-rank-cfe-ensemble-tracking-candidate-v8"
        validation_terminal = ".biohub/automation/peak-rank-cfe-ensemble-validation-controller-v8.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
        builder = "scripts/build-peak-rank-cfe-ensemble-submission-candidate-v8.py"
        candidate_root = "kaggle/biohub-peak-rank-cfe-ensemble-tracking-candidate-v8"
        notebook_name = "biohub-peak-rank-cfe-ensemble-tracking-candidate-v8.ipynb"
        controller_id = "peak-rank-cfe-ensemble-candidate-controller-v8"
        promotion_name = "peak-rank-cfe-ensemble-candidate-promotion-v8.json"
        receipt_name = "peak-rank-cfe-ensemble-candidate-submission-receipt-v8.json"
        expected_run_id = "peak-rank-cfe-ensemble-tracking-candidate-v8"
        output_slug = "peak-rank-cfe-ensemble-tracking-candidate-v8"
    }
}
elseif ($Variant -eq "expanded-local-shape-ensemble-v10") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
        validation_terminal = ".biohub/automation/peak-rank-expanded-local-shape-ensemble-validation-controller-v10.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
        builder = "scripts/build-peak-rank-expanded-local-shape-ensemble-submission-candidate-v10.py"
        candidate_root = "kaggle/biohub-peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
        notebook_name = "biohub-peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10.ipynb"
        controller_id = "peak-rank-expanded-local-shape-ensemble-candidate-controller-v10"
        promotion_name = "peak-rank-expanded-local-shape-ensemble-candidate-promotion-v10.json"
        receipt_name = "peak-rank-expanded-local-shape-ensemble-candidate-submission-receipt-v10.json"
        expected_run_id = "peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
        output_slug = "peak-rank-expanded-local-shape-ensemble-tracking-candidate-v10"
    }
}
elseif ($Variant -eq "expanded-blob-ensemble-v12") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
        validation_terminal = ".biohub/automation/peak-rank-expanded-blob-ensemble-validation-controller-v12.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
        builder = "scripts/build-peak-rank-expanded-blob-ensemble-submission-candidate-v12.py"
        candidate_root = "kaggle/biohub-peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
        notebook_name = "biohub-peak-rank-expanded-blob-ensemble-tracking-candidate-v12.ipynb"
        controller_id = "peak-rank-expanded-blob-ensemble-candidate-controller-v12"
        promotion_name = "peak-rank-expanded-blob-ensemble-candidate-promotion-v12.json"
        receipt_name = "peak-rank-expanded-blob-ensemble-candidate-submission-receipt-v12.json"
        expected_run_id = "peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
        output_slug = "peak-rank-expanded-blob-ensemble-tracking-candidate-v12"
    }
}
else {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-logit-ensemble-validation-runtime-v4"
        kernel_ref = "indarkarhana/biohub-peak-rank-logit-ensemble-tracking-candidate-v4"
        validation_terminal = ".biohub/automation/peak-rank-logit-ensemble-validation-controller-v4.json"
        runtime_root = ".biohub/staging/biohub-peak-rank-logit-ensemble-validation-runtime-v4"
        builder = "scripts/build-peak-rank-logit-ensemble-submission-candidate.py"
        candidate_root = "kaggle/biohub-peak-rank-logit-ensemble-tracking-candidate-v4"
        notebook_name = "biohub-peak-rank-logit-ensemble-tracking-candidate-v4.ipynb"
        controller_id = "peak-rank-logit-ensemble-candidate-controller-v4"
        promotion_name = "peak-rank-logit-ensemble-candidate-promotion-v4.json"
        receipt_name = "peak-rank-logit-ensemble-candidate-submission-receipt-v4.json"
        expected_run_id = "peak-rank-logit-ensemble-tracking-candidate-v4"
        output_slug = "peak-rank-logit-ensemble-tracking-candidate-v4"
    }
}
$runtimeRef = $variantConfig.runtime_ref
$kernelRef = $variantConfig.kernel_ref
$validationTerminal = Join-Path $RepositoryRoot $variantConfig.validation_terminal
$runtimeRoot = Join-Path $RepositoryRoot $variantConfig.runtime_root
$runtimeManifest = Join-Path $runtimeRoot "SOURCE_MANIFEST.json"
$promoter = Join-Path $RepositoryRoot "scripts/promote-peak-rank-validation-runtime.py"
$builder = Join-Path $RepositoryRoot $variantConfig.builder
$verifier = Join-Path $RepositoryRoot "scripts/verify-peak-rank-submission-candidate.py"
$exactScorer = Join-Path $RepositoryRoot "research/peak_rank_detection/score_official_candidate.py"
$submitter = Join-Path $RepositoryRoot "scripts/submit-peak-rank-candidate.py"
$kernelState = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$candidateRoot = Join-Path $RepositoryRoot $variantConfig.candidate_root
$metadataPath = Join-Path $candidateRoot "kernel-metadata.json"
$notebookPath = Join-Path $candidateRoot $variantConfig.notebook_name
$baselineValidator = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv"
$controlGraphs = Join-Path $RepositoryRoot ".biohub/cache/processed-public-control-geffs-v1"
$truthGraphs = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$scorerLock = Join-Path $RepositoryRoot "config/official-scorer.lock.json"
$organizerCheckout = Join-Path $RepositoryRoot ".biohub/vendor/kaggle-cell-tracking-competition"
$tracksdataCheckout = Join-Path $RepositoryRoot ".biohub/vendor/tracksdata"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot ($variantConfig.controller_id + ".json")
$logPath = Join-Path $automationRoot ($variantConfig.controller_id + ".log")
$promotionPath = Join-Path $automationRoot $variantConfig.promotion_name
$receiptPath = Join-Path $automationRoot $variantConfig.receipt_name
$requiredOutputPattern = '(^|.*/)(candidate_evidence\.json|official_validator_candidate\.csv|run_stats\.csv|submission\.csv|validator_results\.csv|launcher_terminal\.json|worker-[01]\.json)$'

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = $variantConfig.controller_id
        variant = $Variant
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
    $promoter, $builder, $verifier, $exactScorer, $submitter, $kernelState,
    $evaluationPython, $baselineValidator, $scorerLock
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required peak-ranking candidate input is missing: $required"
    }
}
foreach ($requiredDirectory in @(
    $controlGraphs, $truthGraphs, $organizerCheckout, $tracksdataCheckout
)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "Required peak-ranking exact-scoring directory is missing: $requiredDirectory"
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
    & $evaluationPython -m py_compile $promoter $builder $verifier $exactScorer $submitter $kernelState
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
    $native = Invoke-NativeOutput {
        & $evaluationPython $promoter --controller $validationTerminal --runtime $runtimeRoot
    }
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
        & kaggle datasets version -p $runtimeRoot -m "Clean-promoted peak-rank detector and production bridge $Variant"
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
        "selected_peak_tta_mode",
        "max_projected_worker_seconds",
        "SEC_EDGE_TTA_ACTIVE",
        "redoctopusk/biohub-948tta2",
        "completed_pending_external_promotion_gate",
        "official_validator_candidate.csv",
        "pending_external_patched_official_scoring"
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
    $downloadRoot = Join-Path $RepositoryRoot (".biohub/cache/kernel-outputs/" + $variantConfig.output_slug + "-version" + $expectedVersion)
    if (Test-Path -LiteralPath $downloadRoot) { throw "Refusing to reuse candidate output" }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput {
        & kaggle kernels output "$kernelRef/$expectedVersion" -p $downloadRoot --force `
            --file-pattern $requiredOutputPattern --page-size 200
    }
    if ($native.ExitCode -ne 0) { throw "Candidate output download failed: $($native.Output)" }
    $officialValidatorMatches = @(
        Get-ChildItem -LiteralPath $downloadRoot -Recurse -File -Filter "official_validator_candidate.csv"
    )
    if ($officialValidatorMatches.Count -ne 1) {
        throw "Expected exactly one materialized official validator CSV"
    }
    $officialMetricResult = Join-Path $downloadRoot "official_metric_result.json"
    $native = Invoke-NativeOutput {
        & $evaluationPython $exactScorer `
            --candidate-validator $officialValidatorMatches[0].FullName `
            --control-dir $controlGraphs --truth-dir $truthGraphs `
            --scorer-lock $scorerLock --organizer-checkout $organizerCheckout `
            --tracksdata-checkout $tracksdataCheckout --output $officialMetricResult
    }
    if ($native.ExitCode -ne 0) {
        Write-Terminal "candidate_rejected_by_patched_official_metric" @{
            kernel_version = $expectedVersion
            download_root = $downloadRoot
            verification_error = $native.Output
            runtime_manifest_sha256 = $manifestHash
            competition_submission_performed = $false
        }
        exit 0
    }
    $native = Invoke-NativeOutput {
        & $evaluationPython $verifier --output-root $downloadRoot `
            --baseline-validator $baselineValidator --runtime-manifest $runtimeManifest `
            --official-metric-result $officialMetricResult `
            --expected-run-id $variantConfig.expected_run_id --report $promotionPath
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
            --kernel-ref $kernelRef --kernel-version $expectedVersion `
            --expected-run-id $variantConfig.expected_run_id --execute
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
        exact_control_score = $promotion.exact_control_score
        exact_candidate_score = $promotion.exact_candidate_score
        exact_score_gain = $promotion.exact_score_gain
        adjusted_edge_delta = $promotion.adjusted_edge_delta
        worst_movie_score_delta = $promotion.worst_movie_score_delta
        official_scorer_lock_sha256 = $promotion.official_scorer_lock_sha256
        official_metric_result_sha256 = $promotion.official_metric_result_sha256
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
