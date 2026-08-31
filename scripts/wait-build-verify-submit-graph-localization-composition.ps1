param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 168.0,
    [int]$PollSeconds = 60,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid graph-localization composition wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$kernelRef = "indarkarhana/biohub-ema-graph-localization-v1"
$competitionRef = "biohub-cell-tracking-during-development"
$graphRuntimeRef = "indarkarhana/biohub-graph-context-consensus-division-v1"
$localizationRuntimeRef = "indarkarhana/biohub-temporal-localization-consensus-v1"
$graphDevelopmentTerminal = Join-Path $RepositoryRoot ".biohub/cache/graph-context-division-recovery-development-v1/development-terminal.json"
$graphLaunchTerminal = Join-Path $RepositoryRoot ".biohub/automation/graph-context-recovery-consensus-candidate-launch-v1.json"
$localizationControllerTerminal = Join-Path $RepositoryRoot ".biohub/automation/temporal-localization-candidate-controller-v1.json"
$localizationPromotion = Join-Path $RepositoryRoot ".biohub/automation/temporal-localization-candidate-promotion.json"
$localizationRuntimeRoot = Join-Path $RepositoryRoot ".biohub/cache/runtime-datasets/biohub-temporal-localization-consensus-v1"
$localizationRuntimeManifest = Join-Path $localizationRuntimeRoot "TEMPORAL_LOCALIZATION_CONSENSUS_MANIFEST.json"
$candidateDir = Join-Path $RepositoryRoot "kaggle/biohub-ema-graph-localization-v1"
$candidateNotebook = Join-Path $candidateDir "biohub-ema-graph-localization-v1.ipynb"
$candidateMetadata = Join-Path $candidateDir "kernel-metadata.json"
$builder = Join-Path $RepositoryRoot "scripts/build-graph-localization-composition-candidate.py"
$verifier = Join-Path $RepositoryRoot "scripts/verify-graph-localization-composition-candidate.py"
$submitter = Join-Path $RepositoryRoot "scripts/submit-graph-localization-composition-candidate.py"
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$baselineValidator = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "graph-localization-composition-controller-v1.json"
$logPath = Join-Path $automationRoot "graph-localization-composition-controller-v1.log"
$promotionPath = Join-Path $automationRoot "graph-localization-composition-promotion-v1.json"
$receiptPath = Join-Path $automationRoot "graph-localization-composition-submission-receipt-v1.json"
$requiredOutputPattern = "^(candidate_evidence\.json|run_stats\.csv|submission\.csv|validator_results\.csv|watchdog-terminal\.json)$"
New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null

function Write-ControllerLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Write-ControllerTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "graph-localization-composition-controller-v1"
        status = $Status
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        target_public_score = 0.945
        expected_gpu_count = 2
        machine_shape = "NvidiaTeslaT4"
        component_order = @("graph_context_division", "temporal_localization")
        independent_component_promotion_required = $true
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Invoke-NativeOutput([scriptblock]$Command) {
    $savedErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $lines = & $Command 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorPreference
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($lines -join "`n")
    }
}

foreach ($required in @(
    $builder, $verifier, $submitter, $kernelStateScript,
    $evaluationPython, $baselineValidator
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required composition controller input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required composition controller command is missing: $command"
    }
}

if ($ValidateOnly) {
    & python -m py_compile $builder $verifier $submitter
    if ($LASTEXITCODE -ne 0) { throw "Composition Python source validation failed" }
    @{
        status = "validated"
        run_id = "graph-localization-composition-controller-v1"
        independent_component_promotion_required = $true
        candidate_submission_requires_composition_gain = $true
    } | ConvertTo-Json
    exit 0
}

foreach ($forbidden in @($terminalPath, $candidateDir, $promotionPath, $receiptPath)) {
    if (Test-Path -LiteralPath $forbidden) {
        throw "Refusing to reuse composition controller state: $forbidden"
    }
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    $graphPromotion = $null
    $graphKernelVersion = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $graphLaunchTerminal -PathType Leaf) {
            $graphLaunch = Get-Content -Raw -LiteralPath $graphLaunchTerminal | ConvertFrom-Json
            if ($graphLaunch.status -ne "launched") {
                Write-ControllerTerminal "skipped_after_graph_component_rejection" @{
                    graph_launch_status = $graphLaunch.status
                }
                exit 0
            }
            $graphKernelVersion = [int]$graphLaunch.kernel_version
            if ($graphKernelVersion -lt 1) { throw "Graph component kernel version is invalid" }
            $candidateGraphPromotion = Join-Path $automationRoot "graph-context-recovery-consensus-candidate-promotion-version$graphKernelVersion.json"
            $graphControllerTerminal = Join-Path $automationRoot "graph-context-recovery-consensus-candidate-controller-version$graphKernelVersion.json"
            if (Test-Path -LiteralPath $graphControllerTerminal -PathType Leaf) {
                $graphController = Get-Content -Raw -LiteralPath $graphControllerTerminal | ConvertFrom-Json
                if ($graphController.status -eq "candidate_rejected" -or $graphController.status -eq "failed") {
                    Write-ControllerTerminal "skipped_after_graph_component_rejection" @{
                        graph_controller_status = $graphController.status
                    }
                    exit 0
                }
            }
            if (Test-Path -LiteralPath $candidateGraphPromotion -PathType Leaf) {
                $graphPromotion = $candidateGraphPromotion
            }
        }
        if (Test-Path -LiteralPath $localizationControllerTerminal -PathType Leaf) {
            $localizationController = Get-Content -Raw -LiteralPath $localizationControllerTerminal | ConvertFrom-Json
            if ($localizationController.status -eq "candidate_rejected" -or $localizationController.status -eq "failed" -or $localizationController.status -eq "skipped_after_scientific_rejection") {
                Write-ControllerTerminal "skipped_after_localization_component_rejection" @{
                    localization_controller_status = $localizationController.status
                }
                exit 0
            }
        }
        if ($null -ne $graphPromotion -and (Test-Path -LiteralPath $localizationPromotion -PathType Leaf)) {
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if ($null -eq $graphPromotion -or -not (Test-Path -LiteralPath $localizationPromotion -PathType Leaf)) {
        throw "Timed out waiting for two independent component promotions"
    }

    $graphPromotionPayload = Get-Content -Raw -LiteralPath $graphPromotion | ConvertFrom-Json
    $localizationPromotionPayload = Get-Content -Raw -LiteralPath $localizationPromotion | ConvertFrom-Json
    foreach ($row in @($graphPromotionPayload, $localizationPromotionPayload)) {
        if ($row.status -ne "eligible_for_submission" -or $row.authorized_for_submission -ne $true) {
            throw "A standalone component promotion is invalid"
        }
    }

    if (-not (Test-Path -LiteralPath $graphDevelopmentTerminal -PathType Leaf)) {
        throw "Graph development terminal is missing after component promotion"
    }
    $graphDevelopment = Get-Content -Raw -LiteralPath $graphDevelopmentTerminal | ConvertFrom-Json
    if ($graphDevelopment.status -ne "runtime_packaged" -or $graphDevelopment.runtime_created -ne $true) {
        throw "Graph runtime was not scientifically packaged"
    }
    $graphRuntimeRoot = [string]$graphDevelopment.runtime_root
    $graphRuntimeManifest = [string]$graphDevelopment.runtime_manifest
    foreach ($required in @(
        $graphRuntimeRoot, $graphRuntimeManifest,
        $localizationRuntimeRoot, $localizationRuntimeManifest
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Promoted composition runtime input is missing: $required"
        }
    }
    $graphRuntimeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $graphRuntimeManifest).Hash.ToLowerInvariant()
    $localizationRuntimeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localizationRuntimeManifest).Hash.ToLowerInvariant()
    if (
        $graphRuntimeHash -ne [string]$graphPromotionPayload.runtime_manifest_sha256 -or
        $localizationRuntimeHash -ne [string]$localizationPromotionPayload.runtime_manifest_sha256
    ) {
        throw "A runtime manifest changed after standalone promotion"
    }

    foreach ($runtime in @(
        @{ Root = $graphRuntimeRoot; Ref = $graphRuntimeRef },
        @{ Root = $localizationRuntimeRoot; Ref = $localizationRuntimeRef }
    )) {
        $metadata = Get-Content -Raw -LiteralPath (Join-Path $runtime.Root "dataset-metadata.json") | ConvertFrom-Json
        if ($metadata.id -ne $runtime.Ref -or $metadata.isPrivate -ne $true) {
            throw "A composition runtime dataset is not private and hash-bound"
        }
    }

    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Composition kernel state lookup failed: $($native.Output)" }
    $beforeState = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$beforeState.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid next composition kernel version" }

    $native = Invoke-NativeOutput {
        & $evaluationPython $builder --graph-root $graphRuntimeRoot `
            --localization-root $localizationRuntimeRoot
    }
    if ($native.ExitCode -ne 0) { throw "Composition build failed: $($native.Output)" }
    $metadata = Get-Content -Raw -LiteralPath $candidateMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne "NvidiaTeslaT4" -or
        $metadata.dataset_sources -notcontains $graphRuntimeRef -or
        $metadata.dataset_sources -notcontains $localizationRuntimeRef -or
        $metadata.competition_sources -notcontains $competitionRef -or
        $metadata.kernel_sources.Count -ne 0
    ) {
        throw "Built composition metadata is invalid"
    }
    $notebookText = Get-Content -Raw -LiteralPath $candidateNotebook
    foreach ($requiredPattern in @(
        "Exactly two T4 GPUs are required",
        "_apply_graph_context_consensus",
        "_apply_temporal_localization",
        "independent_component_promotion_required"
    )) {
        if ($notebookText -notmatch [regex]::Escape($requiredPattern)) {
            throw "Built composition lost a required contract: $requiredPattern"
        }
    }
    if ($notebookText -match "kaggle competitions submit") {
        throw "Built composition notebook contains a submission command"
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
            throw "Composition kernel push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for a Kaggle composition GPU slot" }

    Start-Sleep -Seconds 10
    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Post-launch composition state lookup failed: $($native.Output)" }
    $state = $native.Output | ConvertFrom-Json
    if (
        $state.present -ne $true -or
        [int]$state.current_version_number -ne $expectedVersion -or
        $state.is_private -ne $true -or
        $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or
        $state.enable_internet -ne $false -or
        $state.dataset_sources -notcontains $graphRuntimeRef -or
        $state.dataset_sources -notcontains $localizationRuntimeRef -or
        $state.competition_sources -notcontains $competitionRef
    ) {
        throw "Remote composition kernel state is invalid: $($native.Output)"
    }

    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-ControllerLog "kernel_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)COMPLETE") {
            $complete = $true
            break
        }
        if ($native.Output -match "(?i)ERROR|CANCEL") {
            throw "Composition kernel failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for composition completion" }

    $downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/kernel-outputs/biohub-ema-graph-localization-v1-version$expectedVersion"
    if (Test-Path -LiteralPath $downloadRoot) { throw "Refusing to overwrite composition output" }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $versionedKernelRef = "$kernelRef/$expectedVersion"
    $native = Invoke-NativeOutput {
        & kaggle kernels output $versionedKernelRef -p $downloadRoot --force `
            --file-pattern $requiredOutputPattern --page-size 200
    }
    if ($native.ExitCode -ne 0) { throw "Composition output download failed: $($native.Output)" }

    $native = Invoke-NativeOutput {
        & $evaluationPython $verifier `
            --output-root $downloadRoot `
            --baseline-validator $baselineValidator `
            --graph-runtime-manifest $graphRuntimeManifest `
            --localization-runtime-manifest $localizationRuntimeManifest `
            --graph-promotion-report $graphPromotion `
            --localization-promotion-report $localizationPromotion `
            --report $promotionPath
    }
    if ($native.ExitCode -ne 0) {
        Write-ControllerTerminal "composition_rejected" @{
            kernel_version = $expectedVersion
            download_root = $downloadRoot
            verification_error = $native.Output
        }
        exit 0
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne "eligible_for_submission" -or $promotion.authorized_for_submission -ne $true) {
        throw "Composition promotion report is invalid"
    }
    $native = Invoke-NativeOutput {
        & $evaluationPython $submitter --promotion $promotionPath --receipt $receiptPath `
            --kernel-ref $kernelRef --kernel-version $expectedVersion --execute
    }
    if ($native.ExitCode -ne 0) { throw "Promoted composition submission failed: $($native.Output)" }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne "submitted" -or $receipt.competition_submission_performed -ne $true) {
        throw "Composition submission receipt is invalid"
    }
    Write-ControllerTerminal "submitted" @{
        kernel_version = $expectedVersion
        graph_component_kernel_version = $graphKernelVersion
        graph_runtime_manifest_sha256 = $graphRuntimeHash
        localization_runtime_manifest_sha256 = $localizationRuntimeHash
        submission_sha256 = $promotion.submission_sha256
        proxy_gain = $promotion.proxy_gain
        composition_gain_over_best_component = $promotion.composition_gain_over_best_component
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
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

