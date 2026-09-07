param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 240.0,
    [int]$PollSeconds = 300,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid V30 composition wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$competitionRef = "biohub-cell-tracking-during-development"
$kernelRef = "indarkarhana/biohub-peak-rank-graph-context-composition-v30"
$peakRuntimeRef = "indarkarhana/biohub-peak-rank-xl-hard-mined-temporal-snr-pair-validation-runtime-v28"
$graphRuntimeRef = "indarkarhana/biohub-graph-context-consensus-division-v1"
$peakRuntimeRoot = Join-Path $RepositoryRoot ".biohub/staging/biohub-peak-rank-xl-hard-mined-temporal-snr-pair-validation-runtime-v28"
$peakRuntimeManifest = Join-Path $peakRuntimeRoot "SOURCE_MANIFEST.json"
$peakPromotion = Join-Path $RepositoryRoot ".biohub/automation/peak-rank-xl-hard-mined-temporal-snr-pair-candidate-promotion-v28.json"
$peakController = Join-Path $RepositoryRoot ".biohub/automation/peak-rank-xl-hard-mined-temporal-snr-pair-candidate-controller-v28.json"
$graphDevelopment = Join-Path $RepositoryRoot ".biohub/cache/graph-context-frozen-ensemble-development-v2/development-terminal.json"
$builder = Join-Path $RepositoryRoot "scripts/build-peak-rank-graph-context-composition-v30.py"
$verifier = Join-Path $RepositoryRoot "scripts/verify-peak-rank-graph-context-composition-v30.py"
$exactScorer = Join-Path $RepositoryRoot "research/peak_rank_detection/score_official_candidate.py"
$submitter = Join-Path $RepositoryRoot "scripts/submit-peak-rank-candidate.py"
$kernelState = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$candidateRoot = Join-Path $RepositoryRoot "kaggle/biohub-peak-rank-graph-context-composition-v30"
$candidateNotebook = Join-Path $candidateRoot "biohub-peak-rank-graph-context-composition-v30.ipynb"
$candidateMetadata = Join-Path $candidateRoot "kernel-metadata.json"
$baselineValidator = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv"
$controlGraphs = Join-Path $RepositoryRoot ".biohub/cache/processed-public-control-geffs-v1"
$truthGraphs = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$scorerLock = Join-Path $RepositoryRoot "config/official-scorer.lock.json"
$organizerCheckout = Join-Path $RepositoryRoot ".biohub/vendor/kaggle-cell-tracking-competition"
$tracksdataCheckout = Join-Path $RepositoryRoot ".biohub/vendor/tracksdata"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "peak-rank-graph-context-composition-controller-v30.json"
$logPath = Join-Path $automationRoot "peak-rank-graph-context-composition-controller-v30.log"
$promotionPath = Join-Path $automationRoot "peak-rank-graph-context-composition-promotion-v30.json"
$receiptPath = Join-Path $automationRoot "peak-rank-graph-context-composition-submission-receipt-v30.json"
$requiredOutputPattern = '(^|.*/)(candidate_evidence\.json|official_validator_candidate\.csv|run_stats\.csv|submission\.csv|validator_results\.csv|launcher_terminal\.json|worker-[01]\.json)$'
$gpuReserveHours = 8.0
$declaredWorstCaseGpuHours = 12.0
$gpuMutex = $null
$gpuMutexAcquired = $false

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "peak-rank-graph-context-composition-controller-v30"
        status = $Status
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        target_public_score = 0.945
        component_order = @("peak_rank_detector_v28", "graph_context_division_v2")
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
    return [pscustomobject]@{ ExitCode = $exitCode; Output = ($lines -join "`n") }
}

function Get-KaggleGpuRemainingHours {
    $native = Invoke-NativeOutput { & kaggle quota --format json }
    if ($native.ExitCode -ne 0) { throw "V30 Kaggle quota lookup failed: $($native.Output)" }
    $rows = $native.Output | ConvertFrom-Json
    $gpu = @($rows | Where-Object { $_.resource -eq "GPU" })
    if ($gpu.Count -ne 1) { throw "V30 Kaggle quota response is invalid" }
    $match = [regex]::Match([string]$gpu[0].remaining, '^([0-9]+(?:\.[0-9]+)?)h$')
    if (-not $match.Success) { throw "V30 Kaggle quota response is invalid" }
    return [double]::Parse(
        $match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Enter-KaggleGpuGate([DateTimeOffset]$Deadline) {
    $script:gpuMutex = [System.Threading.Mutex]::new(
        $false, "Global\BiohubKaggleGpuSessionV1"
    )
    while ([DateTimeOffset]::UtcNow -lt $Deadline) {
        $acquired = $false
        try {
            if ($script:gpuMutex.WaitOne(0)) {
                $script:gpuMutexAcquired = $true
                $acquired = $true
            }
        }
        catch [System.Threading.AbandonedMutexException] {
            $script:gpuMutexAcquired = $true
            $acquired = $true
        }
        if (-not $acquired) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $remaining = Get-KaggleGpuRemainingHours
        Write-Log "gpu_gate remaining_hours=$remaining worst_case_hours=$declaredWorstCaseGpuHours reserve_hours=$gpuReserveHours"
        if ($remaining - $declaredWorstCaseGpuHours -ge $gpuReserveHours) {
            return $remaining
        }
        $script:gpuMutex.ReleaseMutex()
        $script:gpuMutexAcquired = $false
        Start-Sleep -Seconds $PollSeconds
    }
    throw "Timed out waiting for V30 Kaggle GPU capacity"
}

New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
foreach ($required in @(
    $builder, $verifier, $exactScorer, $submitter, $kernelState,
    $evaluationPython, $baselineValidator, $scorerLock
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "V30 required input is missing: $required"
    }
}
foreach ($directory in @($controlGraphs, $truthGraphs, $organizerCheckout, $tracksdataCheckout)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "V30 required directory is missing: $directory"
    }
}
& $evaluationPython -m py_compile $builder $verifier $exactScorer $submitter $kernelState
if ($LASTEXITCODE -ne 0) { throw "V30 Python preflight failed" }
if ($ValidateOnly) {
    @{
        status = "validated"
        run_id = "peak-rank-graph-context-composition-controller-v30"
        exact_gpu_count = 2
        patched_official_comparison_to_v28_required = $true
        required_reserve_gpu_hours = $gpuReserveHours
    } | ConvertTo-Json
    exit 0
}
foreach ($forbidden in @($terminalPath, $candidateRoot, $promotionPath, $receiptPath)) {
    if (Test-Path -LiteralPath $forbidden) { throw "Refusing to reuse V30 state: $forbidden" }
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    $graph = $null
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $peakController -PathType Leaf) {
            $peakState = Get-Content -Raw -LiteralPath $peakController | ConvertFrom-Json
            if ($peakState.status -in @(
                "candidate_rejected", "candidate_rejected_by_patched_official_metric",
                "failed", "skipped_for_gpu_reserve", "skipped_after_scientific_rejection"
            )) {
                Write-Terminal "skipped_after_peak_component_rejection" @{
                    peak_component_status = $peakState.status
                }
                exit 0
            }
        }
        if (Test-Path -LiteralPath $graphDevelopment -PathType Leaf) {
            $graph = Get-Content -Raw -LiteralPath $graphDevelopment | ConvertFrom-Json
            if ($graph.status -ne "runtime_packaged") {
                Write-Terminal "skipped_after_graph_component_rejection" @{
                    graph_component_status = $graph.status
                }
                exit 0
            }
        }
        if ($null -ne $graph -and (Test-Path -LiteralPath $peakPromotion -PathType Leaf)) {
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if ($null -eq $graph -or -not (Test-Path -LiteralPath $peakPromotion -PathType Leaf)) {
        throw "Timed out waiting for independently gated V30 components"
    }
    $peak = Get-Content -Raw -LiteralPath $peakPromotion | ConvertFrom-Json
    if ($peak.status -ne "eligible_for_submission" -or $peak.authorized_for_submission -ne $true) {
        throw "V28 peak component promotion is invalid"
    }
    if (-not (Test-Path -LiteralPath $peakRuntimeManifest -PathType Leaf)) {
        throw "V28 promoted runtime is missing"
    }
    $peakManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $peakRuntimeManifest).Hash.ToLowerInvariant()
    if ($peak.runtime_manifest_sha256 -ne $peakManifestHash) {
        throw "V28 runtime changed after promotion"
    }
    $graphRuntimeRoot = [string]$graph.runtime_root
    $graphRuntimeManifest = [string]$graph.runtime_manifest
    if (-not (Test-Path -LiteralPath $graphRuntimeManifest -PathType Leaf)) {
        throw "Graph-context promoted runtime is missing"
    }
    $graphManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $graphRuntimeManifest).Hash.ToLowerInvariant()
    if ($graph.runtime_manifest_sha256 -ne $graphManifestHash) {
        throw "Graph-context runtime changed after development acceptance"
    }

    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s "biohub-graph-context-consensus-division-v1" --format json }
    if ($native.ExitCode -ne 0) { throw "V30 graph runtime inventory failed: $($native.Output)" }
    if ($native.Output -match [regex]::Escape($graphRuntimeRef)) {
        $native = Invoke-NativeOutput { & kaggle datasets version -p $graphRuntimeRoot -m "Frozen-policy graph-context ensemble v2" }
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $graphRuntimeRoot }
    }
    if ($native.ExitCode -ne 0) { throw "V30 graph runtime upload failed: $($native.Output)" }
    $graphReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $graphRuntimeRef --format json }
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)ready|complete") {
            $graphReady = $true
            break
        }
        if ($native.Output -match "(?i)error|failed") {
            throw "V30 graph runtime processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $graphReady) { throw "Timed out waiting for V30 graph runtime readiness" }

    $native = Invoke-NativeOutput {
        & $evaluationPython $builder --peak-runtime-root $peakRuntimeRoot --graph-root $graphRuntimeRoot
    }
    if ($native.ExitCode -ne 0) { throw "V30 build failed: $($native.Output)" }
    $metadata = Get-Content -Raw -LiteralPath $candidateMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or $metadata.machine_shape -ne "NvidiaTeslaT4" -or
        $metadata.dataset_sources -notcontains $peakRuntimeRef -or
        $metadata.dataset_sources -notcontains $graphRuntimeRef -or
        $metadata.competition_sources -notcontains $competitionRef -or
        $metadata.kernel_sources.Count -ne 0
    ) { throw "V30 candidate metadata is invalid" }
    $notebook = Get-Content -Raw -LiteralPath $candidateNotebook
    foreach ($requiredPattern in @(
        "predict_with_official_linker.py", "device_count() != 2",
        "_graph_context_scores_for_candidates", "Graph-context models parked on CPU",
        "Graph-context models restored", "pending_external_patched_official_scoring"
    )) {
        if ($notebook -notmatch [regex]::Escape($requiredPattern)) {
            throw "V30 notebook lost required contract: $requiredPattern"
        }
    }
    if ($notebook -match "kaggle competitions submit") {
        throw "V30 notebook contains a submission command"
    }

    $native = Invoke-NativeOutput { & $evaluationPython $kernelState --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "V30 kernel state lookup failed: $($native.Output)" }
    $before = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$before.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid V30 kernel version" }
    $remaining = Enter-KaggleGpuGate $deadline
    $launched = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels push -p $candidateRoot }
        Write-Log "kernel_push expected_version=$expectedVersion output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)successfully pushed") {
            $launched = $true
            break
        }
        if ($native.Output -notmatch "(?i)quota|accelerator|maximum batch GPU session count|temporarily") {
            throw "V30 kernel push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for V30 Kaggle slot" }
    Start-Sleep -Seconds 15
    $native = Invoke-NativeOutput { & $evaluationPython $kernelState --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "V30 post-launch state lookup failed: $($native.Output)" }
    $state = $native.Output | ConvertFrom-Json
    if (
        $state.present -ne $true -or [int]$state.current_version_number -ne $expectedVersion -or
        $state.is_private -ne $true -or $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or $state.enable_internet -ne $false -or
        $state.dataset_sources -notcontains $peakRuntimeRef -or
        $state.dataset_sources -notcontains $graphRuntimeRef -or
        $state.competition_sources -notcontains $competitionRef
    ) { throw "Remote V30 kernel state is invalid" }
    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-Log "kernel_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)COMPLETE") {
            $complete = $true
            break
        }
        if ($native.Output -match "(?i)ERROR|CANCEL") { throw "V30 Kaggle kernel failed: $($native.Output)" }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for V30 completion" }

    $downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/kernel-outputs/peak-rank-graph-context-composition-v30-version$expectedVersion"
    if (Test-Path -LiteralPath $downloadRoot) { throw "Refusing to reuse V30 output" }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput {
        & kaggle kernels output "$kernelRef/$expectedVersion" -p $downloadRoot --force `
            --file-pattern $requiredOutputPattern --page-size 200
    }
    if ($native.ExitCode -ne 0) { throw "V30 output download failed: $($native.Output)" }
    $officialValidator = @(Get-ChildItem -LiteralPath $downloadRoot -Recurse -File -Filter "official_validator_candidate.csv")
    if ($officialValidator.Count -ne 1) { throw "V30 materialized validator is missing or duplicated" }
    $officialMetricResult = Join-Path $downloadRoot "official_metric_result.json"
    $native = Invoke-NativeOutput {
        & $evaluationPython $exactScorer --candidate-validator $officialValidator[0].FullName `
            --control-dir $controlGraphs --truth-dir $truthGraphs --scorer-lock $scorerLock `
            --organizer-checkout $organizerCheckout --tracksdata-checkout $tracksdataCheckout `
            --output $officialMetricResult
    }
    if ($native.ExitCode -ne 0) {
        Write-Terminal "composition_rejected_by_patched_official_metric" @{
            kernel_version = $expectedVersion; verification_error = $native.Output
        }
        exit 0
    }
    $native = Invoke-NativeOutput {
        & $evaluationPython $verifier --output-root $downloadRoot `
            --baseline-validator $baselineValidator --peak-runtime-manifest $peakRuntimeManifest `
            --graph-runtime-manifest $graphRuntimeManifest --graph-development-terminal $graphDevelopment `
            --peak-promotion $peakPromotion --official-metric-result $officialMetricResult `
            --report $promotionPath
    }
    if ($native.ExitCode -ne 0) {
        Write-Terminal "composition_rejected" @{
            kernel_version = $expectedVersion; verification_error = $native.Output
        }
        exit 0
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne "eligible_for_submission" -or $promotion.authorized_for_submission -ne $true) {
        throw "V30 promotion report is invalid"
    }
    $native = Invoke-NativeOutput {
        & $evaluationPython $submitter --promotion $promotionPath --receipt $receiptPath `
            --kernel-ref $kernelRef --kernel-version $expectedVersion --expected-run-id "peak-rank-graph-context-composition-v30" `
            --message "V28 detector plus frozen graph-context ensemble; patched-official improvement" --execute
    }
    if ($native.ExitCode -ne 0) { throw "V30 promoted submission failed: $($native.Output)" }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne "submitted" -or $receipt.competition_submission_performed -ne $true) {
        throw "V30 submission receipt is invalid"
    }
    Write-Terminal "submitted" @{
        kernel_version = $expectedVersion
        remaining_gpu_hours_before_launch = $remaining
        peak_runtime_manifest_sha256 = $peakManifestHash
        graph_runtime_manifest_sha256 = $graphManifestHash
        exact_candidate_score = $promotion.exact_candidate_score
        composition_gain_over_peak_component = $promotion.composition_gain_over_peak_component
        composition_worst_movie_delta = $promotion.composition_worst_movie_delta
        submission_sha256 = $promotion.submission_sha256
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        competition_submission_performed = $true
    }
}
catch {
    Write-Terminal "failed" @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-Log "failed error=$($_.Exception.Message)"
    throw
}
finally {
    if ($gpuMutexAcquired -and $null -ne $gpuMutex) {
        $gpuMutex.ReleaseMutex()
        $gpuMutexAcquired = $false
    }
    if ($null -ne $gpuMutex) { $gpuMutex.Dispose() }
}
