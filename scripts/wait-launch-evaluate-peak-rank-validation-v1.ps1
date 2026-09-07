param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 36.0,
    [int]$PollSeconds = 120,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid peak-ranking validation wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$runtimeRef = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
$kernelRef = "indarkarhana/biohub-peak-rank-validation-v1"
$competitionRef = "biohub-cell-tracking-during-development"
$harvestTerminal = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-detector-v1/harvest-terminal.json"
$runtimeBuilder = Join-Path $RepositoryRoot "scripts/build-peak-rank-validation-runtime.py"
$kernelBuilder = Join-Path $RepositoryRoot "scripts/build-peak-rank-validation-kernel.py"
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$runtimeRoot = Join-Path $RepositoryRoot ".biohub/staging/biohub-peak-rank-validation-runtime-v1"
$runtimeManifest = Join-Path $runtimeRoot "SOURCE_MANIFEST.json"
$kernelRoot = Join-Path $RepositoryRoot "kaggle/biohub-peak-rank-validation-v1"
$metadataPath = Join-Path $kernelRoot "kernel-metadata.json"
$notebookPath = Join-Path $kernelRoot "biohub-peak-rank-validation-v1.ipynb"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "peak-rank-validation-controller-v1.json"
$logPath = Join-Path $automationRoot "peak-rank-validation-controller-v1.log"

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "peak-rank-validation-controller-v1"
        status = $Status
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        expected_gpu_count = 2
        competition_test_data_read_locally = $false
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
        authorized_for_submission = $false
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
foreach ($required in @($runtimeBuilder, $kernelBuilder, $kernelStateScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required peak-ranking validation input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required peak-ranking command is missing: $command"
    }
}
if ($ValidateOnly) {
    & python -m py_compile $runtimeBuilder $kernelBuilder $kernelStateScript
    if ($LASTEXITCODE -ne 0) { throw "Peak-ranking validation source check failed" }
    @{
        status = "validated"
        harvest_terminal = $harvestTerminal
        expected_gpu_count = 2
        submission_performed = $false
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Peak-ranking validation controller already reached a terminal state"
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    while (-not (Test-Path -LiteralPath $harvestTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Timed out waiting for verified peak-ranking harvest"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $harvest = Get-Content -Raw -LiteralPath $harvestTerminal | ConvertFrom-Json
    if ($harvest.status -ne "harvest_verified") {
        throw "Peak-ranking harvest did not verify"
    }
    if ($harvest.accepted_for_kaggle_validation -ne $true) {
        Write-Terminal "skipped_after_training_rejection" @{
            model_status = $harvest.model_status
            selection_passed = $harvest.selection_passed
            audit_opened = $harvest.audit_opened
            audit_passed = $harvest.audit_passed
            dataset_uploaded = $false
            kernel_launched = $false
        }
        exit 0
    }
    if ($harvest.audit_passed -ne $true -or -not $harvest.checkpoint_sha256) {
        throw "Peak-ranking harvest authorization is internally inconsistent"
    }
    if ((Test-Path -LiteralPath $runtimeRoot) -or (Test-Path -LiteralPath $kernelRoot)) {
        throw "Refusing to overwrite an existing peak-ranking runtime or kernel"
    }
    $native = Invoke-NativeOutput { & python $runtimeBuilder }
    if ($native.ExitCode -ne 0) { throw "Runtime build failed: $($native.Output)" }
    $manifest = Get-Content -Raw -LiteralPath $runtimeManifest | ConvertFrom-Json
    if (
        $manifest.training_audit_passed -ne $true -or
        [int64]$manifest.parameter_count -ne 38381478 -or
        $manifest.checkpoint_sha256 -ne $harvest.checkpoint_sha256
    ) {
        throw "Built runtime manifest differs from verified training evidence"
    }
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeRoot "dataset-metadata.json") | ConvertFrom-Json
    if ($runtimeMetadata.id -ne $runtimeRef -or $runtimeMetadata.isPrivate -ne $true) {
        throw "Peak-ranking runtime dataset metadata is invalid"
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()

    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s "biohub-peak-rank-validation-runtime-v1" --format json }
    if ($native.ExitCode -ne 0) { throw "Dataset inventory failed: $($native.Output)" }
    if ($native.Output -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m "Audited temporal peak-ranking detector v1" }
        $datasetOperation = "versioned"
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOperation = "created"
        if ($native.ExitCode -ne 0 -and $native.Output -match "(?i)already exists|conflict") {
            $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m "Audited temporal peak-ranking detector v1" }
            $datasetOperation = "versioned_after_create_conflict"
        }
    }
    if ($native.ExitCode -ne 0) { throw "Runtime upload failed: $($native.Output)" }
    Write-Log "runtime_uploaded operation=$datasetOperation output=$($native.Output)"
    $ready = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $runtimeRef --format json }
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)ready|complete") {
            $ready = $true
            break
        }
        if ($native.Output -match "(?i)error|failed") {
            throw "Runtime dataset processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $ready) { throw "Timed out waiting for runtime dataset readiness" }

    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Pre-launch kernel lookup failed: $($native.Output)" }
    $before = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$before.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid next validation kernel version" }
    $native = Invoke-NativeOutput { & python $kernelBuilder }
    if ($native.ExitCode -ne 0) { throw "Validation kernel build failed: $($native.Output)" }
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
        throw "Built peak-ranking validation metadata is invalid"
    }
    $notebook = Get-Content -Raw -LiteralPath $notebookPath
    foreach ($requiredPattern in @(
        "torch.cuda.device_count() != 2",
        '"--devices", "0,1"',
        "acceptance_opened",
        "competition_submission_performed"
    )) {
        if ($notebook -notmatch [regex]::Escape($requiredPattern)) {
            throw "Validation notebook lost required boundary: $requiredPattern"
        }
    }

    $launched = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels push -p $kernelRoot }
        Write-Log "kernel_push expected_version=$expectedVersion output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)successfully pushed") {
            $launched = $true
            break
        }
        if ($native.Output -notmatch "(?i)quota|accelerator|maximum batch GPU session count|temporarily") {
            throw "Validation kernel push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for a Kaggle dual-T4 slot" }
    Write-Log "kernel_launched version=$expectedVersion"

    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-Log "kernel_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)COMPLETE") {
            $complete = $true
            break
        }
        if ($native.Output -match "(?i)ERROR|CANCEL") {
            throw "Validation kernel failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for validation completion" }
    $downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/kernel-outputs/peak-rank-validation-v1-version$expectedVersion"
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Refusing to reuse a peak-ranking validation download"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput {
        & kaggle kernels output "$kernelRef/$expectedVersion" -p $downloadRoot --force --page-size 200
    }
    if ($native.ExitCode -ne 0) { throw "Validation output download failed: $($native.Output)" }
    $resultPath = Get-ChildItem -Recurse -File $downloadRoot | Where-Object { $_.Name -eq "peak_rank_validation.json" } | Select-Object -First 1
    $launcherPath = Get-ChildItem -Recurse -File $downloadRoot | Where-Object { $_.Name -eq "launcher_terminal.json" } | Select-Object -First 1
    if ($null -eq $resultPath -or $null -eq $launcherPath) {
        throw "Validation output is missing its result or launcher terminal"
    }
    $launcher = Get-Content -Raw -LiteralPath $launcherPath.FullName | ConvertFrom-Json
    $result = Get-Content -Raw -LiteralPath $resultPath.FullName | ConvertFrom-Json
    $resultHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resultPath.FullName).Hash.ToLowerInvariant()
    if (
        $launcher.status -ne "completed" -or
        $launcher.validation_result_sha256 -ne $resultHash -or
        $launcher.competition_submission_performed -ne $false -or
        $result.run_id -ne "temporal-peak-rank-clean-validation-v1" -or
        $result.competition_test_data_read -ne $false -or
        $result.public_leaderboard_used_for_selection -ne $false -or
        $result.competition_submission_performed -ne $false -or
        ($result.acceptance_opened -eq $true -and $result.selection_passed -ne $true)
    ) {
        throw "Peak-ranking validation evidence is internally inconsistent"
    }
    Write-Terminal "completed" @{
        model_status = $harvest.model_status
        checkpoint_sha256 = $harvest.checkpoint_sha256
        dataset_operation = $datasetOperation
        dataset_uploaded = $true
        runtime_manifest_sha256 = $manifestHash
        kernel_version = $expectedVersion
        kernel_launched = $true
        download_root = $downloadRoot
        validation_result_sha256 = $resultHash
        selection_passed = $result.selection_passed
        selection_recall = $result.selection.annotated_node_recall
        acceptance_opened = $result.acceptance_opened
        acceptance_recall = if ($null -eq $result.acceptance) { $null } else { $result.acceptance.annotated_node_recall }
        promotion_passed = $result.promotion_passed
        accepted_for_candidate_integration = $result.promotion_passed
    }
    exit 0
}
catch {
    Write-Terminal "failed" @{ error = $_.Exception.Message; kernel_launched = $false }
    Write-Log "failed error=$($_.Exception.Message)"
    exit 1
}
