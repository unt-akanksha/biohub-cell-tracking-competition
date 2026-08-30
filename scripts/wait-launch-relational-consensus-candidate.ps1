param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 48.0,
    [int]$PollSeconds = 60,
    [string]$RuntimeDatasetRef = "indarkarhana/biohub-relational-consensus-division-v1",
    [string]$CandidateKernelRef = "indarkarhana/biohub-ema-relational-consensus-v1",
    [string]$DevelopmentTerminalRelative = ".biohub/cache/relational-division-development-v1/development-terminal.json",
    [string]$CandidateDirectoryRelative = "kaggle/biohub-ema-relational-consensus-v1",
    [string]$CandidateBuilderRelative = "scripts/build-relational-consensus-submission-candidate.py",
    [string]$CandidateNotebookName = "biohub-ema-relational-consensus-v1.ipynb",
    [string]$VerifierRelative = "scripts/verify-relational-consensus-submission-candidate.py",
    [string]$SubmitterRelative = "scripts/submit-relational-consensus-candidate.py",
    [string]$RuntimeSearchTerm = "biohub-relational-consensus-division-v1",
    [string]$DatasetVersionMessage = "Development-admitted relational division consensus v1",
    [string]$StatePrefix = "relational-consensus-candidate",
    [string]$LaunchRunId = "relational-consensus-candidate-launch-v1",
    [string]$PromotionRunId = "relational-consensus-candidate-controller-v1",
    [string]$FamilyLabel = "Relational",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 60 -or $MaximumWaitHours -le 0) {
    throw "Invalid $FamilyLabel candidate launch wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$runtimeRef = $RuntimeDatasetRef
$kernelRef = $CandidateKernelRef
$competitionRef = "biohub-cell-tracking-during-development"
$developmentTerminal = Join-Path $RepositoryRoot $DevelopmentTerminalRelative
$candidateDir = Join-Path $RepositoryRoot $CandidateDirectoryRelative
$candidateBuilder = Join-Path $RepositoryRoot $CandidateBuilderRelative
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$promotionController = Join-Path $RepositoryRoot "scripts/wait-verify-submit-strong-member-consensus-candidate.ps1"
$verifier = Join-Path $RepositoryRoot $VerifierRelative
$submitter = Join-Path $RepositoryRoot $SubmitterRelative
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "$LaunchRunId.json"
$logPath = Join-Path $automationRoot "$LaunchRunId.log"
New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null

function Write-LaunchTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = $LaunchRunId
        status = $Status
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition_ref = $competitionRef
        expected_gpu_count = 2
        machine_shape = "NvidiaTeslaT4"
        competition_test_data_read_locally = $false
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Write-LaunchLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
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

foreach ($required in @($candidateBuilder, $kernelStateScript, $promotionController, $verifier, $submitter)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required $FamilyLabel launch input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required $FamilyLabel launch command is missing: $command"
    }
}

if ($ValidateOnly) {
    $parseErrors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile(
        $promotionController,
        [ref]$null,
        [ref]$parseErrors
    )
    if ($parseErrors.Count -ne 0) { throw "$FamilyLabel promotion controller syntax is invalid" }
    & python -m py_compile $candidateBuilder $verifier $submitter
    if ($LASTEXITCODE -ne 0) { throw "$FamilyLabel launch source validation failed" }
    @{
        status = "validated"
        run_id = $LaunchRunId
        scientific_terminal = $developmentTerminal
        candidate_submission_requires_promotion_gate = $true
    } | ConvertTo-Json
    exit 0
}

if (Test-Path -LiteralPath $terminalPath) {
    throw "$FamilyLabel candidate launch already reached a terminal state"
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    while (-not (Test-Path -LiteralPath $developmentTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Timed out waiting for $FamilyLabel development evidence"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $development = Get-Content -Raw -LiteralPath $developmentTerminal | ConvertFrom-Json
    if ($development.status -ne "runtime_packaged" -or $development.runtime_created -ne $true) {
        Write-LaunchTerminal "skipped_after_scientific_rejection" @{
            development_status = $development.status
            dataset_uploaded = $false
            kernel_launched = $false
        }
        exit 0
    }
    if (
        $development.authorized_for_full_candidate_evaluation -ne $true -or
        $development.authorized_for_submission -ne $false
    ) {
        throw "$FamilyLabel development authorization boundary is invalid"
    }
    $runtimeRoot = [string]$development.runtime_root
    $runtimeManifest = [string]$development.runtime_manifest
    foreach ($required in @($runtimeRoot, $runtimeManifest)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Packaged $FamilyLabel runtime input is missing: $required"
        }
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
    if ($manifestHash -ne [string]$development.runtime_manifest_sha256) {
        throw "Packaged $FamilyLabel runtime manifest changed after scientific promotion"
    }
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeRoot "dataset-metadata.json") | ConvertFrom-Json
    if ($runtimeMetadata.id -ne $runtimeRef -or $runtimeMetadata.isPrivate -ne $true) {
        throw "$FamilyLabel runtime dataset metadata is invalid"
    }

    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s $RuntimeSearchTerm --format json }
    if ($native.ExitCode -ne 0) {
        throw "$FamilyLabel dataset inventory failed: $($native.Output)"
    }
    if ($native.Output -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput {
            & kaggle datasets version -p $runtimeRoot -m $DatasetVersionMessage
        }
        $datasetOperation = "versioned"
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOperation = "created"
        if ($native.ExitCode -ne 0 -and $native.Output -match "(?i)already exists|conflict") {
            $native = Invoke-NativeOutput {
                & kaggle datasets version -p $runtimeRoot -m $DatasetVersionMessage
            }
            $datasetOperation = "versioned_after_create_conflict"
        }
    }
    if ($native.ExitCode -ne 0) {
        throw "$FamilyLabel runtime upload failed: $($native.Output)"
    }
    Write-LaunchLog "runtime_uploaded operation=$datasetOperation output=$($native.Output)"

    $datasetReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $runtimeRef --format json }
        Write-LaunchLog "runtime_status output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)ready|complete") {
            $datasetReady = $true
            break
        }
        if ($native.Output -match "(?i)error|failed") {
            throw "$FamilyLabel runtime processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $datasetReady) { throw "Timed out waiting for $FamilyLabel runtime readiness" }

    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Pre-launch kernel state lookup failed: $($native.Output)" }
    $beforeState = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$beforeState.next_version_number
    if ($expectedVersion -lt 1) { throw "Invalid next $FamilyLabel candidate kernel version" }
    if (Test-Path -LiteralPath $candidateDir) {
        throw "Refusing to overwrite existing $FamilyLabel candidate directory"
    }
    $native = Invoke-NativeOutput { & python $candidateBuilder --consensus-root $runtimeRoot }
    if ($native.ExitCode -ne 0) { throw "$FamilyLabel candidate build failed: $($native.Output)" }

    $metadataPath = Join-Path $candidateDir "kernel-metadata.json"
    $notebookPath = Join-Path $candidateDir $CandidateNotebookName
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
        throw "Built $FamilyLabel kernel metadata is invalid"
    }
    $notebook = Get-Content -Raw -LiteralPath $notebookPath
    foreach ($requiredPattern in @(
        "Exactly two T4 GPUs are required",
        "cuda:0",
        "cuda:1",
        "ThreadPoolExecutor"
    )) {
        if ($notebook -notmatch [regex]::Escape($requiredPattern)) {
            throw "Built $FamilyLabel candidate lost required dual-T4 execution: $requiredPattern"
        }
    }

    $launched = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels push -p $candidateDir }
        Write-LaunchLog "kernel_push expected_version=$expectedVersion output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)successfully pushed") {
            $launched = $true
            break
        }
        if ($native.Output -notmatch "(?i)quota|accelerator|maximum batch GPU session count|temporarily") {
            throw "$FamilyLabel candidate push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) { throw "Timed out waiting for a Kaggle two-T4 launch slot" }

    Start-Sleep -Seconds 10
    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) { throw "Post-launch kernel state lookup failed: $($native.Output)" }
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
        throw "Remote $FamilyLabel kernel state is invalid: $($native.Output)"
    }

    $promotionArguments = @(
        "-RuntimeManifest", $runtimeManifest,
        "-KernelVersion", [string]$expectedVersion,
        "-CandidateKernelRef", $kernelRef,
        "-VerifierScript", $verifier,
        "-SubmitterScript", $submitter,
        "-StatePrefix", $StatePrefix,
        "-ControllerRunId", $PromotionRunId
    )
    $native = Invoke-NativeOutput {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $promotionController @promotionArguments -ValidateOnly
    }
    if ($native.ExitCode -ne 0 -or $native.Output -notmatch '"status"\s*:\s*"valid"') {
        throw "$FamilyLabel promotion controller preflight failed: $($native.Output)"
    }
    $promotionProcessArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $promotionController
    ) + $promotionArguments
    $promotionProcess = Start-Process -FilePath powershell.exe `
        -ArgumentList $promotionProcessArguments `
        -WorkingDirectory $RepositoryRoot `
        -WindowStyle Hidden `
        -PassThru
    Write-LaunchTerminal "launched" @{
        dataset_operation = $datasetOperation
        dataset_uploaded = $true
        runtime_manifest_sha256 = $manifestHash
        kernel_version = $expectedVersion
        kernel_metadata_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath).Hash.ToLowerInvariant()
        kernel_notebook_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $notebookPath).Hash.ToLowerInvariant()
        kernel_launched = $true
        promotion_controller_pid = $promotionProcess.Id
    }
    exit 0
}
catch {
    Write-LaunchTerminal "failed" @{
        error = $_.Exception.Message
        kernel_launched = $false
    }
    Write-LaunchLog "failed error=$($_.Exception.Message)"
    exit 1
}
