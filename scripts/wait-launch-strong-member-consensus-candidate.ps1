param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 48.0,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 60 -or $MaximumWaitHours -le 0) {
    throw "Invalid strong-member launch wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$runtimeRef = "indarkarhana/biohub-strong-member-consensus-division-v2"
$kernelRef = "indarkarhana/biohub-ema-strong-member-candidate-v2"
$competitionRef = "biohub-cell-tracking-during-development"
$postHarvestTerminal = Join-Path $RepositoryRoot ".biohub/cache/strong-member-consensus-post-harvest-v1/post-harvest-terminal.json"
$candidateDir = Join-Path $RepositoryRoot "kaggle/biohub-ema-strong-member-candidate-v2"
$candidateBuilder = Join-Path $RepositoryRoot "scripts/build-strong-member-consensus-submission-candidate.py"
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$promotionController = Join-Path $RepositoryRoot "scripts/wait-verify-submit-strong-member-consensus-candidate.ps1"
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "strong-member-candidate-launch-v2.json"
$logPath = Join-Path $automationRoot "strong-member-candidate-launch-v2.log"
New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
if (Test-Path -LiteralPath $terminalPath) {
    throw "Strong-member candidate launch already reached a terminal state"
}

function Write-LaunchTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "strong-member-candidate-launch-v2"
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
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
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

foreach ($required in @($candidateBuilder, $kernelStateScript, $promotionController)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required strong-member launch input is missing: $required"
    }
}
foreach ($command in @("python", "kaggle")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required strong-member launch command is missing: $command"
    }
}

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    while (-not (Test-Path -LiteralPath $postHarvestTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw "Timed out waiting for strong-member post-harvest evidence"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $postHarvest = Get-Content -Raw -LiteralPath $postHarvestTerminal | ConvertFrom-Json
    if ($postHarvest.status -ne "runtime_packaged" -or $postHarvest.runtime_created -ne $true) {
        Write-LaunchTerminal "skipped_after_scientific_rejection" @{
            post_harvest_status = $postHarvest.status
            dataset_uploaded = $false
            kernel_launched = $false
        }
        exit 0
    }
    $runtimeRoot = [string]$postHarvest.runtime_root
    $runtimeManifest = [string]$postHarvest.runtime_manifest
    foreach ($required in @($runtimeRoot, $runtimeManifest)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Packaged runtime input is missing: $required"
        }
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
    if ($manifestHash -ne [string]$postHarvest.runtime_manifest_sha256) {
        throw "Packaged runtime manifest changed after scientific promotion"
    }
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeRoot "dataset-metadata.json") | ConvertFrom-Json
    if ($runtimeMetadata.id -ne $runtimeRef -or $runtimeMetadata.isPrivate -ne $true) {
        throw "Strong-member runtime dataset metadata is invalid"
    }

    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s "biohub-strong-member-consensus-division-v2" --format json }
    if ($native.ExitCode -ne 0) {
        throw "Strong-member dataset inventory failed: $($native.Output)"
    }
    if ($native.Output -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput {
            & kaggle datasets version -p $runtimeRoot -m "Admitted strong-member division consensus v2"
        }
        $datasetOperation = "versioned"
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOperation = "created"
        if ($native.ExitCode -ne 0 -and $native.Output -match "(?i)already exists|conflict") {
            $native = Invoke-NativeOutput {
                & kaggle datasets version -p $runtimeRoot -m "Admitted strong-member division consensus v2"
            }
            $datasetOperation = "versioned_after_create_conflict"
        }
    }
    if ($native.ExitCode -ne 0) {
        throw "Strong-member runtime upload failed: $($native.Output)"
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
            throw "Strong-member runtime processing failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $datasetReady) {
        throw "Timed out waiting for strong-member runtime readiness"
    }

    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) {
        throw "Pre-launch kernel state lookup failed: $($native.Output)"
    }
    $beforeState = $native.Output | ConvertFrom-Json
    $expectedVersion = [int]$beforeState.next_version_number
    if ($expectedVersion -lt 1) {
        throw "Invalid next candidate kernel version"
    }
    if (Test-Path -LiteralPath $candidateDir) {
        throw "Refusing to overwrite existing strong-member candidate directory"
    }
    $native = Invoke-NativeOutput {
        & python $candidateBuilder --consensus-root $runtimeRoot
    }
    if ($native.ExitCode -ne 0) {
        throw "Strong-member candidate build failed: $($native.Output)"
    }
    $metadataPath = Join-Path $candidateDir "kernel-metadata.json"
    $notebookPath = Join-Path $candidateDir "biohub-ema-strong-member-candidate-v2.ipynb"
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
        throw "Built strong-member kernel metadata is invalid"
    }
    $notebook = Get-Content -Raw -LiteralPath $notebookPath
    if ($notebook -notmatch "Exactly two T4 GPUs are required") {
        throw "Built candidate lost its two-T4 runtime guard"
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
            throw "Strong-member candidate push failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $launched) {
        throw "Timed out waiting for a Kaggle two-T4 launch slot"
    }
    Start-Sleep -Seconds 10
    $native = Invoke-NativeOutput { & python $kernelStateScript --kernel-slug $kernelRef }
    if ($native.ExitCode -ne 0) {
        throw "Post-launch kernel state lookup failed: $($native.Output)"
    }
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
        throw "Remote strong-member kernel state is invalid: $($native.Output)"
    }

    $native = Invoke-NativeOutput {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $promotionController `
            -RuntimeManifest $runtimeManifest `
            -KernelVersion $expectedVersion `
            -ValidateOnly
    }
    if ($native.ExitCode -ne 0 -or $native.Output -notmatch '"status"\s*:\s*"valid"') {
        throw "Strong-member promotion controller preflight failed: $($native.Output)"
    }
    $promotionProcess = Start-Process -FilePath powershell.exe -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $promotionController,
        "-RuntimeManifest", $runtimeManifest,
        "-KernelVersion", $expectedVersion
    ) -WorkingDirectory $RepositoryRoot -WindowStyle Hidden -PassThru
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
