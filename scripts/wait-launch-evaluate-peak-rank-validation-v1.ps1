param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 36.0,
    [int]$PollSeconds = 120,
    [ValidateSet("v1", "depth-pu-v2", "capacity-pu-v3", "faint-pu-v4", "expanded-real-faint-v7", "expanded-real-local-shape-v9", "expanded-real-blob-v11", "capacity-faint-ensemble-v5", "capacity-faint-confidence-v6", "capacity-faint-expanded-v8", "expanded-local-shape-ensemble-v10", "expanded-blob-ensemble-v12", "logit-ensemble-v4")]
    [string]$Variant = "v1",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 60) {
    throw "Invalid peak-ranking validation wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$competitionRef = "biohub-cell-tracking-during-development"
$kernelStateScript = Join-Path $RepositoryRoot "scripts/get-kaggle-kernel-state.py"
$variantConfig = if ($Variant -eq "v1") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-validation-runtime-v1"
        kernel_ref = "indarkarhana/biohub-peak-rank-validation-v1"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-detector-v1/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-validation-runtime.py"
        kernel_builder = "scripts/build-peak-rank-validation-kernel.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-validation-runtime-v1"
        kernel_root = "kaggle/biohub-peak-rank-validation-v1"
        notebook_name = "biohub-peak-rank-validation-v1.ipynb"
        controller_id = "peak-rank-validation-controller-v1"
        output_slug = "peak-rank-validation-v1"
        parameter_count = 38381478
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "depth-pu-v2") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-depth-pu-validation-runtime-v2"
        kernel_ref = "indarkarhana/biohub-peak-rank-depth-pu-validation-v2"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-depth-pu-v2/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-depth-pu-validation-runtime.py"
        kernel_builder = "scripts/build-peak-rank-depth-pu-validation-kernel.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-depth-pu-validation-runtime-v2"
        kernel_root = "kaggle/biohub-peak-rank-depth-pu-validation-v2"
        notebook_name = "biohub-peak-rank-depth-pu-validation-v2.ipynb"
        controller_id = "peak-rank-depth-pu-validation-controller-v2"
        output_slug = "peak-rank-depth-pu-validation-v2"
        parameter_count = 38381478
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "capacity-pu-v3") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-pu-validation-runtime-v3"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-pu-validation-v3"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-capacity-pu-v3/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-capacity-pu-validation-runtime.py"
        kernel_builder = "scripts/build-peak-rank-capacity-pu-validation-kernel.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-pu-validation-runtime-v3"
        kernel_root = "kaggle/biohub-peak-rank-capacity-pu-validation-v3"
        notebook_name = "biohub-peak-rank-capacity-pu-validation-v3.ipynb"
        controller_id = "peak-rank-capacity-pu-validation-controller-v3"
        output_slug = "peak-rank-capacity-pu-validation-v3"
        parameter_count = 66977670
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "faint-pu-v4") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-faint-pu-validation-runtime-v4"
        kernel_ref = "indarkarhana/biohub-peak-rank-faint-pu-validation-v4"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-faint-pu-v4/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-faint-pu-validation-runtime.py"
        kernel_builder = "scripts/build-peak-rank-faint-pu-validation-kernel.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-faint-pu-validation-runtime-v4"
        kernel_root = "kaggle/biohub-peak-rank-faint-pu-validation-v4"
        notebook_name = "biohub-peak-rank-faint-pu-validation-v4.ipynb"
        controller_id = "peak-rank-faint-pu-validation-controller-v4"
        output_slug = "peak-rank-faint-pu-validation-v4"
        parameter_count = 66977670
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "expanded-real-faint-v7") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-faint-validation-runtime-v7"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-faint-validation-v7"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-expanded-real-faint-v7/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-expanded-real-faint-validation-runtime-v7.py"
        kernel_builder = "scripts/build-peak-rank-expanded-real-faint-validation-kernel-v7.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-faint-validation-runtime-v7"
        kernel_root = "kaggle/biohub-peak-rank-expanded-real-faint-validation-v7"
        notebook_name = "biohub-peak-rank-expanded-real-faint-validation-v7.ipynb"
        controller_id = "peak-rank-expanded-real-faint-validation-controller-v7"
        output_slug = "peak-rank-expanded-real-faint-validation-v7"
        parameter_count = 66977670
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "expanded-real-local-shape-v9") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-local-shape-validation-v9"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-expanded-real-local-shape-v9/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-expanded-real-local-shape-validation-runtime-v9.py"
        kernel_builder = "scripts/build-peak-rank-expanded-real-local-shape-validation-kernel-v9.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-local-shape-validation-runtime-v9"
        kernel_root = "kaggle/biohub-peak-rank-expanded-real-local-shape-validation-v9"
        notebook_name = "biohub-peak-rank-expanded-real-local-shape-validation-v9.ipynb"
        controller_id = "peak-rank-expanded-real-local-shape-validation-controller-v9"
        output_slug = "peak-rank-expanded-real-local-shape-validation-v9"
        parameter_count = 66977670
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "expanded-real-blob-v11") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-real-blob-validation-runtime-v11"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-real-blob-validation-v11"
        harvest_terminal = ".biohub/cache/antelume-peak-rank-expanded-real-blob-v11/harvest-terminal.json"
        runtime_builder = "scripts/build-peak-rank-expanded-real-blob-validation-runtime-v11.py"
        kernel_builder = "scripts/build-peak-rank-expanded-real-blob-validation-kernel-v11.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-real-blob-validation-runtime-v11"
        kernel_root = "kaggle/biohub-peak-rank-expanded-real-blob-validation-v11"
        notebook_name = "biohub-peak-rank-expanded-real-blob-validation-v11.ipynb"
        controller_id = "peak-rank-expanded-real-blob-validation-controller-v11"
        output_slug = "peak-rank-expanded-real-blob-validation-v11"
        parameter_count = 66984582
        dependency_terminals = @()
    }
}
elseif ($Variant -eq "capacity-faint-ensemble-v5") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-faint-ensemble-validation-runtime-v5"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-faint-ensemble-validation-v5"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-capacity-faint-ensemble-validation-runtime-v5.py"
        kernel_builder = "scripts/build-peak-rank-capacity-faint-ensemble-validation-kernel-v5.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-faint-ensemble-validation-runtime-v5"
        kernel_root = "kaggle/biohub-peak-rank-capacity-faint-ensemble-validation-v5"
        notebook_name = "biohub-peak-rank-capacity-faint-ensemble-validation-v5.ipynb"
        controller_id = "peak-rank-capacity-faint-ensemble-validation-controller-v5"
        output_slug = "peak-rank-capacity-faint-ensemble-validation-v5"
        parameter_count = 133955340
        dependency_terminals = @(
            ".biohub/automation/peak-rank-capacity-pu-validation-controller-v3.json",
            ".biohub/automation/peak-rank-faint-pu-validation-controller-v4.json"
        )
    }
}
elseif ($Variant -eq "capacity-faint-confidence-v6") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
        kernel_ref = "indarkarhana/biohub-peak-rank-capacity-faint-confidence-validation-v6"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-capacity-faint-confidence-ensemble-validation-runtime-v6.py"
        kernel_builder = "scripts/build-peak-rank-capacity-faint-confidence-ensemble-validation-kernel-v6.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-capacity-faint-confidence-validation-runtime-v6"
        kernel_root = "kaggle/biohub-peak-rank-capacity-faint-confidence-validation-v6"
        notebook_name = "biohub-peak-rank-capacity-faint-confidence-validation-v6.ipynb"
        controller_id = "peak-rank-capacity-faint-confidence-validation-controller-v6"
        output_slug = "peak-rank-capacity-faint-confidence-validation-v6"
        parameter_count = 133955340
        dependency_terminals = @(
            ".biohub/automation/peak-rank-capacity-pu-validation-controller-v3.json",
            ".biohub/automation/peak-rank-faint-pu-validation-controller-v4.json"
        )
    }
}
elseif ($Variant -eq "capacity-faint-expanded-v8") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
        kernel_ref = "indarkarhana/biohub-peak-rank-cfe-ensemble-validation-v8"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-cfe-ensemble-validation-runtime-v8.py"
        kernel_builder = "scripts/build-peak-rank-cfe-ensemble-validation-kernel-v8.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-cfe-ensemble-validation-runtime-v8"
        kernel_root = "kaggle/biohub-peak-rank-cfe-ensemble-validation-v8"
        notebook_name = "biohub-peak-rank-cfe-ensemble-validation-v8.ipynb"
        controller_id = "peak-rank-cfe-ensemble-validation-controller-v8"
        output_slug = "peak-rank-cfe-ensemble-validation-v8"
        parameter_count = 200933010
        dependency_terminals = @(
            ".biohub/automation/peak-rank-capacity-pu-validation-controller-v3.json",
            ".biohub/automation/peak-rank-faint-pu-validation-controller-v4.json",
            ".biohub/automation/peak-rank-expanded-real-faint-validation-controller-v7.json"
        )
    }
}
elseif ($Variant -eq "expanded-local-shape-ensemble-v10") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-local-shape-ensemble-validation-v10"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10.py"
        kernel_builder = "scripts/build-peak-rank-expanded-local-shape-ensemble-validation-kernel-v10.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-local-shape-ensemble-validation-runtime-v10"
        kernel_root = "kaggle/biohub-peak-rank-expanded-local-shape-ensemble-validation-v10"
        notebook_name = "biohub-peak-rank-expanded-local-shape-ensemble-validation-v10.ipynb"
        controller_id = "peak-rank-expanded-local-shape-ensemble-validation-controller-v10"
        output_slug = "peak-rank-expanded-local-shape-ensemble-validation-v10"
        parameter_count = 133955340
        dependency_terminals = @(
            ".biohub/automation/peak-rank-expanded-real-faint-validation-controller-v7.json",
            ".biohub/automation/peak-rank-expanded-real-local-shape-validation-controller-v9.json"
        )
    }
}
elseif ($Variant -eq "expanded-blob-ensemble-v12") {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
        kernel_ref = "indarkarhana/biohub-peak-rank-expanded-blob-ensemble-validation-v12"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-expanded-blob-ensemble-validation-runtime-v12.py"
        kernel_builder = "scripts/build-peak-rank-expanded-blob-ensemble-validation-kernel-v12.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-expanded-blob-ensemble-validation-runtime-v12"
        kernel_root = "kaggle/biohub-peak-rank-expanded-blob-ensemble-validation-v12"
        notebook_name = "biohub-peak-rank-expanded-blob-ensemble-validation-v12.ipynb"
        controller_id = "peak-rank-expanded-blob-ensemble-validation-controller-v12"
        output_slug = "peak-rank-expanded-blob-ensemble-validation-v12"
        parameter_count = 200939922
        dependency_terminals = @(
            ".biohub/automation/peak-rank-expanded-real-faint-validation-controller-v7.json",
            ".biohub/automation/peak-rank-expanded-real-local-shape-validation-controller-v9.json",
            ".biohub/automation/peak-rank-expanded-real-blob-validation-controller-v11.json"
        )
    }
}
else {
    @{
        runtime_ref = "indarkarhana/biohub-peak-rank-logit-ensemble-validation-runtime-v4"
        kernel_ref = "indarkarhana/biohub-peak-rank-logit-ensemble-validation-v4"
        harvest_terminal = $null
        runtime_builder = "scripts/build-peak-rank-logit-ensemble-validation-runtime.py"
        kernel_builder = "scripts/build-peak-rank-logit-ensemble-validation-kernel.py"
        runtime_root = ".biohub/staging/biohub-peak-rank-logit-ensemble-validation-runtime-v4"
        kernel_root = "kaggle/biohub-peak-rank-logit-ensemble-validation-v4"
        notebook_name = "biohub-peak-rank-logit-ensemble-validation-v4.ipynb"
        controller_id = "peak-rank-logit-ensemble-validation-controller-v4"
        output_slug = "peak-rank-logit-ensemble-validation-v4"
        parameter_count = 76762956
        dependency_terminals = @(
            ".biohub/automation/peak-rank-validation-controller-v1.json",
            ".biohub/automation/peak-rank-depth-pu-validation-controller-v2.json"
        )
    }
}
$runtimeRef = $variantConfig.runtime_ref
$kernelRef = $variantConfig.kernel_ref
$harvestTerminal = if ($null -eq $variantConfig.harvest_terminal) {
    $null
} else {
    Join-Path $RepositoryRoot $variantConfig.harvest_terminal
}
$dependencyTerminals = @(
    $variantConfig.dependency_terminals | ForEach-Object { Join-Path $RepositoryRoot $_ }
)
$runtimeBuilder = Join-Path $RepositoryRoot $variantConfig.runtime_builder
$kernelBuilder = Join-Path $RepositoryRoot $variantConfig.kernel_builder
$runtimeRoot = Join-Path $RepositoryRoot $variantConfig.runtime_root
$runtimeManifest = Join-Path $runtimeRoot "SOURCE_MANIFEST.json"
$kernelRoot = Join-Path $RepositoryRoot $variantConfig.kernel_root
$metadataPath = Join-Path $kernelRoot "kernel-metadata.json"
$notebookPath = Join-Path $kernelRoot $variantConfig.notebook_name
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot ($variantConfig.controller_id + ".json")
$logPath = Join-Path $automationRoot ($variantConfig.controller_id + ".log")

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = $variantConfig.controller_id
        variant = $Variant
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
        dependency_terminals = $dependencyTerminals
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
    if ($dependencyTerminals.Count -gt 0) {
        while (@($dependencyTerminals | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -gt 0) {
            if ([DateTimeOffset]::UtcNow -ge $deadline) {
                throw "Timed out waiting for clean ensemble member validation"
            }
            Start-Sleep -Seconds $PollSeconds
        }
        $dependencies = @($dependencyTerminals | ForEach-Object {
            Get-Content -Raw -LiteralPath $_ | ConvertFrom-Json
        })
        $membersAccepted = @($dependencies | Where-Object {
            $_.status -eq "completed" -and
            $_.accepted_for_candidate_integration -eq $true -and
            $_.promotion_passed -eq $true -and
            $_.competition_submission_performed -eq $false
        }).Count -eq $dependencyTerminals.Count
        if (-not $membersAccepted) {
            Write-Terminal "skipped_after_member_rejection" @{
                model_status = "one_or_more_ensemble_members_rejected"
                member_statuses = @($dependencies | ForEach-Object { $_.status })
                selection_passed = $false
                audit_opened = $false
                audit_passed = $false
                dataset_uploaded = $false
                kernel_launched = $false
            }
            exit 0
        }
        $harvest = [pscustomobject]@{
            status = "clean_members_verified"
            accepted_for_kaggle_validation = $true
            model_status = "fixed_equal_logit_ensemble"
            selection_passed = $true
            audit_opened = $true
            audit_passed = $true
            checkpoint_sha256 = $null
        }
    }
    else {
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
    }
    if ($harvest.audit_passed -ne $true) {
        throw "Peak-ranking harvest authorization is internally inconsistent"
    }
    if ($dependencyTerminals.Count -eq 0 -and -not $harvest.checkpoint_sha256) {
        throw "Peak-ranking harvest is missing its checkpoint hash"
    }
    if ((Test-Path -LiteralPath $runtimeRoot) -or (Test-Path -LiteralPath $kernelRoot)) {
        throw "Refusing to overwrite an existing peak-ranking runtime or kernel"
    }
    $native = Invoke-NativeOutput { & python $runtimeBuilder }
    if ($native.ExitCode -ne 0) { throw "Runtime build failed: $($native.Output)" }
    $manifest = Get-Content -Raw -LiteralPath $runtimeManifest | ConvertFrom-Json
    if ($dependencyTerminals.Count -gt 0) {
        $harvest.checkpoint_sha256 = $manifest.checkpoint_sha256
    }
    if (
        $manifest.training_audit_passed -ne $true -or
        [int64]$manifest.parameter_count -ne [int64]$variantConfig.parameter_count -or
        $manifest.checkpoint_sha256 -ne $harvest.checkpoint_sha256
    ) {
        throw "Built runtime manifest differs from verified training evidence"
    }
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeRoot "dataset-metadata.json") | ConvertFrom-Json
    if ($runtimeMetadata.id -ne $runtimeRef -or $runtimeMetadata.isPrivate -ne $true) {
        throw "Peak-ranking runtime dataset metadata is invalid"
    }
    $manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()

    $runtimeSlug = $runtimeRef.Split("/", 2)[1]
    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s $runtimeSlug --format json }
    if ($native.ExitCode -ne 0) { throw "Dataset inventory failed: $($native.Output)" }
    if ($native.Output -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m "Audited temporal peak-ranking detector $Variant" }
        $datasetOperation = "versioned"
    }
    else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOperation = "created"
        if ($native.ExitCode -ne 0 -and $native.Output -match "(?i)already exists|conflict") {
            $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m "Audited temporal peak-ranking detector $Variant" }
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
        '"--tta-modes", "none,rot4,d4"',
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
    $downloadRoot = Join-Path $RepositoryRoot (".biohub/cache/kernel-outputs/" + $variantConfig.output_slug + "-version" + $expectedVersion)
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
        selection_recall = if ($null -eq $result.selection) { $null } else { $result.selection.annotated_node_recall }
        selected_tta_mode = $result.selected_tta_mode
        selected_tta_views = $result.selected_tta_views
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
