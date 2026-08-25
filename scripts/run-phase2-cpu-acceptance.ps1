[CmdletBinding()]
param([switch]$Execute)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".biohub/evaluation-venv/Scripts/python.exe"
$biohub = Join-Path $projectRoot ".biohub/evaluation-venv/Scripts/biohub.exe"
$stageRoot = Join-Path $projectRoot ".biohub/phase2-acceptance"
$currentStage = Join-Path $stageRoot "current"
$runtimeDownloadStage = Join-Path $currentStage "runtime-download"
$runtimeStage = Join-Path $currentStage "runtime-current"
$kernelStage = Join-Path $currentStage "kernel-next"
$serverStage = Join-Path $currentStage "kernel-server"
$downloadStage = Join-Path $currentStage "download"
$requestPath = Join-Path $currentStage "acceptance-request.json"
$quotaBeforePath = Join-Path $currentStage "gpu-quota-before.json"
$quotaPrePushPath = Join-Path $currentStage "gpu-quota-pre-push.json"
$quotaAfterPath = Join-Path $currentStage "gpu-quota-after.json"
$kernelLogPath = Join-Path $currentStage "kernel.log"
$datasetInventoryPath = Join-Path $currentStage "dataset-inventory.json"
$observedArtifactsPath = Join-Path $currentStage "observed-artifacts.json"
$configPath = Join-Path $projectRoot "config/phase2-control.json"
$competitionSlug = "biohub-cell-tracking-during-development"
$kernelSlug = "indarkarhana/biohub-phase-2-cpu-acceptance"
$datasetSlug = "indarkarhana/biohub-phase2-runtime"
$bundleName = "biohub-runtime-v1.biohubbundle"
$minimumRuntimeDatasetVersion = 9
$datasetRef = ""
$kernelRef = ""
$env:PYTHONPATH = Join-Path $projectRoot "src"

function Assert-ChildPath([string]$Candidate) {
    $full = [System.IO.Path]::GetFullPath($Candidate)
    $root = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\') + '\'
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing path outside project root: $full"
    }
}

function Invoke-Kaggle([string[]]$Arguments) {
    $output = & kaggle @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle command failed: kaggle $($Arguments -join ' ')`n$output"
    }
    $text = $output -join "`n"
    if ($text -match "(?i)dataset (creation|version) error:") {
        throw "Kaggle command reported an API error: kaggle $($Arguments -join ' ')`n$text"
    }
    return $text
}

function Invoke-KaggleReadWithRetry(
    [string[]]$Arguments,
    [int]$MaxAttempts = 10
) {
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            return Invoke-Kaggle $Arguments
        } catch {
            $message = $_.Exception.Message
            if (
                $message -notmatch "(?i)429|Too Many Requests" -or
                $attempt -eq $MaxAttempts
            ) {
                throw
            }
            Write-Output "KAGGLE_READ_RATE_LIMITED: attempt=$attempt/$MaxAttempts"
            Start-Sleep -Seconds 30
        }
    }
    throw "unreachable Kaggle read retry state"
}

function Write-KaggleJson([string[]]$Arguments, [string]$Path) {
    $text = Invoke-KaggleReadWithRetry $Arguments
    $null = $text | ConvertFrom-Json
    [System.IO.File]::WriteAllText(
        $Path, $text + "`n", [System.Text.UTF8Encoding]::new($false)
    )
}

function Get-DatasetState {
    return (
        Invoke-Kaggle @("datasets", "status", $datasetSlug, "--format", "json")
    ) | ConvertFrom-Json
}

function Assert-DatasetState($State, [int]$ExpectedVersion) {
    if ([int]$State.current_version_number -ne $ExpectedVersion) {
        throw "runtime dataset version mismatch: expected $ExpectedVersion"
    }
    if ([string]$State.status -notmatch "(?i)READY|COMPLETE") {
        throw "runtime dataset is not ready: $($State | ConvertTo-Json -Compress)"
    }
}

function Wait-DatasetReady([int]$ExpectedVersion) {
    $deadline = [DateTimeOffset]::UtcNow.AddMinutes(30)
    $last = ""
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        try {
            $state = Get-DatasetState
            $last = $state | ConvertTo-Json -Compress
            if (
                [int]$state.current_version_number -eq $ExpectedVersion -and
                [string]$state.status -match "(?i)READY|COMPLETE"
            ) {
                return $state
            }
            if ([string]$state.status -match "(?i)ERROR|FAILED") {
                throw "runtime dataset version failed: $last"
            }
        } catch {
            $last = $_.Exception.Message
        }
        Start-Sleep -Seconds 10
    }
    throw "runtime dataset did not become ready within 30 minutes: $last"
}

function Get-OwnedKernelState {
    $stateText = & $python (
        Join-Path $projectRoot "scripts/get-kaggle-kernel-state.py"
    ) --kernel-slug $kernelSlug
    if ($LASTEXITCODE -ne 0) {
        throw "authenticated SDK kernel-version lookup failed closed"
    }
    return ($stateText | ConvertFrom-Json)
}

function Assert-OwnedKernelServerState(
    $State,
    [int]$ExpectedVersion,
    [bool]$AllowDetachedRuntimeDataset = $false
) {
    if ($State.present -ne $true -or [string]$State.kernel_slug -ne $kernelSlug) {
        throw "exact owned CPU kernel is absent or has the wrong identity"
    }
    if ([int]$State.current_version_number -ne $ExpectedVersion) {
        throw "owned CPU kernel version mismatch: expected $ExpectedVersion"
    }
    if (
        $State.is_private -ne $true -or
        $State.enable_gpu -ne $false -or
        $State.enable_tpu -ne $false -or
        $State.enable_internet -ne $false
    ) {
        throw "server CPU kernel metadata is not private, CPU-only, and offline"
    }
    if ([string]$State.language -ne "python" -or [string]$State.kernel_type -ne "script") {
        throw "server CPU kernel source type changed"
    }
    $serverDatasetSources = @($State.dataset_sources)
    if (
        $serverDatasetSources.Count -gt 1 -or
        ($serverDatasetSources.Count -eq 1 -and [string]$serverDatasetSources[0] -ne $datasetSlug) -or
        (-not $AllowDetachedRuntimeDataset -and $serverDatasetSources.Count -ne 1)
    ) {
        throw "server runtime dataset source identity changed"
    }
    if (
        @($State.competition_sources).Count -ne 1 -or
        [string]$State.competition_sources[0] -ne $competitionSlug
    ) {
        throw "server competition source identity changed"
    }
    if (@($State.kernel_sources).Count -ne 0 -or @($State.model_sources).Count -ne 0) {
        throw "unexpected server kernel or model source"
    }
}

function Assert-NoActiveOwnedKernel {
    $status = Invoke-Kaggle @("kernels", "status", $kernelSlug)
    if ($status -match "(?i)RUNNING|QUEUED|STARTING") {
        throw "owned CPU acceptance kernel is already active: $status"
    }
    return $status
}

function Get-DatasetInventory {
    $text = & $python (
        Join-Path $projectRoot "scripts/get-kaggle-dataset-state.py"
    ) --dataset-ref $datasetRef
    if ($LASTEXITCODE -ne 0) {
        throw "authenticated SDK dataset inventory lookup failed closed"
    }
    return ($text | ConvertFrom-Json)
}

function Assert-DatasetInventory($Inventory, [long]$ExpectedSize) {
    if ([string]$Inventory.dataset_ref -ne $datasetRef) {
        throw "dataset inventory reference mismatch"
    }
    if ([int]$Inventory.file_count -ne 1 -or @($Inventory.files).Count -ne 1) {
        throw "dataset v3 must expose exactly one data file"
    }
    $file = @($Inventory.files)[0]
    if (
        [string]$file.name -ne $bundleName -or
        [long]$file.total_bytes -ne $ExpectedSize
    ) {
        throw "dataset v3 opaque bundle name or size mismatch"
    }
}

function Assert-GpuQuotaUnchanged([string]$Before, [string]$After) {
    $beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Before).Hash
    $afterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $After).Hash
    if ($beforeHash -ne $afterHash) {
        throw "GPU/TPU quota response changed during CPU acceptance"
    }
}

function Get-CanonicalTextSha256([string]$Path) {
    $strictUtf8 = [System.Text.UTF8Encoding]::new($false, $true)
    $text = $strictUtf8.GetString([System.IO.File]::ReadAllBytes($Path))
    $normalized = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($normalized)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace(
            "-", ""
        ).ToLowerInvariant()
    } finally {
        $hasher.Dispose()
    }
}

function Capture-TerminalEvidence {
    $hashes = @{}
    try {
        $logText = Invoke-KaggleReadWithRetry @("kernels", "logs", $kernelSlug)
        [System.IO.File]::WriteAllText(
            $kernelLogPath,
            $logText + "`n",
            [System.Text.UTF8Encoding]::new($false)
        )
        $hashes["kernel_log"] = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $kernelLogPath
        ).Hash.ToLowerInvariant()
    } catch { }
    try {
        Write-KaggleJson @("quota", "--format", "json") $quotaAfterPath
        $hashes["gpu_quota_after"] = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $quotaAfterPath
        ).Hash.ToLowerInvariant()
    } catch { }
    $observedJson = $hashes | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText(
        $observedArtifactsPath,
        $observedJson + "`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    return $observedArtifactsPath
}

Assert-ChildPath $currentStage
if (Test-Path -LiteralPath $currentStage) {
    $resolved = (Resolve-Path -LiteralPath $currentStage).Path
    Assert-ChildPath $resolved
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
New-Item -ItemType Directory -Path (
    $runtimeDownloadStage,
    $runtimeStage,
    $kernelStage,
    $serverStage,
    $downloadStage
) -Force | Out-Null

$metadataPath = Join-Path $projectRoot "kaggle/phase2-cpu-acceptance/kernel-metadata.json"
$kernelSourcePath = Join-Path $projectRoot "kaggle/phase2-cpu-acceptance/phase2_acceptance.py"
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$controlConfig = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
$cpuWatchdogSeconds = [int]$controlConfig.cpu_watchdog_minutes * 60
if ($cpuWatchdogSeconds -le 0 -or $cpuWatchdogSeconds -ge (12 * 60 * 60)) {
    throw "CPU watchdog must be positive and remain below Kaggle's 12-hour limit"
}
if (
    $metadata.enable_gpu -ne $false -or
    $metadata.enable_tpu -ne $false -or
    $metadata.enable_internet -ne $false -or
    [string]$metadata.id -ne $kernelSlug
) {
    throw "canonical kernel metadata is not the frozen private CPU/offline asset"
}
$kernelSource = Get-Content -Raw -LiteralPath $kernelSourcePath
$forbidden = @(
    "kaggle competitions submit",
    "biohub launch execute",
    "torch.cuda",
    ".cuda("
)
foreach ($token in $forbidden) {
    if ($kernelSource.ToLowerInvariant().Contains($token)) {
        throw "forbidden execution path in kernel source: $token"
    }
}

& $biohub --root $projectRoot cpu-acceptance preflight
if ($LASTEXITCODE -ne 0) { throw "local acceptance lifecycle preflight failed" }
$datasetState = Get-DatasetState
$currentDatasetVersion = [int]$datasetState.current_version_number
if ($currentDatasetVersion -lt $minimumRuntimeDatasetVersion) {
    throw "runtime dataset predates the verified opaque-bundle layout"
}
Assert-DatasetState $datasetState $currentDatasetVersion
$datasetRef = "$datasetSlug/$currentDatasetVersion"
$kernelState = Get-OwnedKernelState
$currentKernelVersion = [int]$kernelState.current_version_number
Assert-OwnedKernelServerState $kernelState $currentKernelVersion $true
$targetKernelVersion = [int]$kernelState.next_version_number
$kernelRef = [string]$kernelState.next_kernel_ref
if ($kernelRef -ne "$kernelSlug/$targetKernelVersion") {
    throw "owned CPU kernel next-version selection is not authoritative"
}
$null = Assert-NoActiveOwnedKernel
Write-KaggleJson @("quota", "--format", "json") $quotaBeforePath
$null = Invoke-Kaggle @(
    "competitions", "files", $competitionSlug, "--page-size", "1", "--format", "json"
)

$null = Invoke-Kaggle @(
    "datasets", "download", $datasetRef,
    "-p", $runtimeDownloadStage, "-q", "-o"
)
$downloadArchives = @(
    Get-ChildItem -LiteralPath $runtimeDownloadStage -File -Filter "*.zip"
)
if ($downloadArchives.Count -ne 1) {
    throw "expected exactly one read-only runtime dataset download archive"
}
Expand-Archive -LiteralPath $downloadArchives[0].FullName -DestinationPath $runtimeStage
$bundlePath = Join-Path $runtimeStage $bundleName
if (-not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) {
    throw "current runtime dataset did not contain the opaque bundle"
}
$bundleInspectionText = & $python (
    Join-Path $projectRoot "scripts/inspect-runtime-bundle.py"
) --bundle $bundlePath
if ($LASTEXITCODE -ne 0) { throw "runtime bundle inspection failed closed" }
$bundleInspection = $bundleInspectionText | ConvertFrom-Json
if (
    [string]$bundleInspection.runtime_bundle_name -ne $bundleName -or
    [int]$bundleInspection.runtime_bundle_file_count -ne 8801 -or
    [long]$bundleInspection.runtime_bundle_uncompressed_size_bytes -ne 1080199544 -or
    [string]$bundleInspection.runtime_bundle_sha256 -ne "235fc0a0ae41c405d8df9478d491773181934db26286e3663d31d4c09612898b" -or
    [string]$bundleInspection.runtime_bundle_inventory_sha256 -ne "94d1c272189e72f389a73aeffea777687ba1341d88de6e36fff8fdcd8eb083c3"
) {
    throw "runtime bundle differs from the fully verified immutable v9 evidence"
}
$inventory = Get-DatasetInventory
Assert-DatasetInventory $inventory (Get-Item -LiteralPath $bundlePath).Length

if (-not $Execute) {
    Write-Output "OFFLINE_PREFLIGHT_READY: current opaque bundle matches frozen v9 evidence; no request or Kaggle asset mutated."
    exit 0
}

$runStamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$runId = "phase2-cpu-control-$runStamp"
$evaluationRunId = "phase2-control-evaluation-$runStamp"
$registered = $false

try {
    & $biohub --root $projectRoot cpu-acceptance register `
        --config $configPath `
        --run-id $runId `
        --evaluation-run-id $evaluationRunId `
        --kernel-ref $kernelRef `
        --runtime-dataset-ref $datasetRef `
        --runtime-bundle $bundlePath `
        --output $requestPath
    if ($LASTEXITCODE -ne 0) { throw "CPU acceptance request registration failed" }
    $registered = $true

    Copy-Item -LiteralPath $metadataPath -Destination (
        Join-Path $kernelStage "kernel-metadata.json"
    )
    $requestBytes = [System.IO.File]::ReadAllBytes($requestPath)
    $requestB64 = [Convert]::ToBase64String($requestBytes)
    $requestSentinel = 'EMBEDDED_ACCEPTANCE_REQUEST_B64 = ""'
    $stagedKernelSource = $kernelSource.Replace(
        $requestSentinel,
        "EMBEDDED_ACCEPTANCE_REQUEST_B64 = `"$requestB64`""
    )
    if ($stagedKernelSource -eq $kernelSource) {
        throw "kernel request embedding sentinel is missing"
    }
    $stagedKernelPath = Join-Path $kernelStage "phase2_acceptance.py"
    [System.IO.File]::WriteAllText(
        $stagedKernelPath,
        $stagedKernelSource,
        [System.Text.UTF8Encoding]::new($false)
    )

    $datasetState = Get-DatasetState
    Assert-DatasetState $datasetState $currentDatasetVersion
    $inventory = Get-DatasetInventory
    Assert-DatasetInventory $inventory (Get-Item -LiteralPath $bundlePath).Length
    [System.IO.File]::WriteAllText(
        $datasetInventoryPath,
        ($inventory | ConvertTo-Json -Depth 8 -Compress) + "`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Output "RUNTIME_DATASET_READY: $datasetRef contains one bound opaque bundle."

    $datasetState = Get-DatasetState
    Assert-DatasetState $datasetState $currentDatasetVersion
    $inventory = Get-DatasetInventory
    Assert-DatasetInventory $inventory (Get-Item -LiteralPath $bundlePath).Length
    $kernelState = Get-OwnedKernelState
    Assert-OwnedKernelServerState $kernelState $currentKernelVersion $true
    if (
        [int]$kernelState.next_version_number -ne $targetKernelVersion -or
        [string]$kernelState.next_kernel_ref -ne $kernelRef
    ) {
        throw "owned CPU kernel changed between request registration and push"
    }
    $null = Assert-NoActiveOwnedKernel
    Write-KaggleJson @("quota", "--format", "json") $quotaPrePushPath
    Assert-GpuQuotaUnchanged $quotaBeforePath $quotaPrePushPath

    $null = Invoke-Kaggle @(
        "kernels", "push", "-p", $kernelStage,
        "-t", [string]$cpuWatchdogSeconds
    )
    $postPush = Get-OwnedKernelState
    Assert-OwnedKernelServerState $postPush $targetKernelVersion
    if (
        [int]$postPush.current_version_number -ne $targetKernelVersion -or
        [int]$postPush.next_version_number -ne ($targetKernelVersion + 1)
    ) {
        throw "server did not advance exactly to the authoritative target kernel"
    }
    $null = Invoke-Kaggle @("kernels", "pull", $kernelSlug, "-p", $serverStage, "-m")
    $serverSources = @(
        Get-ChildItem -LiteralPath $serverStage -File -Filter "*.py"
    )
    if ($serverSources.Count -ne 1) {
        throw "server kernel pull did not return exactly one source file"
    }
    $stagedRawHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $stagedKernelPath
    ).Hash.ToLowerInvariant()
    $serverRawHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $serverSources[0].FullName
    ).Hash.ToLowerInvariant()
    $stagedCanonicalHash = Get-CanonicalTextSha256 $stagedKernelPath
    $serverCanonicalHash = Get-CanonicalTextSha256 $serverSources[0].FullName
    if ($stagedCanonicalHash -ne $serverCanonicalHash) {
        throw "staged/server canonical kernel source hash mismatch"
    }
    & $biohub --root $projectRoot cpu-acceptance start $runId `
        --kernel-ref $kernelRef --runtime-dataset-ref $datasetRef
    if ($LASTEXITCODE -ne 0) { throw "CPU acceptance start event failed" }
    Write-Output (
        "KERNEL_LAUNCHED: $kernelRef canonical_source=$serverCanonicalHash " +
        "staged_raw=$stagedRawHash server_raw=$serverRawHash"
    )

    $deadline = [DateTimeOffset]::UtcNow.AddMinutes(660)
    $terminal = ""
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $terminal = Invoke-Kaggle @("kernels", "status", $kernelSlug)
        Write-Output "[$([DateTimeOffset]::UtcNow.ToString('u'))] $terminal"
        if ($terminal -match "(?i)COMPLETE") { break }
        if ($terminal -match "(?i)ERROR|FAILED|CANCEL") { break }
        Start-Sleep -Seconds 60
    }
    if ($terminal -notmatch "(?i)COMPLETE") {
        throw "CPU acceptance kernel did not complete: $terminal"
    }

    Start-Sleep -Seconds 30
    $null = Invoke-KaggleReadWithRetry @(
        "kernels", "output", $kernelSlug, "-p", $downloadStage,
        "-o", "-q", "--file-pattern", "pending-control-report.json"
    )
    $null = Capture-TerminalEvidence
    Assert-GpuQuotaUnchanged $quotaBeforePath $quotaAfterPath
    $pendingPath = Join-Path $downloadStage "pending-control-report.json"
    if (-not (Test-Path -LiteralPath $pendingPath -PathType Leaf)) {
        throw "compact pending control output is missing"
    }
    & $biohub --root $projectRoot cpu-acceptance reconcile $runId `
        --evidence $pendingPath `
        --quota-before $quotaBeforePath `
        --quota-after $quotaAfterPath `
        --manifest-output "manifests/reciprocal-embryo-v1.json" `
        --report-output "reports/exact/phase2-control-acceptance.json"
    if ($LASTEXITCODE -ne 0) { throw "local CPU control reconciliation failed" }
    Write-Output "TERMINAL_COMPLETE: $kernelRef reconciled against $datasetRef"
} catch {
    $failureDetail = $_.Exception.Message
    if ($failureDetail.Length -gt 1800) {
        $failureDetail = $failureDetail.Substring(0, 1800)
    }
    $observedArtifacts = Capture-TerminalEvidence
    if ($registered) {
        try {
            & $biohub --root $projectRoot cpu-acceptance fail $runId `
                --evaluation-run-id $evaluationRunId `
                --reason-code ACCEPTANCE_WRAPPER_FAILED `
                --detail $failureDetail `
                --observed-artifacts $observedArtifacts
        } catch { }
    }
    Write-Output "TERMINAL_ERROR: $failureDetail"
    throw
}
