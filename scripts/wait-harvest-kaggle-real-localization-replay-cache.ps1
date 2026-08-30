param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [string]$KernelRef = "indarkarhana/biohub-real-localization-replay-cache-v1",
    [int]$ExpectedKernelVersion = 1,
    [int]$ExistingOvernightOrchestratorPid = 0,
    [int]$PollSeconds = 120,
    [int]$MaximumWaitHours = 8
)

$ErrorActionPreference = "Stop"
if ($ExpectedKernelVersion -lt 1 -or $PollSeconds -lt 30 -or $MaximumWaitHours -lt 1) {
    throw "Invalid Kaggle replay-cache harvest bounds"
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot

$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/aws-4gpu-temporal-localizer-v1"
$downloadRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-kaggle-output-v1"
$canonicalRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-shards-v1"
$canonicalManifest = Join-Path $canonicalRoot "real_localization_shard_manifest.json"
$validator = Join-Path $RepositoryRoot "scripts/validate-real-localization-shard-cache.py"
$launcher = Join-Path $RepositoryRoot "scripts/wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1"
$terminalPath = Join-Path $stateRoot "kaggle-replay-fallback-terminal.json"
$logPath = Join-Path $stateRoot "kaggle-replay-fallback.log"
$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
$versionedKernelRef = "$KernelRef/$ExpectedKernelVersion"

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$([DateTimeOffset]::UtcNow.ToString('o')) $Message" -Encoding utf8
}

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "competition-real-localization-kaggle-fallback-v1"
        status = $Status
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
        kernel_ref = $versionedKernelRef
        accelerator = "cpu"
        kaggle_gpu_used = $false
        competition_train_data_read = $true
        competition_test_data_read = $false
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
        authorized_for_submission = $false
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Invoke-NativeOutput([scriptblock]$Command) {
    $output = (& $Command 2>&1) -join "`n"
    return @{ ExitCode = $LASTEXITCODE; Output = $output }
}

function Stop-ExactOvernightOrchestrator([int]$ProcessId) {
    if ($ProcessId -le 0) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $false }
    if (
        $process.Name -notmatch "(?i)^powershell(.exe)?$" -or
        [string]$process.CommandLine -notmatch "wait-build-launch-aws-temporal-localizer-v2\.ps1"
    ) {
        throw "Refusing to stop PID $ProcessId because its command identity changed"
    }
    $descendants = New-Object System.Collections.Generic.List[object]
    $frontier = @($ProcessId)
    while ($frontier.Count -gt 0) {
        $parents = $frontier
        $frontier = @()
        foreach ($parent in $parents) {
            $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $parent" -ErrorAction SilentlyContinue)
            foreach ($child in $children) {
                $descendants.Add($child)
                $frontier += [int]$child.ProcessId
            }
        }
    }
    foreach ($child in @($descendants | Sort-Object ProcessId -Descending)) {
        Stop-Process -Id ([int]$child.ProcessId) -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $ProcessId -Force
    Write-Log "stopped superseded local downloader orchestrator pid=$ProcessId descendants=$($descendants.Count)"
    return $true
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
foreach ($required in @($validator, $launcher)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Kaggle replay fallback input is missing: $required"
    }
}
if (Test-Path -LiteralPath $terminalPath -PathType Leaf) {
    throw "Kaggle replay fallback terminal already exists"
}

try {
    Write-Terminal "kernel_waiting" @{ expected_kernel_version = $ExpectedKernelVersion }
    $completed = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle kernels status $KernelRef }
        Write-Log "kernel_status exit=$($native.ExitCode) output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match "(?i)complete") {
            $completed = $true
            break
        }
        if ($native.Output -match "(?i)(error|failed|cancel)") {
            throw "Kaggle replay cache kernel failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $completed) { throw "Timed out waiting for Kaggle replay cache kernel" }
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Kaggle replay fallback download root already exists"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput {
        & kaggle kernels output $versionedKernelRef -p $downloadRoot --force
    }
    if ($native.ExitCode -ne 0) {
        throw "Kaggle replay cache output download failed: $($native.Output)"
    }
    Write-Log "kernel_output_downloaded output=$($native.Output)"
    $manifests = @(Get-ChildItem -LiteralPath $downloadRoot -Recurse -File -Filter "real_localization_shard_manifest.json")
    if ($manifests.Count -ne 1) {
        throw "Expected exactly one downloaded real-localization shard manifest"
    }
    $stagedRoot = $manifests[0].Directory.FullName
    $native = Invoke-NativeOutput { & python $validator --root $stagedRoot }
    if ($native.ExitCode -ne 0) {
        throw "Downloaded Kaggle replay cache failed validation: $($native.Output)"
    }
    $validation = $native.Output | ConvertFrom-Json
    if ($validation.status -ne "complete" -or $validation.source_mode -ne "kaggle_cpu_direct_competition_train") {
        throw "Downloaded Kaggle replay cache has the wrong source mode"
    }

    if (Test-Path -LiteralPath $canonicalManifest -PathType Leaf) {
        $native = Invoke-NativeOutput { & python $validator --root $canonicalRoot }
        if ($native.ExitCode -ne 0) {
            throw "Concurrent canonical replay cache failed validation: $($native.Output)"
        }
        Write-Terminal "unused_canonical_cache_ready" @{
            downloaded_manifest_sha256 = $validation.manifest_sha256
            canonical_validation = ($native.Output | ConvertFrom-Json)
        }
        exit 0
    }

    $stopped = Stop-ExactOvernightOrchestrator $ExistingOvernightOrchestratorPid
    if (Test-Path -LiteralPath $canonicalRoot) {
        $cacheParent = (Resolve-Path -LiteralPath (Split-Path -Parent $canonicalRoot)).Path
        if (-not $canonicalRoot.StartsWith($cacheParent, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Canonical replay path escaped its cache parent"
        }
        $quarantine = "$canonicalRoot.superseded-$([DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssZ'))"
        Move-Item -LiteralPath $canonicalRoot -Destination $quarantine
        Write-Log "quarantined partial canonical cache path=$quarantine"
    }
    Move-Item -LiteralPath $stagedRoot -Destination $canonicalRoot
    $native = Invoke-NativeOutput { & python $validator --root $canonicalRoot }
    if ($native.ExitCode -ne 0) {
        throw "Promoted Kaggle replay cache failed validation: $($native.Output)"
    }
    Write-Terminal "integrated_launcher_started" @{
        stopped_local_orchestrator = $stopped
        promoted_manifest_sha256 = $validation.manifest_sha256
        promoted_shards = $validation.shards
        promoted_bytes = $validation.bytes
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher
    if ($LASTEXITCODE -ne 0) {
        throw "AWS temporal-localizer launch/harvest controller failed after fallback integration"
    }
    Write-Terminal "integrated_launcher_completed" @{
        promoted_manifest_sha256 = $validation.manifest_sha256
        harvest_terminal = (Join-Path $stateRoot "harvest-terminal.json")
    }
}
catch {
    Write-Terminal "failed" @{
        error = $_.Exception.Message
        downloaded_output_preserved = (Test-Path -LiteralPath $downloadRoot)
        canonical_cache_preserved = (Test-Path -LiteralPath $canonicalRoot)
    }
    throw
}
