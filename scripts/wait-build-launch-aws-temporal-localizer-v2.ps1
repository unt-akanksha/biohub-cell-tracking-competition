param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$InitialKaggleCooldownSeconds = 900,
    [int]$RetryCooldownSeconds = 900,
    [int]$MaximumDownloadAttempts = 48
)

$ErrorActionPreference = "Stop"
if (
    $InitialKaggleCooldownSeconds -lt 120 -or
    $RetryCooldownSeconds -lt 120 -or
    $MaximumDownloadAttempts -lt 1
) {
    throw "Invalid real-replay orchestration bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/aws-4gpu-temporal-localizer-v1"
$inventoryPath = Join-Path $RepositoryRoot ".biohub/cache/analysis/competition-real-localization-inventory-v1.json"
$inventorySha256 = "55159ef0636d49fcc31eea6d5fe9c327be59c2813d6d0083d6cdfabc9f6112e1"
$frameRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-replay-v1"
$frameManifest = Join-Path $frameRoot "real_localization_frame_cache_manifest.json"
$geffRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-train-geffs-packed-v1"
$shardRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-shards-v1"
$shardManifest = Join-Path $shardRoot "real_localization_shard_manifest.json"
$downloader = Join-Path $RepositoryRoot "scripts/download-competition-real-localization-replay.py"
$builder = Join-Path $RepositoryRoot "research/build_competition_real_localization_shards.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$launcher = Join-Path $RepositoryRoot "scripts/wait-launch-harvest-aws-4gpu-temporal-localizer-v1.ps1"
$terminalPath = Join-Path $stateRoot "overnight-orchestrator-terminal.json"

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "aws-real-replay-temporal-localizer-overnight-orchestrator-v2"
        status = $Status
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
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

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
foreach ($required in @(
    $inventoryPath, $downloader, $builder, $evaluationPython, $launcher
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Overnight orchestration input is missing: $required"
    }
}
if (-not (Test-Path -LiteralPath $geffRoot -PathType Container)) {
    throw "Hash-bound competition train GEFF cache is missing"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inventoryPath).Hash.ToLowerInvariant() -ne $inventorySha256) {
    throw "Real localization inventory changed"
}

try {
    if (-not (Test-Path -LiteralPath $frameManifest -PathType Leaf)) {
        Start-Sleep -Seconds $InitialKaggleCooldownSeconds
        $downloadComplete = $false
        for ($attempt = 1; $attempt -le $MaximumDownloadAttempts; $attempt++) {
            & python $downloader `
                --inventory $inventoryPath `
                --expected-inventory-sha256 $inventorySha256 `
                --output-root $frameRoot `
                --request-delay-seconds 5.0
            if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $frameManifest -PathType Leaf)) {
                $downloadComplete = $true
                break
            }
            if ($attempt -lt $MaximumDownloadAttempts) {
                Start-Sleep -Seconds $RetryCooldownSeconds
            }
        }
        if (-not $downloadComplete) {
            throw "Real localization replay did not complete within the bounded retries"
        }
    }

    if (-not (Test-Path -LiteralPath $shardManifest -PathType Leaf)) {
        if (Test-Path -LiteralPath $shardRoot) {
            throw "Partial real localization shard root requires inspection"
        }
        & $evaluationPython $builder `
            --inventory $inventoryPath `
            --inventory-sha256 $inventorySha256 `
            --frame-root $frameRoot `
            --geff-cache-root $geffRoot `
            --output-root $shardRoot
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $shardManifest -PathType Leaf)) {
            throw "Real localization replay shard build failed"
        }
    }

    Write-Terminal "aws_credential_wait_armed" @{
        frame_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $frameManifest).Hash.ToLowerInvariant()
        shard_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $shardManifest).Hash.ToLowerInvariant()
        planned_instance_type = "g5.12xlarge"
        planned_gpu_count = 4
        planned_model_count = 4
        parameters_per_model = 71249805
        steps_per_model = 40000
        real_replay_probability = 0.25
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher
    if ($LASTEXITCODE -ne 0) {
        throw "AWS temporal-localizer launch/harvest controller failed"
    }
    Write-Terminal "harvest_controller_completed" @{
        launcher = $launcher
        harvest_terminal = (Join-Path $stateRoot "harvest-terminal.json")
    }
}
catch {
    Write-Terminal "failed" @{
        error = $_.Exception.Message
        partial_frame_cache_preserved = (Test-Path -LiteralPath $frameRoot)
        recoverable_shard_root_preserved = (Test-Path -LiteralPath $shardRoot)
    }
    throw
}
