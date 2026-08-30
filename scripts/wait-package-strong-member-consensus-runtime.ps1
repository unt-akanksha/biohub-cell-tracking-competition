param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [string]$SklearnWheel = "C:/Users/IndarKumar/Documents/Comp/Biohub/.biohub/cache/sklearn-1.9.0-cp312-linux/scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 720
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 15 -or $MaximumPolls -lt 1) {
    throw "Invalid post-harvest wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot

$harvestRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-seed-probe-harvest-v1"
$harvestTerminal = Join-Path $harvestRoot "harvest-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/strong-member-consensus-post-harvest-v1"
$terminalPath = Join-Path $stateRoot "post-harvest-terminal.json"
$logPath = Join-Path $stateRoot "post-harvest.log"
$verificationPath = Join-Path $stateRoot "harvest-verification.json"
$baselinePath = Join-Path $stateRoot "development-baseline.json"
$developmentPath = Join-Path $stateRoot "strong-member-development-evidence.json"
$runtimeRoot = Join-Path $stateRoot "strong-member-consensus-division-v2"
if (Test-Path -LiteralPath $terminalPath) {
    throw "Strong-member post-harvest controller already reached a terminal state"
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "strong-member-consensus-post-harvest-v1"
    $Payload["competition_test_data_read"] = $false
    $Payload["public_leaderboard_used_for_selection"] = $false
    $Payload["competition_submission_performed"] = $false
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Invoke-LoggedPython([string[]]$Arguments) {
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    return $exitCode
}

$harvestReady = $false
for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
    if (Test-Path -LiteralPath $harvestTerminal -PathType Leaf) {
        $harvestReady = $true
        break
    }
    if ($poll -lt $MaximumPolls) {
        Start-Sleep -Seconds $PollSeconds
    }
}
if (-not $harvestReady) {
    Write-Terminal @{ status = "harvest_wait_timed_out"; runtime_created = $false }
    exit 3
}

$harvest = Get-Content -Raw -LiteralPath $harvestTerminal | ConvertFrom-Json
if ($harvest.status -ne "harvest_verified") {
    Write-Terminal @{
        status = "skipped_after_harvest_failure"
        harvest_status = $harvest.status
        runtime_created = $false
    }
    exit 4
}
$archivePath = [string]$harvest.archive_path
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Verified harvest archive is missing"
}
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLower()
if ($archiveHash -ne [string]$harvest.archive_sha256) {
    throw "Verified harvest archive changed after download"
}
$extractRoot = Join-Path $stateRoot "extracted-$($archiveHash.Substring(0, 16))"
$extractExit = Invoke-LoggedPython @(
    "scripts/verify-antelume-seed-probe-harvest.py",
    "--archive", $archivePath,
    "--report", $verificationPath,
    "--extract-to", $extractRoot
)
if ($extractExit -ne 0) {
    throw "Harvest verification/extraction failed"
}
$harvestManifest = Get-Content -Raw -LiteralPath (Join-Path $extractRoot "HARVEST_MANIFEST.json") | ConvertFrom-Json
if ($harvestManifest.probe_controller_status -ne "completed") {
    Write-Terminal @{
        status = "skipped_after_selection_rejection"
        probe_controller_status = $harvestManifest.probe_controller_status
        archive_sha256 = $archiveHash
        extracted_to = $extractRoot
        runtime_created = $false
    }
    exit 0
}
if ([int]$harvestManifest.member_count -lt 1) {
    throw "Completed probe harvest contains no selected strong member"
}

$probePath = Join-Path $extractRoot "probe/seed_ensemble_probe.json"
$selectionPath = Join-Path $extractRoot "ensemble/seed_ensemble_terminal.json"
$sweepRoot = Join-Path $extractRoot "sweep"
$morphologyRoot = Join-Path $RepositoryRoot ".biohub/results/competition-real-handcrafted-division-gate-v1"
$morphologyProbe = Join-Path $morphologyRoot "handcrafted_division_probe.json"
$predictionRoot = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
$truthRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$legacyDeepProbe = Join-Path $RepositoryRoot ".biohub/results/competition-real-division-gate-focused-v1/probe.json"
$archivedEvidence = Join-Path $RepositoryRoot ".biohub/results/competition-ranked-consensus-division-development-v1.json"
$replayEvidence = Join-Path $RepositoryRoot ".biohub/cache/analysis/ranked-consensus-ema0940-replay-20260830.json"

$baselineExit = Invoke-LoggedPython @(
    "scripts/verify-ranked-consensus-development-baseline.py",
    "--prediction-root", $predictionRoot,
    "--truth-root", $truthRoot,
    "--deep-probe", $legacyDeepProbe,
    "--morphology-probe", $morphologyProbe,
    "--evaluator-source", "research/evaluate_ranked_consensus_division_recovery.py",
    "--archived-evidence", $archivedEvidence,
    "--replay-evidence", $replayEvidence,
    "--output", $baselinePath
)
if ($baselineExit -ne 0) {
    throw "Exact development baseline verification failed"
}

$developmentExit = Invoke-LoggedPython @(
    "research/evaluate_ranked_consensus_division_recovery.py",
    "--deep-probe", $probePath,
    "--morphology-probe", $morphologyProbe,
    "--prediction-root", $predictionRoot,
    "--truth-root", $truthRoot,
    "--output", $developmentPath
)
if ($developmentExit -ne 0) {
    $developmentStatus = "evaluation_failed"
    if (Test-Path -LiteralPath $developmentPath -PathType Leaf) {
        $development = Get-Content -Raw -LiteralPath $developmentPath | ConvertFrom-Json
        $developmentStatus = [string]$development.status
    }
    Write-Terminal @{
        status = "development_rejected"
        development_status = $developmentStatus
        development_evidence = $developmentPath
        archive_sha256 = $archiveHash
        runtime_created = $false
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $SklearnWheel -PathType Leaf)) {
    throw "Pinned scikit-learn wheel is missing"
}
$runtimeExit = Invoke-LoggedPython @(
    "scripts/build-strong-member-consensus-division-dataset.py",
    "--sweep-root", $sweepRoot,
    "--ensemble-terminal", $selectionPath,
    "--probe-output", $probePath,
    "--morphology-root", $morphologyRoot,
    "--development-evidence", $developmentPath,
    "--sklearn-wheel", $SklearnWheel,
    "--output-root", $runtimeRoot
)
if ($runtimeExit -ne 0) {
    throw "Strong-member runtime packaging failed"
}
$manifestPath = Join-Path $runtimeRoot "STRONG_MEMBER_CONSENSUS_MANIFEST.json"
$manifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLower()
Write-Terminal @{
    status = "runtime_packaged"
    archive_sha256 = $archiveHash
    extracted_to = $extractRoot
    development_evidence = $developmentPath
    development_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentPath).Hash.ToLower()
    runtime_root = $runtimeRoot
    runtime_manifest = $manifestPath
    runtime_manifest_sha256 = $manifestHash
    runtime_created = $true
}
