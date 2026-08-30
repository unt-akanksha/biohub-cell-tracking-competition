param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 1440,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 15 -or $MaximumPolls -lt 1) { throw "Invalid relational development wait bounds" }
Set-Location -LiteralPath $RepositoryRoot
$harvestRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-relational-division-harvest-v1"
$harvestTerminal = Join-Path $harvestRoot "harvest-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/relational-division-development-v1"
$terminalPath = Join-Path $stateRoot "development-terminal.json"
$verificationPath = Join-Path $stateRoot "harvest-verification.json"
$developmentPath = Join-Path $stateRoot "relational-development-evidence.json"
$logPath = Join-Path $stateRoot "development.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-relational-division-harvest.py"
$evaluator = Join-Path $RepositoryRoot "research/evaluate_relational_division_development.py"
$runtimeBuilder = Join-Path $RepositoryRoot "scripts/build-relational-consensus-division-dataset.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$morphologyRoot = Join-Path $RepositoryRoot ".biohub/results/competition-real-handcrafted-division-gate-v1"
$morphologyProbe = Join-Path $morphologyRoot "handcrafted_division_probe.json"
$predictionRoot = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
$truthRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$baselinePath = Join-Path $RepositoryRoot ".biohub/results/competition-ranked-consensus-development-baseline-v1.json"
$sklearnWheel = Join-Path $RepositoryRoot ".biohub/cache/sklearn-1.9.0-cp312-linux/scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
$runtimeRoot = Join-Path $stateRoot "relational-consensus-division-v1"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "relational-division-development-controller-v1"
    $Payload["competition_test_data_read"] = $false
    $Payload["public_leaderboard_used_for_selection"] = $false
    $Payload["competition_submission_performed"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Invoke-Logged([string]$Program, [string[]]$Arguments) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
    $status = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    return $status
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if ($ValidateOnly) {
    foreach ($path in @($verifier, $evaluator, $runtimeBuilder, $evaluationPython, $morphologyRoot, $morphologyProbe, $predictionRoot, $truthRoot, $baselinePath, $sklearnWheel)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Relational development input is missing: $path" }
    }
    & python -m py_compile $verifier $evaluator $runtimeBuilder
    if ($LASTEXITCODE -ne 0) { throw "Relational development source validation failed" }
    @{ status = "validated"; stage = "relational_division_development" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) { throw "Relational development controller already reached a terminal state" }

try {
    $ready = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        if (Test-Path -LiteralPath $harvestTerminal -PathType Leaf) { $ready = $true; break }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $ready) { throw "Timed out waiting for relational harvest" }
    $harvest = Get-Content -Raw -LiteralPath $harvestTerminal | ConvertFrom-Json
    if ($harvest.status -ne "harvest_verified") {
        Write-Terminal @{ status = "skipped_after_harvest_failure"; harvest_status = $harvest.status; authorized_for_full_candidate_evaluation = $false }
        exit 0
    }
    $archivePath = Join-Path $harvestRoot "relational-division-results.tar.gz"
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) { throw "Verified relational harvest archive is missing" }
    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($archiveHash -ne [string]$harvest.archive_sha256) { throw "Relational harvest changed after verification" }
    $extractRoot = Join-Path $stateRoot "extracted-$($archiveHash.Substring(0, 16))"
    $extractStatus = Invoke-Logged "python" @($verifier, "--archive", $archivePath, "--report", $verificationPath, "--extract-to", $extractRoot)
    if ($extractStatus -ne 0) { throw "Relational harvest extraction failed" }
    $resultRoot = Join-Path $extractRoot "competition-relational-division-sweep-v1"
    $probePath = Join-Path $resultRoot "relational_development_probe.json"
    if ([int]$harvest.training_exit_code -ne 0 -or -not (Test-Path -LiteralPath $probePath -PathType Leaf)) {
        Write-Terminal @{
            status = "skipped_after_relational_rejection"
            training_exit_code = [int]$harvest.training_exit_code
            development_probe_present = Test-Path -LiteralPath $probePath -PathType Leaf
            archive_sha256 = $archiveHash
            extracted_to = $extractRoot
            authorized_for_full_candidate_evaluation = $false
        }
        exit 0
    }
    $evaluationStatus = Invoke-Logged $evaluationPython @(
        $evaluator,
        "--relational-probe", $probePath,
        "--morphology-probe", $morphologyProbe,
        "--prediction-root", $predictionRoot,
        "--truth-root", $truthRoot,
        "--baseline", $baselinePath,
        "--output", $developmentPath
    )
    $development = Get-Content -Raw -LiteralPath $developmentPath | ConvertFrom-Json
    if ($evaluationStatus -ne 0 -or $development.status -ne "development_positive") {
        Write-Terminal @{
            status = "development_rejected"
            development_status = $development.status
            development_evidence = $developmentPath
            development_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentPath).Hash.ToLowerInvariant()
            archive_sha256 = $archiveHash
            extracted_to = $extractRoot
            authorized_for_full_candidate_evaluation = $false
        }
        exit 0
    }
    $runtimeStatus = Invoke-Logged "python" @(
        $runtimeBuilder,
        "--results-root", (Join-Path $resultRoot "models"),
        "--probe", $probePath,
        "--morphology-root", $morphologyRoot,
        "--development-evidence", $developmentPath,
        "--sklearn-wheel", $sklearnWheel,
        "--output-root", $runtimeRoot
    )
    if ($runtimeStatus -ne 0) { throw "Relational consensus runtime packaging failed" }
    $runtimeManifest = Join-Path $runtimeRoot "RELATIONAL_CONSENSUS_MANIFEST.json"
    Write-Terminal @{
        status = "runtime_packaged"
        development_evidence = $developmentPath
        development_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentPath).Hash.ToLowerInvariant()
        relational_probe = $probePath
        relational_probe_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $probePath).Hash.ToLowerInvariant()
        archive_sha256 = $archiveHash
        extracted_to = $extractRoot
        selected = [int]$development.selected
        true_positives = [int]$development.tp
        false_positives = [int]$development.fp
        runtime_root = $runtimeRoot
        runtime_manifest = $runtimeManifest
        runtime_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeManifest).Hash.ToLowerInvariant()
        runtime_created = $true
        authorized_for_full_candidate_evaluation = $true
        authorized_for_submission = $false
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message; runtime_created = $false; authorized_for_full_candidate_evaluation = $false; authorized_for_submission = $false }
    throw
}
