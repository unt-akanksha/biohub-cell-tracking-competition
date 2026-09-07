param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 300,
    [int]$MaximumPolls = 2880,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 30 -or $MaximumPolls -lt 1) {
    throw "Invalid graph-context v2 development wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$harvestRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2"
$harvestTerminal = Join-Path $harvestRoot "harvest-terminal.json"
$verificationTerminal = Join-Path $harvestRoot "verification-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/graph-context-frozen-ensemble-development-v2"
$terminalPath = Join-Path $stateRoot "development-terminal.json"
$developmentPath = Join-Path $stateRoot "graph-context-development-evidence.json"
$logPath = Join-Path $stateRoot "development.log"
$evaluator = Join-Path $RepositoryRoot "research/evaluate_graph_context_division_development.py"
$runtimeBuilder = Join-Path $RepositoryRoot "scripts/build-graph-context-consensus-division-dataset.py"
$evaluationPython = Join-Path $RepositoryRoot ".biohub/evaluation-venv/Scripts/python.exe"
$morphologyRoot = Join-Path $RepositoryRoot ".biohub/results/competition-real-handcrafted-division-gate-v1"
$morphologyProbe = Join-Path $morphologyRoot "handcrafted_division_probe.json"
$predictionRoot = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
$truthRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$baselinePath = Join-Path $RepositoryRoot ".biohub/results/competition-ranked-consensus-development-baseline-v1.json"
$sklearnWheel = Join-Path $RepositoryRoot ".biohub/cache/sklearn-1.9.0-cp312-linux/scikit_learn-1.9.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
$runtimeRoot = Join-Path $RepositoryRoot ".biohub/cache/runtime-datasets/biohub-graph-context-consensus-division-v2"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "graph-context-frozen-ensemble-development-controller-v2"
    $Payload["competition_test_data_read"] = $false
    $Payload["public_leaderboard_used_for_selection"] = $false
    $Payload["competition_submission_performed"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
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
foreach ($required in @(
    $evaluator, $runtimeBuilder, $evaluationPython, $morphologyRoot,
    $morphologyProbe, $predictionRoot, $truthRoot, $baselinePath, $sklearnWheel
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Graph-context v2 development input is missing: $required"
    }
}
& python -m py_compile $evaluator $runtimeBuilder
if ($LASTEXITCODE -ne 0) { throw "Graph-context v2 development preflight failed" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "graph_context_frozen_ensemble_development_v2" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Graph-context v2 development already reached a terminal state"
}

try {
    $ready = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        if (Test-Path -LiteralPath $harvestTerminal -PathType Leaf) {
            $ready = $true
            break
        }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $ready) { throw "Timed out waiting for graph-context v2 harvest" }
    $harvest = Get-Content -Raw -LiteralPath $harvestTerminal | ConvertFrom-Json
    if ($harvest.status -eq "scientifically_rejected") {
        Write-Terminal @{
            status = "skipped_after_graph_context_rejection"
            harvest_status = $harvest.status
            authorized_for_full_candidate_evaluation = $false
            authorized_for_submission = $false
        }
        exit 0
    }
    if ($harvest.status -ne "accepted" -or -not (Test-Path -LiteralPath $verificationTerminal -PathType Leaf)) {
        throw "Graph-context v2 harvest was not accepted and verified"
    }
    $verification = Get-Content -Raw -LiteralPath $verificationTerminal | ConvertFrom-Json
    if ($verification.status -ne "accepted" -or [int]$verification.training_exit_code -ne 0) {
        throw "Graph-context v2 verification terminal is ineligible"
    }
    $resultRoot = [string]$harvest.extracted_to
    $probePath = Join-Path $resultRoot "graph_context_development_probe.json"
    $resultsRoot = Join-Path $resultRoot "models"
    foreach ($required in @($probePath, $resultsRoot)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Accepted graph-context v2 artifact is missing: $required"
        }
    }
    $evaluationStatus = Invoke-Logged $evaluationPython @(
        $evaluator,
        "--graph-context-probe", $probePath,
        "--morphology-probe", $morphologyProbe,
        "--prediction-root", $predictionRoot,
        "--truth-root", $truthRoot,
        "--baseline", $baselinePath,
        "--output", $developmentPath
    )
    if (-not (Test-Path -LiteralPath $developmentPath -PathType Leaf)) {
        throw "Graph-context v2 development evaluator produced no evidence"
    }
    $development = Get-Content -Raw -LiteralPath $developmentPath | ConvertFrom-Json
    if ($evaluationStatus -ne 0 -or $development.status -ne "development_positive") {
        Write-Terminal @{
            status = "development_rejected"
            development_status = $development.status
            development_evidence = $developmentPath
            development_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentPath).Hash.ToLowerInvariant()
            authorized_for_full_candidate_evaluation = $false
            authorized_for_submission = $false
        }
        exit 0
    }
    $runtimeStatus = Invoke-Logged "python" @(
        $runtimeBuilder,
        "--results-root", $resultsRoot,
        "--probe", $probePath,
        "--morphology-root", $morphologyRoot,
        "--development-evidence", $developmentPath,
        "--sklearn-wheel", $sklearnWheel,
        "--output-root", $runtimeRoot
    )
    if ($runtimeStatus -ne 0) { throw "Graph-context v2 runtime packaging failed" }
    $runtimeManifest = Join-Path $runtimeRoot "GRAPH_CONTEXT_CONSENSUS_MANIFEST.json"
    Write-Terminal @{
        status = "runtime_packaged"
        development_evidence = $developmentPath
        development_evidence_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentPath).Hash.ToLowerInvariant()
        graph_context_probe = $probePath
        graph_context_probe_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $probePath).Hash.ToLowerInvariant()
        archive_sha256 = $harvest.archive_sha256
        extracted_to = $resultRoot
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
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        runtime_created = $false
        authorized_for_full_candidate_evaluation = $false
        authorized_for_submission = $false
    }
    throw
}
