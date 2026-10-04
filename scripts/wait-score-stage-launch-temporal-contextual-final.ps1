param(
    [double]$MinimumQuotaHours = 12.0,
    [double]$MaximumWaitHours = 96.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$processedLaunchTerminal = Join-Path $automationDir 'temporal-contextual-processed-launch.json'
$logPath = Join-Path $automationDir 'temporal-contextual-final-launch.log'
$terminalPath = Join-Path $automationDir 'temporal-contextual-final-launch.json'
$processedRef = 'indarkarhana/biohub-temporal-contextual-processed-acceptance-v3'
$acceptanceDatasetRef = 'indarkarhana/biohub-temporal-contextual-exact-acceptance-v3'
$finalRuntimeRef = 'indarkarhana/biohub-temporal-contextual-final-runtime-v1'
$finalRef = 'indarkarhana/biohub-temporal-contextual-submission-candidate-v3'
$processedDownloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-processed-v3-autochain'
$appearanceRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain'
$exactEvidence = Join-Path $automationDir 'temporal-contextual-exact-acceptance.json'
$exactLog = Join-Path $automationDir 'temporal-contextual-exact-acceptance.stdout.log'
$exactError = Join-Path $automationDir 'temporal-contextual-exact-acceptance.stderr.log'
$stagingRoot = Join-Path $automationDir 'temporal-contextual-exact-staging'
$acceptanceDatasetDir = Join-Path $stagingRoot 'biohub-temporal-contextual-exact-acceptance-v3'
$preflightOutput = Join-Path $automationDir 'temporal-contextual-final-preflight.json'
$preflightLog = Join-Path $automationDir 'temporal-contextual-final-preflight.stdout.log'
$preflightError = Join-Path $automationDir 'temporal-contextual-final-preflight.stderr.log'
$exactRunner = Join-Path $projectRoot 'scripts/run-temporal-contextual-exact-acceptance.py'
$stageScript = Join-Path $projectRoot 'scripts/stage-temporal-contextual-kaggle-artifacts.py'
$preflightScript = Join-Path $projectRoot 'scripts/build-temporal-contextual-submission-candidate-preflight.py'
$kernelStateScript = Join-Path $projectRoot 'scripts/get-kaggle-kernel-state.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$controlCsv = Join-Path $projectRoot '.biohub/cache/datasets/biohub-hoct-processed-validation-v1-version1/processed_validation.csv'
$truthDir = Join-Path $projectRoot '.biohub/cache/competition-truth/public-node-acceptance-v1'
$scorerLock = Join-Path $projectRoot 'config/official-scorer.lock.json'
$organizerCheckout = Join-Path $projectRoot '.biohub/vendor/kaggle-cell-tracking-competition'
$tracksdataCheckout = Join-Path $projectRoot '.biohub/vendor/tracksdata'
$runtimeRoot = Join-Path $projectRoot '.biohub/cache/dataset-redownloads/biohub-temporal-contextual-final-runtime-v1-version2'
$finalDir = Join-Path $projectRoot 'kaggle/biohub-temporal-contextual-submission-candidate-v3'
$finalMetadata = Join-Path $finalDir 'kernel-metadata.json'
$finalNotebook = Join-Path $finalDir 'biohub-temporal-contextual-submission-candidate-v3.ipynb'
$expectedControlSha256 = '6613545843ebd743dac66b5a0598702faaa5b3c0870566e55fa60250a009615b'
$expectedExactRunnerSha256 = 'dafd6fb978f1abc8ad56612113ec480a54cbd490bc12ff2ba88c4dae8133012f'
$expectedStageScriptSha256 = '1bd2319310c2e24c637b0d2b0d909fe11f756af7d2e5e450cafc614fee4f98fa'
$expectedPreflightScriptSha256 = 'b785a75f088c68ede25359a652510126a49a1758c9fb729ce956183576147b0f'
$expectedFinalMetadataSha256 = 'ef010e45a6a10d1f00efee2d696a8c5a218c52b29e039673b32aa128ff248f43'
$expectedFinalNotebookSha256 = 'f5a5e40827c0cd1b846dad0702faa7ae3697288f59c7ce24e8da7797d179db01'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'temporal-contextual-pair-fusion-candidate-v3'
        status = $Status
        processed_ref = $processedRef
        acceptance_dataset_ref = $acceptanceDatasetRef
        final_runtime_ref = $finalRuntimeRef
        final_runtime_version = 2
        final_ref = $finalRef
        minimum_quota_hours = $MinimumQuotaHours
        quota_reserve_hours = 0.0
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        final_metadata_sha256 = $expectedFinalMetadataSha256
        final_notebook_sha256 = $expectedFinalNotebookSha256
        public_leaderboard_used_for_selection = $false
        competition_submission_performed = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label changed: $actual"
    }
}

try {
    Require-Hash $exactRunner $expectedExactRunnerSha256 'Exact acceptance runner'
    Require-Hash $stageScript $expectedStageScriptSha256 'Exact artifact staging script'
    Require-Hash $preflightScript $expectedPreflightScriptSha256 'Final preflight script'
    Require-Hash $finalMetadata $expectedFinalMetadataSha256 'Final metadata'
    Require-Hash $finalNotebook $expectedFinalNotebookSha256 'Final notebook'
    Require-Hash $controlCsv $expectedControlSha256 'Processed control CSV'
    foreach ($required in @(
        $evaluationPython,
        $truthDir,
        $scorerLock,
        $organizerCheckout,
        $tracksdataCheckout,
        $runtimeRoot
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Required exact-scoring input is missing: $required"
        }
    }
    $metadata = Get-Content -Raw -LiteralPath $finalMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $finalRef -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.dataset_sources -notcontains $finalRuntimeRef -or
        $metadata.dataset_sources -notcontains $acceptanceDatasetRef -or
        $metadata.kernel_sources -notcontains 'indarkarhana/biohub-temporal-contextual-transfer-v3' -or
        $metadata.kernel_sources -notcontains 'indarkarhana/biohub-trackastra-dual-fold-synthetic-v1' -or
        $metadata.competition_sources -notcontains 'biohub-cell-tracking-during-development'
    ) {
        throw 'Final candidate metadata contract is invalid'
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $processedLaunchTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the processed launch terminal'
        }
        Start-Sleep -Seconds 60
    }
    $processedLaunch = Get-Content -Raw -LiteralPath $processedLaunchTerminal | ConvertFrom-Json
    if (
        $processedLaunch.status -ne 'launched' -or
        $processedLaunch.processed_ref -ne $processedRef -or
        $processedLaunch.expected_gpu_count -ne 2 -or
        $processedLaunch.public_leaderboard_used_for_selection -ne $false -or
        $processedLaunch.submission_created -ne $false
    ) {
        throw 'Processed autolaunch did not produce valid launch evidence'
    }

    $externalPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($externalPoll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $externalPoll += 1
        $statusOutput = (& kaggle kernels status $processedRef 2>&1) -join "`n"
        Write-ChainLog "processed_status poll=$externalPoll output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Contextual processed stage did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for contextual processed completion'
    }

    if (Test-Path -LiteralPath $processedDownloadRoot) {
        if (-not (Test-Path -LiteralPath (Join-Path $processedDownloadRoot 'processed_launcher_terminal.json'))) {
            throw "Processed download target exists without expected launcher: $processedDownloadRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $processedDownloadRoot | Out-Null
        $downloadOutput = (& kaggle kernels output $processedRef -p $processedDownloadRoot --force 2>&1) -join "`n"
        Write-ChainLog "processed_download output=$downloadOutput"
    }

    if (Test-Path -LiteralPath $exactEvidence) {
        throw "Refusing to overwrite exact acceptance evidence: $exactEvidence"
    }
    & $evaluationPython $exactRunner --processed-root $processedDownloadRoot --control-csv $controlCsv --truth-dir $truthDir --scorer-lock $scorerLock --organizer-checkout $organizerCheckout --tracksdata-checkout $tracksdataCheckout --output $exactEvidence 1> $exactLog 2> $exactError
    if ($LASTEXITCODE -ne 0) {
        throw "Exact processed CPU acceptance failed; see $exactError"
    }
    $acceptance = Get-Content -Raw -LiteralPath $exactEvidence | ConvertFrom-Json
    if (
        $acceptance.status -ne 'accepted' -or
        $acceptance.run_id -ne 'trackastra-dual-fold-processed-exact-v1' -or
        $acceptance.exact_processed_gate_passed -ne $true -or
        $acceptance.authorized_for_submission -ne $false -or
        $acceptance.public_leaderboard_used_for_selection -ne $false -or
        $acceptance.competition_submission_performed -ne $false
    ) {
        throw 'Exact processed CPU acceptance evidence is invalid'
    }

    if (Test-Path -LiteralPath $stagingRoot) {
        throw "Refusing to reuse exact acceptance staging root: $stagingRoot"
    }
    $stageOutput = (& python $stageScript --appearance-root $appearanceRoot --acceptance-evidence $exactEvidence --staging-root $stagingRoot 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Exact acceptance dataset staging failed: $stageOutput"
    }
    Write-ChainLog "acceptance_staged output=$stageOutput"

    & python $preflightScript --appearance-root $appearanceRoot --acceptance-root $acceptanceDatasetDir --processed-root $processedDownloadRoot --runtime-root $runtimeRoot --output $preflightOutput 1> $preflightLog 2> $preflightError
    if ($LASTEXITCODE -ne 0) {
        throw "Final candidate preflight failed; see $preflightError"
    }

    $listOutput = (& kaggle datasets list --mine -s biohub-temporal-contextual-exact-acceptance-v3 --format json 2>&1) -join "`n"
    Push-Location -LiteralPath $acceptanceDatasetDir
    try {
        if ($listOutput -match [regex]::Escape($acceptanceDatasetRef)) {
            $datasetOutput = (& kaggle datasets version -p . -m 'Exact clean processed acceptance for contextual v3' 2>&1) -join "`n"
            $datasetOperation = 'versioned'
        } else {
            $datasetOutput = (& kaggle datasets create -p . 2>&1) -join "`n"
            $datasetOperation = 'created'
            if ($LASTEXITCODE -ne 0 -and $datasetOutput -match '(?i)already exists|conflict') {
                $datasetOutput = (& kaggle datasets version -p . -m 'Exact clean processed acceptance for contextual v3' 2>&1) -join "`n"
                $datasetOperation = 'versioned_after_create_conflict'
            }
        }
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Exact acceptance dataset upload failed: $datasetOutput"
    }
    Write-ChainLog "acceptance_dataset operation=$datasetOperation output=$datasetOutput"

    $datasetReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $datasetStatus = (& kaggle datasets status $acceptanceDatasetRef --format json 2>&1) -join "`n"
        Write-ChainLog "acceptance_dataset_status output=$datasetStatus"
        if ($LASTEXITCODE -eq 0 -and $datasetStatus -match '(?i)ready|complete') {
            $datasetReady = $true
            break
        }
        if ($datasetStatus -match '(?i)error|failed') {
            throw "Exact acceptance dataset processing failed: $datasetStatus"
        }
        Start-Sleep -Seconds 60
    }
    if (-not $datasetReady) {
        throw 'Timed out waiting for exact acceptance dataset readiness'
    }

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-ChainLog "final_quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $finalDir 2>&1) -join "`n"
        Write-ChainLog "final_push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $finalStatus = (& kaggle kernels status $finalRef 2>&1) -join "`n"
            if ($finalStatus -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Final push returned success but status is invalid: $finalStatus"
            }
            $stateText = (& python $kernelStateScript --kernel-slug $finalRef 2>&1) -join "`n"
            $state = $stateText | ConvertFrom-Json
            $version = [int]$state.current_version_number
            if (
                $state.kernel_slug -ne $finalRef -or
                $state.present -ne $true -or
                $version -lt 1 -or
                $state.is_private -ne $true -or
                $state.enable_gpu -ne $true -or
                $state.enable_tpu -ne $false -or
                $state.enable_internet -ne $false
            ) {
                throw "Final remote kernel state is invalid: $stateText"
            }
            Write-ChainTerminal 'launched' @{
                quota_before_hours = $remaining
                exact_acceptance_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exactEvidence).Hash.ToLowerInvariant()
                preflight_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $preflightOutput).Hash.ToLowerInvariant()
                acceptance_dataset_operation = $datasetOperation
                acceptance_dataset_status = $datasetStatus
                kernel_version = $version
                kaggle_status = $finalStatus
                push_output = $pushOutput
            }
            Write-ChainLog "final_launched version=$version status=$finalStatus"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Final candidate push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch contextual final candidate'
} catch {
    Write-ChainTerminal 'failed' @{ error = $_.Exception.Message }
    Write-ChainLog "failed error=$($_.Exception.Message)"
    exit 1
}
