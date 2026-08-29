param(
    [double]$MaximumWaitHours = 48.0,
    [int]$PollSeconds = 300,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$stagingParent = Join-Path $projectRoot '.biohub/staging'
$runtimeRoot = Join-Path $stagingParent 'biohub-external-division-policy-runtime-v1-20260829'
$kernelDir = Join-Path $projectRoot 'kaggle/biohub-external-division-policy-v1'
$pretrainTerminal = Join-Path $automationDir 'zebrahub-multiscale-pretrain-recovery.json'
$publishedPolicy = Join-Path $stagingParent 'external-division-recovery-policy-v1.json'
$runtimeBuilder = Join-Path $PSScriptRoot 'build-external-division-policy-runtime.py'
$kernelBuilder = Join-Path $PSScriptRoot 'build-external-division-policy-kernel.py'
$outputVerifier = Join-Path $PSScriptRoot 'verify-external-division-policy-output.py'
$kernelStateScript = Join-Path $PSScriptRoot 'get-kaggle-kernel-state.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$runtimeRef = 'indarkarhana/biohub-external-division-policy-runtime-v1'
$kernelRef = 'indarkarhana/biohub-external-division-policy-v1'
$pretrainRef = 'indarkarhana/biohub-zebrahub-multiscale-pretrain-v1'
$logPath = Join-Path $automationDir 'kaggle-external-division-policy-controller.log'
$terminalPath = Join-Path $automationDir 'kaggle-external-division-policy-controller.json'
$verificationPath = Join-Path $automationDir 'kaggle-external-division-policy-verification.json'
$verificationError = Join-Path $automationDir 'kaggle-external-division-policy-verification.stderr.log'

function Write-PolicyLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-PolicyTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'kaggle-external-division-policy-controller-v1'
        status = $Status
        target_public_score = 0.945
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        pretrain_ref = $pretrainRef
        competition_data_read = $false
        public_leaderboard_used_for_selection = $false
        submission_created = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Assert-LocalContract {
    if ($PollSeconds -lt 60) {
        throw 'PollSeconds must be at least 60'
    }
    foreach ($required in @(
        $runtimeBuilder,
        $kernelBuilder,
        $outputVerifier,
        $kernelStateScript,
        $evaluationPython
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required Kaggle fallback input is missing: $required"
        }
    }
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        throw 'Kaggle CLI is unavailable'
    }
}

New-Item -ItemType Directory -Force -Path $automationDir, $stagingParent | Out-Null
Assert-LocalContract
if ($ValidateOnly) {
    [ordered]@{
        status = 'valid'
        run_id = 'kaggle-external-division-policy-controller-v1'
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition_data_read = $false
        submission_created = $false
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Refusing to reuse Kaggle fallback terminal: $terminalPath"
}

try {
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    if (Test-Path -LiteralPath $runtimeRoot) {
        throw "Refusing to reuse external policy runtime staging: $runtimeRoot"
    }
    $runtimeOutput = (& $evaluationPython $runtimeBuilder --output-root $runtimeRoot 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "External policy runtime build failed: $runtimeOutput"
    }
    Write-PolicyLog "runtime_built output=$runtimeOutput"

    $datasetList = (& kaggle datasets list --mine -s biohub-external-division-policy-runtime-v1 --format json 2>&1) -join "`n"
    if ($datasetList -match [regex]::Escape($runtimeRef)) {
        $datasetOutput = (& kaggle datasets version -p $runtimeRoot -m 'External division policy calibration runtime v1' 2>&1) -join "`n"
        $datasetOperation = 'versioned'
    } else {
        $datasetOutput = (& kaggle datasets create -p $runtimeRoot 2>&1) -join "`n"
        $datasetOperation = 'created'
        if ($LASTEXITCODE -ne 0 -and $datasetOutput -match '(?i)already exists|conflict') {
            $datasetOutput = (& kaggle datasets version -p $runtimeRoot -m 'External division policy calibration runtime v1' 2>&1) -join "`n"
            $datasetOperation = 'versioned_after_create_conflict'
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "External policy runtime upload failed: $datasetOutput"
    }
    Write-PolicyLog "runtime_uploaded operation=$datasetOperation output=$datasetOutput"

    $datasetReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $datasetStatus = (& kaggle datasets status $runtimeRef --format json 2>&1) -join "`n"
        Write-PolicyLog "runtime_status output=$datasetStatus"
        if ($LASTEXITCODE -eq 0 -and $datasetStatus -match '(?i)ready|complete') {
            $datasetReady = $true
            break
        }
        if ($datasetStatus -match '(?i)error|failed') {
            throw "External policy runtime processing failed: $datasetStatus"
        }
        Start-Sleep -Seconds 60
    }
    if (-not $datasetReady) {
        throw 'Timed out waiting for external policy runtime readiness'
    }

    $pretrainVerified = $false
    $poll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $poll += 1
        if (Test-Path -LiteralPath $pretrainTerminal -PathType Leaf) {
            $pretrain = Get-Content -Raw -LiteralPath $pretrainTerminal | ConvertFrom-Json
            if ($pretrain.status -eq 'verified') {
                $pretrainVerified = $true
                break
            }
            if ($pretrain.status -eq 'failed') {
                throw 'V4 pretraining recovery failed before Kaggle policy fallback'
            }
        }
        Write-PolicyLog "waiting_for_verified_v4 poll=$poll"
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $pretrainVerified) {
        throw 'Timed out waiting for verified v4 pretraining'
    }

    if (
        (Test-Path -LiteralPath $kernelDir) -and
        @(Get-ChildItem -LiteralPath $kernelDir -Force).Count -gt 0
    ) {
        throw "Refusing to overwrite external policy kernel: $kernelDir"
    }
    $kernelBuild = (& $evaluationPython $kernelBuilder --runtime-root $runtimeRoot 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "External policy kernel build failed: $kernelBuild"
    }
    Write-PolicyLog "kernel_built output=$kernelBuild"

    $pushOutput = (& kaggle kernels push -p $kernelDir 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0 -or $pushOutput -notmatch '(?i)successfully pushed') {
        throw "External policy kernel push failed: $pushOutput"
    }
    Write-PolicyLog "kernel_pushed output=$pushOutput"
    Start-Sleep -Seconds 10
    $stateText = (& $evaluationPython $kernelStateScript --kernel-slug $kernelRef 2>&1) -join "`n"
    $state = $stateText | ConvertFrom-Json
    $kernelVersion = [int]$state.current_version_number
    if (
        $state.present -ne $true -or
        $kernelVersion -lt 1 -or
        $state.is_private -ne $true -or
        $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or
        $state.enable_internet -ne $false -or
        $state.dataset_sources -notcontains $runtimeRef -or
        $state.dataset_sources -notcontains 'indarkarhana/biohub-zebrahub-contextual-shards-v1' -or
        $state.kernel_sources -notcontains $pretrainRef -or
        @($state.competition_sources).Count -ne 0
    ) {
        throw "External policy remote kernel state is invalid: $stateText"
    }

    $complete = $false
    $kernelPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $kernelPoll += 1
        $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
        Write-PolicyLog "kernel_status poll=$kernelPoll version=$kernelVersion output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            $complete = $true
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "External policy kernel did not complete: $statusOutput"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) {
        throw 'Timed out waiting for external policy kernel completion'
    }

    $downloadRoot = Join-Path $projectRoot ".biohub/cache/kernel-outputs/external-division-policy-v1-20260829-v$kernelVersion"
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Refusing to reuse external policy output root: $downloadRoot"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $downloadOutput = (& kaggle kernels output $kernelRef -p $downloadRoot --force 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "External policy output download failed: $downloadOutput"
    }
    Write-PolicyLog "output_downloaded output=$downloadOutput"

    & $evaluationPython $outputVerifier --output-root $downloadRoot --report $verificationPath --publish-policy $publishedPolicy 1> (Join-Path $automationDir 'kaggle-external-division-policy-verification.stdout.log') 2> $verificationError
    if ($LASTEXITCODE -ne 0) {
        throw "External policy output verification failed; see $verificationError"
    }
    $verification = Get-Content -Raw -LiteralPath $verificationPath | ConvertFrom-Json
    if ($verification.status -ne 'verified' -or $verification.authorized_for_candidate_handoff -ne $true) {
        throw 'External policy verification report is invalid'
    }
    Write-PolicyTerminal 'verified_and_published' @{
        kernel_version = $kernelVersion
        download_root = $downloadRoot
        policy_sha256 = $verification.policy_sha256
        frozen_division_logit_threshold = $verification.frozen_division_logit_threshold
        verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verificationPath).Hash.ToLowerInvariant()
        published_policy = $publishedPolicy
    }
    Write-PolicyLog "verified_and_published version=$kernelVersion policy_sha256=$($verification.policy_sha256)"
    exit 0
} catch {
    Write-PolicyTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-PolicyLog "failed error=$($_.Exception.Message)"
    exit 1
}
