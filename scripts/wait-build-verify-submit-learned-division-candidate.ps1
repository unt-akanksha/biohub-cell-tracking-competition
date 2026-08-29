param(
    [double]$MaximumWaitHours = 48.0,
    [int]$PollSeconds = 300,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$stagingParent = Join-Path $projectRoot '.biohub/staging'
$policyPath = Join-Path $stagingParent 'external-division-recovery-policy-v1.json'
$runtimeRoot = Join-Path $stagingParent 'biohub-learned-division-recovery-runtime-v1-20260829'
$candidateDir = Join-Path $projectRoot 'kaggle/biohub-ema-learned-division-candidate-v1'
$baselineValidator = Join-Path $projectRoot '.biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv'
$runtimeBuilder = Join-Path $PSScriptRoot 'build-learned-division-recovery-runtime.py'
$candidateBuilder = Join-Path $PSScriptRoot 'build-learned-division-submission-candidate.py'
$candidateVerifier = Join-Path $PSScriptRoot 'verify-learned-division-submission-candidate.py'
$candidateSubmitter = Join-Path $PSScriptRoot 'submit-learned-division-candidate.py'
$kernelStateScript = Join-Path $PSScriptRoot 'get-kaggle-kernel-state.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$logPath = Join-Path $automationDir 'learned-division-candidate-controller.log'
$terminalPath = Join-Path $automationDir 'learned-division-candidate-controller.json'
$promotionPath = Join-Path $automationDir 'learned-division-candidate-promotion.json'
$promotionError = Join-Path $automationDir 'learned-division-candidate-promotion.stderr.log'
$receiptPath = Join-Path $automationDir 'learned-division-candidate-submission-receipt.json'
$runtimeRef = 'indarkarhana/biohub-learned-division-recovery-runtime-v1'
$kernelRef = 'indarkarhana/biohub-ema-learned-division-candidate-v1'
$competition = 'biohub-cell-tracking-during-development'
$awsProfile = '148971207977_InventoryOptimization-EC2-Access'
$awsRegion = 'us-east-1'
$instanceId = 'i-0d12195df0d3558f3'
$availabilityZone = 'us-east-1b'
$remoteHost = 'ubuntu@100.52.213.73'
$publicKey = 'C:/Users/IndarKumar/.ssh/rsna_ec2.pub'
$privateKey = 'C:/Users/IndarKumar/.ssh/rsna_ec2'
$remotePolicy = '/home/ubuntu/biohub-results/division-policy-v1/policy.json'

function Write-ControllerLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ControllerTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'learned-division-candidate-autonomous-chain-v1'
        status = $Status
        target_public_score = 0.945
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        competition = $competition
        public_control = 'grafael/biohub-ct-0940-ema'
        exact_public_replica_forbidden = $true
        public_metric_hack_forbidden = $true
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Invoke-NativeOutput([scriptblock]$Command) {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $lines = & $Command 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    [pscustomobject]@{
        ExitCode = $exitCode
        Output = ($lines -join "`n")
    }
}

function Assert-LocalContract {
    if ($PollSeconds -lt 60) {
        throw 'PollSeconds must be at least 60'
    }
    foreach ($required in @(
        $baselineValidator,
        $runtimeBuilder,
        $candidateBuilder,
        $candidateVerifier,
        $candidateSubmitter,
        $kernelStateScript,
        $evaluationPython,
        $publicKey,
        $privateKey
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required autonomous-chain input is missing: $required"
        }
    }
    foreach ($command in @('aws', 'ssh', 'scp', 'kaggle')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
}

function Grant-TemporarySshKey {
    $native = Invoke-NativeOutput {
        & aws ec2-instance-connect send-ssh-public-key `
            --profile $awsProfile `
            --region $awsRegion `
            --instance-id $instanceId `
            --availability-zone $availabilityZone `
            --instance-os-user ubuntu `
            --ssh-public-key "file://$publicKey"
    }
    $grantOutput = $native.Output
    if ($native.ExitCode -ne 0 -or $grantOutput -notmatch '(?i)success') {
        throw "EC2 Instance Connect key injection failed: $grantOutput"
    }
}

New-Item -ItemType Directory -Force -Path $automationDir, $stagingParent | Out-Null
Assert-LocalContract
if ($ValidateOnly) {
    [ordered]@{
        status = 'valid'
        run_id = 'learned-division-candidate-autonomous-chain-v1'
        target_public_score = 0.945
        runtime_ref = $runtimeRef
        kernel_ref = $kernelRef
        candidate_submission_requires_promotion_gate = $true
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Refusing to reuse autonomous-chain terminal: $terminalPath"
}
if (Test-Path -LiteralPath $receiptPath) {
    throw "Refusing to risk a duplicate submission: $receiptPath"
}

try {
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    $policyDownloaded = $false
    $poll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $poll += 1
        if (Test-Path -LiteralPath $policyPath -PathType Leaf) {
            $policyDownloaded = $true
            Write-ControllerLog "policy_discovered_locally poll=$poll"
            break
        }
        try {
            Grant-TemporarySshKey
            $native = Invoke-NativeOutput { & ssh -i $privateKey -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $remoteHost "test -f '$remotePolicy'" }
            if ($native.ExitCode -eq 0) {
                if (Test-Path -LiteralPath $policyPath) {
                    throw "Refusing to overwrite downloaded policy: $policyPath"
                }
                $native = Invoke-NativeOutput { & scp -i $privateKey -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$remoteHost`:$remotePolicy" $policyPath }
                if ($native.ExitCode -ne 0) {
                    throw "Antelume policy download failed: $($native.Output)"
                }
                $policyDownloaded = $true
                Write-ControllerLog "policy_downloaded poll=$poll"
                break
            }
            Write-ControllerLog "policy_not_ready poll=$poll"
        } catch {
            Write-ControllerLog "policy_poll_transient_failure poll=$poll error=$($_.Exception.Message)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $policyDownloaded) {
        throw 'Timed out waiting for the Antelume external division policy'
    }
    $policy = Get-Content -Raw -LiteralPath $policyPath | ConvertFrom-Json
    if ($policy.status -ne 'accepted') {
        Write-ControllerTerminal 'external_policy_rejected' @{
            policy_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $policyPath).Hash.ToLowerInvariant()
            submission_created = $false
            competition_submission_performed = $false
        }
        Write-ControllerLog 'external_policy_rejected'
        exit 4
    }

    if (Test-Path -LiteralPath $runtimeRoot) {
        throw "Refusing to reuse learned-division runtime staging: $runtimeRoot"
    }
    $native = Invoke-NativeOutput { & $evaluationPython $runtimeBuilder --policy $policyPath --output-root $runtimeRoot }
    $runtimeOutput = $native.Output
    if ($native.ExitCode -ne 0) {
        throw "Learned-division runtime build failed: $runtimeOutput"
    }
    Write-ControllerLog "runtime_built output=$runtimeOutput"

    $native = Invoke-NativeOutput { & kaggle datasets list --mine -s biohub-learned-division-recovery-runtime-v1 --format json }
    $datasetList = $native.Output
    if ($datasetList -match [regex]::Escape($runtimeRef)) {
        $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m 'Frozen external division recovery policy v1' }
        $datasetOutput = $native.Output
        $datasetOperation = 'versioned'
    } else {
        $native = Invoke-NativeOutput { & kaggle datasets create -p $runtimeRoot }
        $datasetOutput = $native.Output
        $datasetOperation = 'created'
        if ($native.ExitCode -ne 0 -and $datasetOutput -match '(?i)already exists|conflict') {
            $native = Invoke-NativeOutput { & kaggle datasets version -p $runtimeRoot -m 'Frozen external division recovery policy v1' }
            $datasetOutput = $native.Output
            $datasetOperation = 'versioned_after_create_conflict'
        }
    }
    if ($native.ExitCode -ne 0) {
        throw "Learned-division runtime upload failed: $datasetOutput"
    }
    Write-ControllerLog "runtime_uploaded operation=$datasetOperation output=$datasetOutput"

    $datasetReady = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $native = Invoke-NativeOutput { & kaggle datasets status $runtimeRef --format json }
        $datasetStatus = $native.Output
        Write-ControllerLog "runtime_status output=$datasetStatus"
        if ($native.ExitCode -eq 0 -and $datasetStatus -match '(?i)ready|complete') {
            $datasetReady = $true
            break
        }
        if ($datasetStatus -match '(?i)error|failed') {
            throw "Learned-division runtime processing failed: $datasetStatus"
        }
        Start-Sleep -Seconds 60
    }
    if (-not $datasetReady) {
        throw 'Timed out waiting for learned-division runtime readiness'
    }

    if (
        (Test-Path -LiteralPath $candidateDir) -and
        @(Get-ChildItem -LiteralPath $candidateDir -Force).Count -gt 0
    ) {
        throw "Refusing to overwrite candidate kernel directory: $candidateDir"
    }
    $native = Invoke-NativeOutput { & $evaluationPython $candidateBuilder --runtime-root $runtimeRoot }
    $buildOutput = $native.Output
    if ($native.ExitCode -ne 0) {
        throw "Candidate kernel build failed: $buildOutput"
    }
    Write-ControllerLog "candidate_built output=$buildOutput"

    $native = Invoke-NativeOutput { & kaggle kernels push -p $candidateDir }
    $pushOutput = $native.Output
    if ($native.ExitCode -ne 0 -or $pushOutput -notmatch '(?i)successfully pushed') {
        throw "Candidate kernel push failed: $pushOutput"
    }
    Write-ControllerLog "candidate_pushed output=$pushOutput"
    Start-Sleep -Seconds 10
    $native = Invoke-NativeOutput { & $evaluationPython $kernelStateScript --kernel-slug $kernelRef }
    $stateText = $native.Output
    if ($native.ExitCode -ne 0) {
        throw "Candidate kernel state lookup failed: $stateText"
    }
    $state = $stateText | ConvertFrom-Json
    $kernelVersion = [int]$state.current_version_number
    if (
        $state.present -ne $true -or
        $kernelVersion -lt 1 -or
        $state.is_private -ne $true -or
        $state.enable_gpu -ne $true -or
        $state.enable_tpu -ne $false -or
        $state.enable_internet -ne $false -or
        $state.machine_shape -eq 'NvidiaTeslaTpuV3' -or
        $state.dataset_sources -notcontains $runtimeRef -or
        $state.kernel_sources -notcontains 'indarkarhana/biohub-zebrahub-multiscale-pretrain-v1' -or
        $state.competition_sources -notcontains $competition
    ) {
        throw "Candidate remote kernel state is invalid: $stateText"
    }

    $candidateComplete = $false
    $candidatePoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $candidatePoll += 1
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        $statusOutput = $native.Output
        Write-ControllerLog "candidate_status poll=$candidatePoll version=$kernelVersion output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            $candidateComplete = $true
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Candidate kernel did not complete: $statusOutput"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $candidateComplete) {
        throw 'Timed out waiting for candidate kernel completion'
    }

    $downloadRoot = Join-Path $projectRoot ".biohub/cache/kernel-outputs/ema-learned-division-candidate-v1-20260829-v$kernelVersion"
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Refusing to reuse candidate output directory: $downloadRoot"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $native = Invoke-NativeOutput { & kaggle kernels output $kernelRef -p $downloadRoot --force }
    $downloadOutput = $native.Output
    if ($native.ExitCode -ne 0) {
        throw "Candidate output download failed: $downloadOutput"
    }
    Write-ControllerLog "candidate_downloaded output=$downloadOutput"

    $native = Invoke-NativeOutput { & $evaluationPython $candidateVerifier --output-root $downloadRoot --baseline-validator $baselineValidator --report $promotionPath }
    $native.Output | Set-Content -LiteralPath (Join-Path $automationDir 'learned-division-candidate-promotion.stdout.log') -Encoding UTF8
    if ($native.ExitCode -ne 0) {
        $native.Output | Set-Content -LiteralPath $promotionError -Encoding UTF8
        Write-ControllerTerminal 'candidate_rejected' @{
            kernel_version = $kernelVersion
            download_root = $downloadRoot
            promotion_stderr = $promotionError
            submission_created = $true
            competition_submission_performed = $false
        }
        Write-ControllerLog "candidate_rejected promotion_stderr=$promotionError"
        exit 5
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne 'eligible_for_submission' -or $promotion.authorized_for_submission -ne $true) {
        throw 'Candidate promotion report is invalid'
    }

    $native = Invoke-NativeOutput { & $evaluationPython $candidateSubmitter --promotion $promotionPath --receipt $receiptPath --kernel-ref $kernelRef --kernel-version $kernelVersion --execute }
    $submitOutput = $native.Output
    if ($native.ExitCode -ne 0) {
        throw "Promoted candidate submission failed: $submitOutput"
    }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne 'submitted' -or $receipt.competition_submission_performed -ne $true) {
        throw 'Candidate submission receipt is invalid'
    }
    Write-ControllerTerminal 'submitted' @{
        kernel_version = $kernelVersion
        download_root = $downloadRoot
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        submission_sha256 = $promotion.submission_sha256
        proxy_gain = $promotion.proxy_gain
        learned_edges_added = $promotion.learned_edges_added
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        submission_created = $true
        competition_submission_performed = $true
    }
    Write-ControllerLog "submitted version=$kernelVersion sha256=$($promotion.submission_sha256)"
    exit 0
} catch {
    Write-ControllerTerminal 'failed' @{
        error = $_.Exception.Message
        submission_receipt_exists = (Test-Path -LiteralPath $receiptPath -PathType Leaf)
        competition_submission_performed = $false
    }
    Write-ControllerLog "failed error=$($_.Exception.Message)"
    exit 1
}
