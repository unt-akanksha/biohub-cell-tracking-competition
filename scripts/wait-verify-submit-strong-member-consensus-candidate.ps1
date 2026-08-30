param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeManifest,
    [double]$MaximumWaitHours = 2.0,
    [int]$PollSeconds = 300,
    [int]$KernelVersion = 1,
    [string]$CandidateKernelRef = 'indarkarhana/biohub-ema-strong-member-candidate-v2',
    [string]$VerifierScript = 'verify-strong-member-consensus-submission-candidate.py',
    [string]$SubmitterScript = 'submit-strong-member-consensus-candidate.py',
    [string]$StatePrefix = 'strong-member-candidate',
    [string]$ControllerRunId = 'strong-member-candidate-controller-v2',
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$kernelRef = $CandidateKernelRef
$baselineValidator = Join-Path $projectRoot '.biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/validator_results.csv'
$verifier = if ([IO.Path]::IsPathRooted($VerifierScript)) { $VerifierScript } else { Join-Path $PSScriptRoot $VerifierScript }
$submitter = if ([IO.Path]::IsPathRooted($SubmitterScript)) { $SubmitterScript } else { Join-Path $PSScriptRoot $SubmitterScript }
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$downloadRoot = Join-Path $projectRoot ".biohub/cache/kernel-outputs/$StatePrefix-version$KernelVersion-20260830"
$promotionPath = Join-Path $automationDir "$StatePrefix-promotion-version$KernelVersion.json"
$receiptPath = Join-Path $automationDir "$StatePrefix-submission-receipt.json"
$terminalPath = Join-Path $automationDir "$StatePrefix-controller-version$KernelVersion.json"
$logPath = Join-Path $automationDir "$StatePrefix-controller-version$KernelVersion.log"
$requiredOutputPattern = '^(candidate_evidence\.json|run_stats\.csv|submission\.csv|validator_results\.csv|watchdog-terminal\.json)$'

function Write-ControllerLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ControllerTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = $ControllerRunId
        status = $Status
        kernel_ref = $kernelRef
        kernel_version = $KernelVersion
        target_public_score = 0.945
        runtime_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $RuntimeManifest).Hash.ToLowerInvariant()
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

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null
if ($PollSeconds -lt 60) {
    throw 'PollSeconds must be at least 60'
}
if ($KernelVersion -lt 1) {
    throw 'KernelVersion must be positive'
}
foreach ($required in @($RuntimeManifest, $baselineValidator, $verifier, $submitter, $evaluationPython)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required controller input is missing: $required"
    }
}
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw 'Kaggle CLI is unavailable'
}
if ($ValidateOnly) {
    @{
        status = 'valid'
        run_id = $ControllerRunId
        kernel_ref = $kernelRef
        kernel_version = $KernelVersion
        runtime_manifest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $RuntimeManifest).Hash.ToLowerInvariant()
        candidate_submission_requires_promotion_gate = $true
    } | ConvertTo-Json
    exit 0
}
foreach ($forbidden in @($terminalPath, $promotionPath, $receiptPath, $downloadRoot)) {
    if (Test-Path -LiteralPath $forbidden) {
        throw "Refusing to reuse controller state: $forbidden"
    }
}

try {
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    $complete = $false
    $poll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $poll += 1
        $native = Invoke-NativeOutput { & kaggle kernels status $kernelRef }
        Write-ControllerLog "candidate_status poll=$poll output=$($native.Output)"
        if ($native.ExitCode -eq 0 -and $native.Output -match '(?i)COMPLETE') {
            $complete = $true
            break
        }
        if ($native.Output -match '(?i)ERROR|CANCEL') {
            throw "Candidate kernel failed: $($native.Output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) {
        throw "Timed out waiting for candidate completion: $kernelRef"
    }

    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $versionedKernelRef = "$kernelRef/$KernelVersion"
    $native = Invoke-NativeOutput {
        & kaggle kernels output $versionedKernelRef `
            -p $downloadRoot `
            --force `
            --file-pattern $requiredOutputPattern `
            --page-size 200
    }
    if ($native.ExitCode -ne 0) {
        throw "Candidate output download failed: $($native.Output)"
    }
    Write-ControllerLog "candidate_downloaded output=$($native.Output)"

    $native = Invoke-NativeOutput {
        & $evaluationPython $verifier `
            --output-root $downloadRoot `
            --baseline-validator $baselineValidator `
            --runtime-manifest $RuntimeManifest `
            --report $promotionPath
    }
    if ($native.ExitCode -ne 0) {
        Write-ControllerTerminal 'candidate_rejected' @{
            download_root = $downloadRoot
            verification_error = $native.Output
            competition_submission_performed = $false
        }
        Write-ControllerLog "candidate_rejected error=$($native.Output)"
        exit 5
    }
    $promotion = Get-Content -Raw -LiteralPath $promotionPath | ConvertFrom-Json
    if ($promotion.status -ne 'eligible_for_submission' -or $promotion.authorized_for_submission -ne $true) {
        throw 'Candidate promotion report is invalid'
    }

    $native = Invoke-NativeOutput {
        & $evaluationPython $submitter `
            --promotion $promotionPath `
            --receipt $receiptPath `
            --kernel-ref $kernelRef `
            --kernel-version $KernelVersion `
            --execute
    }
    if ($native.ExitCode -ne 0) {
        throw "Promoted candidate submission failed: $($native.Output)"
    }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -ne 'submitted' -or $receipt.competition_submission_performed -ne $true) {
        throw 'Candidate submission receipt is invalid'
    }
    Write-ControllerTerminal 'submitted' @{
        download_root = $downloadRoot
        submission_sha256 = $promotion.submission_sha256
        proxy_gain = $promotion.proxy_gain
        ranked_edges_added = $promotion.ranked_edges_added
        deep_member_count = $promotion.deep_member_count
        promotion_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $promotionPath).Hash.ToLowerInvariant()
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        competition_submission_performed = $true
    }
    Write-ControllerLog "submitted sha256=$($promotion.submission_sha256)"
    exit 0
} catch {
    Write-ControllerTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-ControllerLog "failed error=$($_.Exception.Message)"
    exit 1
}
