param(
    [double]$MaximumWaitHours = 13.0,
    [int]$PollSeconds = 600,
    [int]$KernelVersion = 1,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$kernelRef = 'indarkarhana/biohub-948tta2-lsm-consensus-v1'
$verifier = Join-Path $PSScriptRoot 'verify-948tta2-lsm-consensus-candidate.py'
$submitter = Join-Path $PSScriptRoot 'submit-948tta2-lsm-consensus-candidate.py'
$downloadRoot = Join-Path $projectRoot ".biohub/cache/kernel-outputs/biohub-948tta2-lsm-consensus-v1-version$KernelVersion-20260908"
$promotionPath = Join-Path $automationDir "948tta2-lsm-consensus-promotion-version$KernelVersion.json"
$receiptPath = Join-Path $automationDir "948tta2-lsm-consensus-submission-version$KernelVersion.json"
$terminalPath = Join-Path $automationDir "948tta2-lsm-consensus-controller-version$KernelVersion.json"
$logPath = Join-Path $automationDir "948tta2-lsm-consensus-controller-version$KernelVersion.log"
$requiredOutputPattern = '^(candidate_evidence\.json|launcher_terminal\.json|run_stats\.csv|submission\.csv|validator_results\.csv)$'

function Write-ControllerLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ControllerTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = '948tta2-lsm-consensus-controller-v1'
        status = $Status
        kernel_ref = $kernelRef
        kernel_version = $KernelVersion
        target_public_score = 0.945
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
    [pscustomobject]@{ExitCode = $exitCode; Output = ($lines -join "`n")}
}

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null
if ($PollSeconds -lt 60) { throw 'PollSeconds must be at least 60' }
if ($KernelVersion -lt 1) { throw 'KernelVersion must be positive' }
foreach ($required in @($verifier, $submitter)) {
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
        run_id = '948tta2-lsm-consensus-controller-v1'
        kernel_ref = $kernelRef
        kernel_version = $KernelVersion
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
    if (-not $complete) { throw 'Timed out waiting for LSM-consensus candidate completion' }

    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $versionedKernelRef = "$kernelRef/$KernelVersion"
    $native = Invoke-NativeOutput {
        & kaggle kernels output $versionedKernelRef -p $downloadRoot --force `
            --file-pattern $requiredOutputPattern --page-size 200
    }
    if ($native.ExitCode -ne 0) {
        throw "Candidate output download failed: $($native.Output)"
    }
    Write-ControllerLog "candidate_downloaded output=$($native.Output)"

    $native = Invoke-NativeOutput {
        & python $verifier --output-root $downloadRoot --report $promotionPath
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
        & python $submitter --promotion $promotionPath --receipt $receiptPath `
            --kernel-ref $kernelRef --kernel-version $KernelVersion --execute
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
        production_coordinate_changes = $promotion.production_coordinate_changes
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
