param(
    [double]$MaximumWaitHours = 120.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$finalLaunchTerminal = Join-Path $automationDir 'temporal-contextual-final-launch.json'
$logPath = Join-Path $automationDir 'temporal-contextual-submission.log'
$terminalPath = Join-Path $automationDir 'temporal-contextual-submission.json'
$receiptPath = Join-Path $automationDir 'temporal-contextual-submission-receipt.json'
$finalRef = 'indarkarhana/biohub-temporal-contextual-submission-candidate-v3'
$finalDir = Join-Path $projectRoot 'kaggle/biohub-temporal-contextual-submission-candidate-v3'
$finalMetadata = Join-Path $finalDir 'kernel-metadata.json'
$finalNotebook = Join-Path $finalDir 'biohub-temporal-contextual-submission-candidate-v3.ipynb'
$submitScript = Join-Path $projectRoot 'scripts/submit-temporal-contextual-kernel.py'
$eventsPath = Join-Path $projectRoot 'experiments/events.jsonl'
$downloadDir = Join-Path $projectRoot '.biohub/cache/kernel-outputs/temporal-contextual-submission-v3-autochain'
$submitOutput = Join-Path $automationDir 'temporal-contextual-submission.stdout.log'
$submitError = Join-Path $automationDir 'temporal-contextual-submission.stderr.log'
$expectedSubmitScriptSha256 = 'f6122f465c9c39abd6f5e4dbbc476b00fa3c7dd2fe1d256826d22e41ea1be043'
$expectedFinalMetadataSha256 = '429ddae35e0633069a5ac44151cd6a1e4782ccd130b1f315d448eb43be8276fe'
$expectedFinalNotebookSha256 = '5dad76f56003be6f84e381dbf1a735cb2e091dcedf8f238edb184fd0091c03af'

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
        competition = 'biohub-cell-tracking-during-development'
        kernel_ref = $finalRef
        submit_script_sha256 = $expectedSubmitScriptSha256
        final_metadata_sha256 = $expectedFinalMetadataSha256
        final_notebook_sha256 = $expectedFinalNotebookSha256
        public_leaderboard_used_for_selection = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label changed: $actual"
    }
}

try {
    Require-Hash $submitScript $expectedSubmitScriptSha256 'Submission verifier'
    Require-Hash $finalMetadata $expectedFinalMetadataSha256 'Final metadata'
    Require-Hash $finalNotebook $expectedFinalNotebookSha256 'Final notebook'
    if (-not (Test-Path -LiteralPath $eventsPath)) {
        throw "Submission authorization log is missing: $eventsPath"
    }
    if (Test-Path -LiteralPath $receiptPath) {
        throw "Submission receipt already exists: $receiptPath"
    }
    if (Test-Path -LiteralPath $downloadDir) {
        throw "Submission download directory already exists: $downloadDir"
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $finalLaunchTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for the final candidate launch terminal'
        }
        Start-Sleep -Seconds 60
    }
    $finalLaunch = Get-Content -Raw -LiteralPath $finalLaunchTerminal | ConvertFrom-Json
    $version = [int]$finalLaunch.kernel_version
    if (
        $finalLaunch.status -ne 'launched' -or
        $finalLaunch.final_ref -ne $finalRef -or
        $finalLaunch.expected_gpu_count -ne 2 -or
        $version -lt 1 -or
        $finalLaunch.public_leaderboard_used_for_selection -ne $false -or
        $finalLaunch.competition_submission_performed -ne $false
    ) {
        throw 'Final autolaunch did not produce valid launch evidence'
    }

    $externalPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($externalPoll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $externalPoll += 1
        $statusOutput = (& kaggle kernels status $finalRef 2>&1) -join "`n"
        Write-ChainLog "final_status poll=$externalPoll version=$version output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Final contextual candidate did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for final contextual candidate completion'
    }

    & python $submitScript --kernel-dir $finalDir --kernel-version $version --download-dir $downloadDir --events $eventsPath --receipt $receiptPath --message 'Contextual v3: clean reciprocal exact-gated non-replica candidate' --execute 1> $submitOutput 2> $submitError
    if ($LASTEXITCODE -ne 0) {
        throw "Verified candidate submission failed; see $submitError"
    }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if (
        $receipt.status -ne 'submitted' -or
        $receipt.run_id -ne 'temporal-contextual-pair-fusion-candidate-v3' -or
        $receipt.kernel_ref -ne $finalRef -or
        [int]$receipt.kernel_version -ne $version -or
        $receipt.public_leaderboard_used_for_selection -ne $false -or
        $receipt.competition_submission_performed -ne $true
    ) {
        throw 'Submission receipt is invalid'
    }
    Write-ChainTerminal 'submitted' @{
        kernel_version = $version
        candidate_sha256 = $receipt.candidate_sha256
        candidate_report_sha256 = $receipt.candidate_report_sha256
        launcher_terminal_sha256 = $receipt.launcher_terminal_sha256
        receipt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $receiptPath).Hash.ToLowerInvariant()
        competition_submission_performed = $true
        submitted_at = $receipt.submitted_at
        message = $receipt.message
    }
    Write-ChainLog "submitted version=$version candidate_sha256=$($receipt.candidate_sha256)"
    exit 0
} catch {
    Write-ChainTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-ChainLog "failed error=$($_.Exception.Message)"
    exit 1
}
