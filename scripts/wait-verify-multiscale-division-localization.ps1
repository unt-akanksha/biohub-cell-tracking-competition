param(
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$launchTerminal = Join-Path $automationDir 'multiscale-division-localization-direct-launch.json'
$logPath = Join-Path $automationDir 'multiscale-division-localization-direct-verification.log'
$terminalPath = Join-Path $automationDir 'multiscale-division-localization-direct-verification.json'
$verificationOutput = Join-Path $automationDir 'multiscale-division-localization-direct-verification.stdout.json'
$verificationError = Join-Path $automationDir 'multiscale-division-localization-direct-verification.stderr.log'
$downloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/multiscale-division-localization-v1-direct'
$kernelRef = 'indarkarhana/biohub-multiscale-division-localization-v1'
$verifier = Join-Path $projectRoot 'research/temporal_contrastive/verify_division_localization_training_output.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$expectedVerifierSha256 = '6d3d527cffb6693db7ea5f9af429a1685d5d85e8db7c212a04e40c73af4a138c'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'multiscale-division-localization-v1'
        status = $Status
        kernel_ref = $kernelRef
        verifier_sha256 = $expectedVerifierSha256
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

try {
    $actualVerifier = (Get-FileHash -Algorithm SHA256 -LiteralPath $verifier).Hash.ToLowerInvariant()
    if ($actualVerifier -ne $expectedVerifierSha256) {
        throw "Localization verifier changed: $actualVerifier"
    }
    if (-not (Test-Path -LiteralPath $evaluationPython -PathType Leaf)) {
        throw "Evaluation Python is missing: $evaluationPython"
    }
    if ($ValidateOnly) {
        [pscustomobject]@{
            status = 'validated'
            stage = 'multiscale_division_localization_direct_verification'
        } | ConvertTo-Json
        exit 0
    }
    if (Test-Path -LiteralPath $terminalPath) {
        throw "Localization verification terminal already exists: $terminalPath"
    }
    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $launchTerminal -PathType Leaf)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for localization launch evidence'
        }
        Start-Sleep -Seconds 60
    }
    $launch = Get-Content -Raw -LiteralPath $launchTerminal | ConvertFrom-Json
    $version = [int]$launch.kernel_version
    if (
        $launch.status -ne 'launched' -or
        $launch.kernel_ref -ne $kernelRef -or
        $launch.expected_gpu_count -ne 2 -or
        $launch.quota_reserve_hours -ne 0.0 -or
        $version -lt 1 -or
        $launch.competition_data_attached -ne $false -or
        $launch.competition_submission_performed -ne $false
    ) {
        throw 'Localization launch evidence is invalid'
    }
    $poll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($poll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $poll += 1
        $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
        Write-ChainLog "kernel_status poll=$poll version=$version output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Localization kernel did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for localization completion'
    }
    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Localization download root already exists: $downloadRoot"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $downloadOutput = (& kaggle kernels output $kernelRef -p $downloadRoot --force 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Localization output download failed: $downloadOutput"
    }
    & $evaluationPython $verifier --root $downloadRoot 1> $verificationOutput 2> $verificationError
    if ($LASTEXITCODE -ne 0) {
        throw "Localization output verification failed; see $verificationError"
    }
    $verified = Get-Content -Raw -LiteralPath $verificationOutput | ConvertFrom-Json
    if (
        $verified.run_id -ne 'multiscale-division-localization-v1' -or
        $verified.family -ne 'multiscale_division_localization_v1' -or
        $verified.gpu_count -ne 2 -or
        $verified.folds.PSObject.Properties.Count -ne 2 -or
        $verified.authorized_for_graph_evaluation -ne $true -or
        $verified.authorized_for_submission -ne $false -or
        $verified.competition_submission_performed -ne $false
    ) {
        throw 'Localization verification evidence is invalid'
    }
    Write-ChainTerminal 'verified' @{
        kernel_version = $version
        localization_terminal_sha256 = $verified.terminal_sha256
        verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verificationOutput).Hash.ToLowerInvariant()
        authorized_for_graph_evaluation = $true
        authorized_for_submission = $false
    }
    Write-ChainLog "localization_verified version=$version"
    exit 0
} catch {
    if (-not $ValidateOnly) {
        Write-ChainTerminal 'failed' @{
            error = $_.Exception.Message
            authorized_for_graph_evaluation = $false
            authorized_for_submission = $false
        }
        Write-ChainLog "failed error=$($_.Exception.Message)"
    }
    throw
}
