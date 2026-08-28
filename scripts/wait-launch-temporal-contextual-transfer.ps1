param(
    [double]$MinimumQuotaHours = 11.0,
    [double]$MaximumWaitHours = 12.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$kernelDir = Join-Path $projectRoot 'kaggle/biohub-temporal-contextual-transfer-v3'
$metadataPath = Join-Path $kernelDir 'kernel-metadata.json'
$notebookPath = Join-Path $kernelDir 'biohub-temporal-contextual-transfer-v3.ipynb'
$automationDir = Join-Path $projectRoot '.biohub/automation'
$logPath = Join-Path $automationDir 'temporal-contextual-transfer-launch.log'
$terminalPath = Join-Path $automationDir 'temporal-contextual-transfer-launch.json'
$kernelRef = 'indarkarhana/biohub-temporal-contextual-transfer-v3'
$acceptanceRef = 'indarkarhana/biohub-zsns001-contextual-gate-v1'
$expectedMetadataSha256 = 'ad275e1f816bd629b6b6bd677d1d9cb86029605abd1f04e44d5874ffc037461e'
$expectedNotebookSha256 = '17f0f25562c4a0878d9070ea0b238867bf190b1b9b754899bf0d395dee86f353'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-LaunchLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-LaunchTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'temporal-contextual-pair-fusion-v3'
        status = $Status
        kernel_ref = $kernelRef
        minimum_quota_hours = $MinimumQuotaHours
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        metadata_sha256 = $expectedMetadataSha256
        notebook_sha256 = $expectedNotebookSha256
        public_leaderboard_used_for_selection = $false
        submission_created = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

try {
    $metadataSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadataPath).Hash.ToLowerInvariant()
    $notebookSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $notebookPath).Hash.ToLowerInvariant()
    if ($metadataSha256 -ne $expectedMetadataSha256) {
        throw "Transfer metadata changed: $metadataSha256"
    }
    if ($notebookSha256 -ne $expectedNotebookSha256) {
        throw "Transfer notebook changed: $notebookSha256"
    }
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.kernel_sources -notcontains $acceptanceRef
    ) {
        throw 'Transfer metadata contract is invalid'
    }
    $acceptanceStatus = (& kaggle kernels status $acceptanceRef 2>&1) -join "`n"
    if ($acceptanceStatus -notmatch 'COMPLETE') {
        throw "Accepted ZSNS001 source is not complete: $acceptanceStatus"
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    $quota = (& kaggle quota --format json 2>&1) -join "`n"
    $resources = $quota | ConvertFrom-Json
    $gpu = $resources | Where-Object resource -eq 'GPU'
    $remaining = [double](($gpu.remaining -replace 'h', ''))
    if ($remaining -lt $MinimumQuotaHours) {
        $refreshText = [string]$gpu.refreshAt
        if ($refreshText -notmatch 'Z|[+-]\d\d:\d\d$') {
            $refreshText += 'Z'
        }
        $wakeAt = [DateTimeOffset]::Parse($refreshText).AddMinutes(2)
        Write-LaunchLog "waiting_for_refresh remaining=$remaining wake_at=$($wakeAt.ToString('o'))"
        while ([DateTimeOffset]::UtcNow -lt $wakeAt) {
            if ([DateTimeOffset]::UtcNow -ge $deadline) {
                throw 'Timed out before the Kaggle quota refresh'
            }
            Start-Sleep -Seconds 60
        }
    }

    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quota = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quota | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-LaunchLog "quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $kernelDir 2>&1) -join "`n"
        Write-LaunchLog "push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
            if ($statusOutput -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Transfer push returned success but status is invalid: $statusOutput"
            }
            Write-LaunchTerminal 'launched' @{
                quota_before_hours = $remaining
                kaggle_status = $statusOutput
                push_output = $pushOutput
            }
            Write-LaunchLog "launched status=$statusOutput"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Transfer push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch contextual transfer'
} catch {
    Write-LaunchTerminal 'failed' @{ error = $_.Exception.Message }
    Write-LaunchLog "failed error=$($_.Exception.Message)"
    exit 1
}
