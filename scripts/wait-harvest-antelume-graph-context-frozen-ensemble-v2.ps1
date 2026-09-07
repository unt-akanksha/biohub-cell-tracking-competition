param(
    [string]$AwsProfile = "148971207977_InventoryOptimization-EC2-Access",
    [string]$AwsRegion = "us-east-1",
    [string]$InstanceId = "i-0d12195df0d3558f3",
    [string]$AvailabilityZone = "us-east-1b",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$PublicKey = "C:/Users/IndarKumar/.ssh/rsna_ec2.pub",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 300,
    [int]$MaximumPolls = 2880,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 30 -or $MaximumPolls -lt 1) {
    throw "Invalid graph-context v2 harvest wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$deployTerminal = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2-deploy/deployment-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2"
$archivePath = Join-Path $stateRoot "graph-context-frozen-ensemble-v2-results.tar.gz"
$shaPath = "$archivePath.sha256"
$verificationTerminal = Join-Path $stateRoot "verification-terminal.json"
$harvestTerminal = Join-Path $stateRoot "harvest-terminal.json"
$logPath = Join-Path $stateRoot "harvest.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-graph-context-frozen-ensemble-v2-harvest.py"
$remoteArchive = "/home/ubuntu/biohub-graph-context-frozen-ensemble-v2-results.tar.gz"
$remoteSha = "$remoteArchive.sha256"
$remoteAcknowledgement = "/home/ubuntu/biohub-graph-context-frozen-ensemble-v2/harvest.verified"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-graph-context-frozen-ensemble-v2-harvest"
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $partial = "$harvestTerminal.partial"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $partial -Encoding utf8
    Move-Item -LiteralPath $partial -Destination $harvestTerminal -Force
}

function Publish-Key {
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) {
    throw "Graph-context v2 harvest verifier is missing"
}
& python -m py_compile $verifier
if ($LASTEXITCODE -ne 0) { throw "Graph-context v2 harvest verifier is invalid" }
if ($ValidateOnly) {
    @{ status = "validated"; stage = "graph_context_frozen_ensemble_v2_harvest" } | ConvertTo-Json
    exit 0
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if (Test-Path -LiteralPath $harvestTerminal) {
    throw "Graph-context v2 harvest already reached a terminal state"
}

try {
    $remote = $null
    $ready = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $deployed = $false
        if (Test-Path -LiteralPath $deployTerminal -PathType Leaf) {
            $deploy = Get-Content -Raw -LiteralPath $deployTerminal | ConvertFrom-Json
            $deployed = $deploy.status -eq "deployed_and_queued"
        }
        & aws sts get-caller-identity --profile $AwsProfile --region $AwsRegion 2> $null | Out-Null
        $credentialsReady = $LASTEXITCODE -eq 0
        if ($deployed -and $credentialsReady) {
            $remoteHost = (& aws ec2 describe-instances `
                --profile $AwsProfile `
                --region $AwsRegion `
                --instance-ids $InstanceId `
                --query "Reservations[0].Instances[0].PublicIpAddress" `
                --output text 2> $null).Trim()
            if ($LASTEXITCODE -eq 0 -and $remoteHost -and $remoteHost -ne "None") {
                $remote = "${RemoteUser}@${remoteHost}"
                if (Publish-Key) {
                    & ssh.exe -i $SshKey -o StrictHostKeyChecking=no $remote "test -s '$remoteArchive' -a -s '$remoteSha'" 2> $null
                    $ready = $LASTEXITCODE -eq 0
                }
            }
        }
        $ErrorActionPreference = $savedPreference
        if ($ready) { break }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $ready -or -not $remote) {
        throw "Timed out waiting for the graph-context v2 result archive"
    }

    if (-not (Publish-Key)) { throw "Failed to publish harvest SSH key" }
    & scp.exe -i $SshKey -o StrictHostKeyChecking=no "${remote}:$remoteSha" $shaPath 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Failed to copy graph-context v2 archive hash" }
    $expectedSha = ((Get-Content -Raw -LiteralPath $shaPath).Trim() -split '\s+')[0].ToLowerInvariant()
    if ($expectedSha -notmatch '^[0-9a-f]{64}$') { throw "Invalid graph-context v2 archive hash" }
    if (-not (Publish-Key)) { throw "Failed to publish archive-copy SSH key" }
    & scp.exe -i $SshKey -o StrictHostKeyChecking=no "${remote}:$remoteArchive" $archivePath 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Failed to copy graph-context v2 result archive" }
    $actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($actualSha -ne $expectedSha) { throw "Graph-context v2 result archive changed in transit" }

    $extractRoot = Join-Path $stateRoot ("extracted-" + $expectedSha.Substring(0, 12))
    & python $verifier `
        --archive $archivePath `
        --archive-sha256 $expectedSha `
        --output-root $extractRoot `
        --terminal $verificationTerminal 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Graph-context v2 result verification failed" }
    $verification = Get-Content -Raw -LiteralPath $verificationTerminal | ConvertFrom-Json

    if (-not (Publish-Key)) { throw "Failed to publish acknowledgement SSH key" }
    & ssh.exe -i $SshKey -o StrictHostKeyChecking=no $remote "printf '%s\n' '$expectedSha' > '$remoteAcknowledgement'" 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Failed to acknowledge graph-context v2 harvest" }
    Write-Terminal @{
        status = $verification.status
        archive_sha256 = $expectedSha
        archive_bytes = (Get-Item -LiteralPath $archivePath).Length
        extracted_to = $extractRoot
        verification_terminal = $verificationTerminal
        deployment_member_count = $verification.deployment_member_count
        development_probe_metrics = $verification.development_probe_metrics
        remote_harvest_acknowledged = $true
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        remote_harvest_acknowledged = $false
    }
    throw
}
