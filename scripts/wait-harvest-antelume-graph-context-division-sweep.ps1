param(
    [string]$AwsProfile = "148971207977_InventoryOptimization-EC2-Access",
    [string]$AwsRegion = "us-east-1",
    [string]$InstanceId = "i-0d12195df0d3558f3",
    [string]$AvailabilityZone = "us-east-1b",
    [string]$RemoteHost = "100.52.213.73",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$PublicKey = "C:/Users/IndarKumar/.ssh/rsna_ec2.pub",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 1440,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 15 -or $MaximumPolls -lt 1) {
    throw "Invalid graph-context harvest wait bounds"
}
$deployTerminal = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-division-sweep-v1/deployment-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-division-harvest-v1"
$terminalPath = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "graph-context-division-results.tar.gz"
$reportPath = Join-Path $stateRoot "harvest-verification.json"
$sshErrorPath = Join-Path $stateRoot "ssh.stderr.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-graph-context-division-harvest.py"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-graph-context-division-harvest-v1"
    $Payload["competition_submission_performed"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if ($ValidateOnly) {
    & python -m py_compile $verifier
    if ($LASTEXITCODE -ne 0) { throw "Graph-context harvest verifier validation failed" }
    @{ status = "validated"; stage = "antelume_graph_context_division_harvest" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Graph-context harvest already reached a terminal state"
}

try {
    $deploymentReady = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        if (Test-Path -LiteralPath $deployTerminal -PathType Leaf) {
            $deployment = Get-Content -Raw -LiteralPath $deployTerminal | ConvertFrom-Json
            if ($deployment.status -eq "deployed_and_queued") {
                $deploymentReady = $true
                break
            }
            throw "Graph-context deployment did not queue: $($deployment.status)"
        }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $deploymentReady) { throw "Timed out waiting for graph-context deployment" }

    $credentialReady = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & aws sts get-caller-identity --profile $AwsProfile --region $AwsRegion 2> $null | Out-Null
        $status = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($status -eq 0) { $credentialReady = $true; break }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $credentialReady) {
        throw "Timed out waiting for refreshed AWS credentials for graph-context harvest"
    }
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    if ($LASTEXITCODE -ne 0) { throw "Failed to publish graph-context harvest SSH key" }

    $remoteCommand = @'
set -euo pipefail
archive=/home/ubuntu/biohub-graph-context-division-sweep-v1-results.tar.gz
sums=/home/ubuntu/biohub-graph-context-division-sweep-v1-results.tar.gz.sha256
for poll in $(seq 1 1440); do
  if test -f "$archive" && test -f "$sums"; then
    break
  fi
  sleep 60
done
test -f "$archive"
test -f "$sums"
sha256sum -c "$sums" >/dev/null
cat "$archive"
'@
    $process = Start-Process -FilePath ssh.exe -ArgumentList @(
        "-i", $SshKey,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30",
        "${RemoteUser}@${RemoteHost}",
        $remoteCommand
    ) -WindowStyle Hidden -RedirectStandardOutput $archivePath -RedirectStandardError $sshErrorPath -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Graph-context remote harvest failed with SSH exit code $($process.ExitCode)"
    }
    & python $verifier --archive $archivePath --report $reportPath *> $null
    if ($LASTEXITCODE -ne 0) { throw "Graph-context harvest verification failed" }
    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    Write-Terminal @{
        status = "harvest_verified"
        archive_sha256 = $report.archive_sha256
        archive_bytes = [int64]$report.archive_bytes
        training_exit_code = [int]$report.training_exit_code
        development_probe_exit_code = [int]$report.development_probe_exit_code
        completed_model_count = $report.completed_model_count
        selection_accepted_members = $report.selection_accepted_members
        independently_strong_members = $report.independently_strong_members
        deployment_members = $report.deployment_members
        ensemble_eligible = $report.ensemble_eligible
        final_probe_opened = $report.development_probe_present
        authorized_for_submission = $false
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        authorized_for_submission = $false
    }
    throw
}
