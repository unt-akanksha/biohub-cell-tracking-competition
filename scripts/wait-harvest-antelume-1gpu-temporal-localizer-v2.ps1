param(
    [string]$RemoteHost = "3.233.240.87",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 1200,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 30 -or $MaximumPolls -lt 1) {
    throw "Invalid Antelume temporal-localizer harvest bounds"
}
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/aws-4gpu-temporal-localizer-v1"
$terminalPath = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz"
$sshErrorPath = Join-Path $stateRoot "antelume-v2-harvest.stderr.log"
$remoteArchive = "/home/ubuntu/biohub-temporal-localizer-v2/biohub-synthetic256-real-replay-temporal-node-localizer-v2-results.tar.gz"
$remoteSums = "$remoteArchive.sha256"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "aws-4gpu-temporal-localizer-harvest-v1"
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if ($ValidateOnly) {
    @{
        status = "validated"
        stage = "antelume_1gpu_temporal_localizer_harvest_v2"
        autonomous_candidate_handoff = $true
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Temporal-localizer harvest already reached a terminal state"
}

try {
    $remoteCommand = "set -euo pipefail; for poll in `$(seq 1 $MaximumPolls); do if test -f '$remoteArchive' && test -f '$remoteSums'; then break; fi; sleep $PollSeconds; done; test -f '$remoteArchive'; test -f '$remoteSums'; sha256sum -c '$remoteSums' >/dev/null; cat '$remoteArchive'"
    $process = Start-Process -FilePath ssh.exe -ArgumentList @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30",
        "${RemoteUser}@${RemoteHost}",
        $remoteCommand
    ) -WindowStyle Hidden -RedirectStandardOutput $archivePath -RedirectStandardError $sshErrorPath -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Antelume temporal-localizer remote harvest failed with SSH exit code $($process.ExitCode)"
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    $bytes = (Get-Item -LiteralPath $archivePath).Length
    if ($bytes -lt 1) { throw "Antelume temporal-localizer result archive is empty" }
    Write-Terminal @{
        status = "harvested"
        result_sha256 = $hash
        result_bytes = [int64]$bytes
        source_instance_id = "i-0d12195df0d3558f3"
        source_instance_type = "g5.xlarge"
        training_gpu_count = 1
        execution_policy = "four independent members trained sequentially on one Antelume A10G"
        recoverable_instance_preserved = $true
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        recoverable_instance_preserved = $true
    }
    throw
}
