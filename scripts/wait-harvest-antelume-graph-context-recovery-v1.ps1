param(
    [string]$RemoteHost = "3.233.240.87",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 1800,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 30 -or $MaximumPolls -lt 1) {
    throw "Invalid graph-context recovery harvest bounds"
}
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-recovery-v1"
$terminalPath = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "graph-context-recovery-results.tar.gz"
$reportPath = Join-Path $stateRoot "harvest-verification.json"
$sshErrorPath = Join-Path $stateRoot "harvest.stderr.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-graph-context-division-harvest.py"
$remoteArchive = "/home/ubuntu/biohub-graph-context-recovery-v1-results.tar.gz"
$remoteSums = "$remoteArchive.sha256"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-graph-context-recovery-harvest-v1"
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if ($ValidateOnly) {
    & python -m py_compile $verifier
    if ($LASTEXITCODE -ne 0) { throw "Graph-context recovery verifier validation failed" }
    @{
        status = "validated"
        stage = "antelume_graph_context_recovery_harvest"
        recovered_completed_members = 5
        autonomous_development_handoff = $true
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Graph-context recovery harvest already reached a terminal state"
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
        throw "Graph-context recovery remote harvest failed with SSH exit code $($process.ExitCode)"
    }
    & python $verifier --archive $archivePath --report $reportPath *> $null
    if ($LASTEXITCODE -ne 0) { throw "Graph-context recovery harvest verification failed" }
    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    if (
        [int]$report.completed_model_count -ne 8 -or
        [int]$report.resumed_completed_member_count -ne 5 -or
        $report.recovery_manifest_present -ne $true
    ) {
        throw "Graph-context recovery completion evidence is invalid"
    }
    Write-Terminal @{
        status = "harvest_verified"
        archive_sha256 = $report.archive_sha256
        archive_bytes = [int64]$report.archive_bytes
        training_exit_code = [int]$report.training_exit_code
        development_probe_exit_code = [int]$report.development_probe_exit_code
        completed_model_count = [int]$report.completed_model_count
        resumed_completed_member_count = [int]$report.resumed_completed_member_count
        selection_accepted_members = $report.selection_accepted_members
        independently_strong_members = $report.independently_strong_members
        deployment_members = $report.deployment_members
        ensemble_eligible = $report.ensemble_eligible
        development_probe_present = $report.development_probe_present
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
    }
    throw
}
