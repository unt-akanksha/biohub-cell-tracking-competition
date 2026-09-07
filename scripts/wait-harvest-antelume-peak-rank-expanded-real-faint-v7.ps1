param(
    [string]$RemoteHost = "100.54.137.149",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 7200,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 30 -or $MaximumPolls -lt 1) {
    throw "Invalid expanded-real faint detector harvest bounds"
}
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-peak-rank-expanded-real-faint-v7"
$terminalPath = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "peak-rank-expanded-real-faint-v7-results.tar.gz"
$reportPath = Join-Path $stateRoot "harvest-verification.json"
$sshErrorPath = Join-Path $stateRoot "harvest.stderr.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-peak-rank-expanded-real-faint-v7-harvest.py"
$remoteArchive = "/home/ubuntu/biohub-peak-rank-expanded-real-faint-v7-results.tar.gz"
$remoteSums = "$remoteArchive.sha256"
$remoteAcknowledgement = "/home/ubuntu/biohub-peak-rank-detector-v1/expanded-real-v7/harvest.verified"

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-peak-rank-expanded-real-faint-v7-harvest"
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
    if ($LASTEXITCODE -ne 0) { throw "Expanded-real faint archive verifier validation failed" }
    @{ status = "validated"; stage = "antelume_peak_rank_expanded_real_faint_v7_harvest" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Expanded-real faint harvest already reached a terminal state"
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
        throw "Expanded-real faint remote harvest failed with SSH exit code $($process.ExitCode)"
    }
    & python $verifier --archive $archivePath --report $reportPath *> $null
    if ($LASTEXITCODE -ne 0) { throw "Expanded-real faint harvest verification failed" }
    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    $ackCommand = "set -euo pipefail; test -f '$remoteArchive'; test -f '$remoteSums'; sha256sum -c '$remoteSums' >/dev/null; printf '%s %s\n' '$($report.archive_sha256)' '$($report.archive_bytes)' > '$remoteAcknowledgement.partial'; mv '$remoteAcknowledgement.partial' '$remoteAcknowledgement'"
    $ackProcess = Start-Process -FilePath ssh.exe -ArgumentList @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "${RemoteUser}@${RemoteHost}",
        $ackCommand
    ) -WindowStyle Hidden -Wait -PassThru
    if ($ackProcess.ExitCode -ne 0) { throw "Remote expanded-real faint harvest acknowledgement failed" }
    Write-Terminal @{
        status = "harvest_verified"
        archive_sha256 = $report.archive_sha256
        archive_bytes = [int64]$report.archive_bytes
        training_exit_code = [int]$report.training_exit_code
        model_status = $report.model_status
        selection_passed = $report.selection_passed
        audit_opened = $report.audit_opened
        audit_passed = $report.audit_passed
        accepted_for_kaggle_validation = $report.accepted_for_kaggle_validation
        best_step = [int]$report.best_step
        parameter_count = [int64]$report.parameter_count
        checkpoint_sha256 = $report.checkpoint_sha256
        variant = $report.variant
    }
}
catch {
    Write-Terminal @{ status = "failed"; error = $_.Exception.Message }
    throw
}
