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
    [int]$MaximumPolls = 720
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 15 -or $MaximumPolls -lt 1) {
    throw "Invalid harvest wait bounds"
}
$deployRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-seed-probe-deploy-v1"
$deployTerminal = Join-Path $deployRoot "deployment-terminal.json"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-seed-probe-harvest-v1"
$terminalPath = Join-Path $stateRoot "harvest-terminal.json"
$archivePath = Join-Path $stateRoot "seed-probe-harvest.tar.gz"
$reportPath = Join-Path $stateRoot "harvest-verification.json"
$sshErrorPath = Join-Path $stateRoot "ssh.stderr.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-antelume-seed-probe-harvest.py"
if (Test-Path -LiteralPath $terminalPath) {
    throw "Antelume seed-probe harvest already reached a terminal state"
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null

function Write-Terminal([hashtable]$Payload) {
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

$deployReady = $false
for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
    if (Test-Path -LiteralPath $deployTerminal -PathType Leaf) {
        $deploy = Get-Content -Raw -LiteralPath $deployTerminal | ConvertFrom-Json
        if ($deploy.status -eq "deployed") {
            $deployReady = $true
            break
        }
        Write-Terminal @{
            schema_version = 1
            status = "skipped_after_deployment_failure"
            run_id = "antelume-seed-probe-harvest-controller-v1"
            deployment_status = $deploy.status
            remote_harvest_started = $false
            competition_submission_performed = $false
        }
        exit 4
    }
    if ($poll -lt $MaximumPolls) {
        Start-Sleep -Seconds $PollSeconds
    }
}
if (-not $deployReady) {
    Write-Terminal @{
        schema_version = 1
        status = "deployment_wait_timed_out"
        run_id = "antelume-seed-probe-harvest-controller-v1"
        remote_harvest_started = $false
        competition_submission_performed = $false
    }
    exit 3
}

$credentialReady = $false
for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
    $savedErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & aws sts get-caller-identity --profile $AwsProfile --region $AwsRegion 2> $null | Out-Null
    $stsExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorPreference
    if ($stsExitCode -eq 0) {
        $credentialReady = $true
        break
    }
    if ($poll -lt $MaximumPolls) {
        Start-Sleep -Seconds $PollSeconds
    }
}
if (-not $credentialReady) {
    Write-Terminal @{
        schema_version = 1
        status = "credential_wait_timed_out"
        run_id = "antelume-seed-probe-harvest-controller-v1"
        remote_harvest_started = $false
        competition_submission_performed = $false
    }
    exit 3
}

$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& aws ec2-instance-connect send-ssh-public-key `
    --profile $AwsProfile `
    --region $AwsRegion `
    --instance-id $InstanceId `
    --availability-zone $AvailabilityZone `
    --instance-os-user $RemoteUser `
    --ssh-public-key "file://$PublicKey" 2> $null | Out-Null
$eicExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
if ($eicExitCode -ne 0) {
    throw "Failed to publish the harvest SSH key"
}

$remoteCommand = "bash /home/ubuntu/biohub/scripts/wait-package-antelume-seed-probe-harvest-v1.sh"
$process = Start-Process -FilePath ssh.exe -ArgumentList @(
    "-i", $SshKey,
    "-o", "StrictHostKeyChecking=no",
    "${RemoteUser}@${RemoteHost}",
    $remoteCommand
) -WindowStyle Hidden -RedirectStandardOutput $archivePath -RedirectStandardError $sshErrorPath -Wait -PassThru
if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Write-Terminal @{
        schema_version = 1
        status = "remote_harvest_failed"
        run_id = "antelume-seed-probe-harvest-controller-v1"
        ssh_exit_code = $process.ExitCode
        remote_harvest_started = $true
        competition_submission_performed = $false
    }
    exit 5
}

& python $verifier --archive $archivePath --report $reportPath *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Downloaded Antelume harvest failed verification"
}
$verification = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
Write-Terminal @{
    schema_version = 1
    status = "harvest_verified"
    run_id = "antelume-seed-probe-harvest-controller-v1"
    archive_path = $archivePath
    archive_sha256 = $verification.archive_sha256
    probe_controller_status = $verification.probe_controller_status
    member_count = $verification.member_count
    remote_harvest_started = $true
    competition_submission_performed = $false
}
