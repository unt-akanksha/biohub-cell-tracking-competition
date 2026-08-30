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
    throw "Invalid credential wait bounds"
}

$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-seed-probe-deploy-v1"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
if (Test-Path -LiteralPath $terminalPath) {
    throw "Antelume seed-probe deployment already reached a terminal state"
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null

function Write-Terminal([hashtable]$Payload) {
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Invoke-External([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

Set-Location -LiteralPath $RepositoryRoot
$credentialReady = $false
for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
    # Windows PowerShell promotes native stderr to a terminating error when the
    # script-wide preference is Stop. An expired token is expected while this
    # controller waits, so sample the native exit code under Continue.
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
        run_id = "antelume-seed-probe-deploy-v1"
        credential_became_valid = $false
        remote_mutation_performed = $false
        competition_submission_performed = $false
    }
    exit 3
}

try {
    Invoke-External "aws" @(
        "ec2-instance-connect", "send-ssh-public-key",
        "--profile", $AwsProfile,
        "--region", $AwsRegion,
        "--instance-id", $InstanceId,
        "--availability-zone", $AvailabilityZone,
        "--instance-os-user", $RemoteUser,
        "--ssh-public-key", "file://$PublicKey"
    )

    $scpBase = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no")
    Invoke-External "scp" ($scpBase + @(
        "research/temporal_contrastive/score_real_division_seed_ensemble_probe.py",
        "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/research/temporal_contrastive/score_real_division_seed_ensemble_probe.py"
    ))
    Invoke-External "scp" ($scpBase + @(
        "research/temporal_contrastive/overnight_seed_policy.py",
        "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/research/temporal_contrastive/overnight_seed_policy.py"
    ))
    Invoke-External "scp" ($scpBase + @(
        "scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh",
        "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh"
    ))
    Invoke-External "scp" ($scpBase + @(
        "scripts/wait-package-antelume-seed-probe-harvest-v1.sh",
        "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/scripts/wait-package-antelume-seed-probe-harvest-v1.sh"
    ))

    $remoteCommand = @'
cd /home/ubuntu/biohub
chmod +x scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh
chmod +x scripts/wait-package-antelume-seed-probe-harvest-v1.sh
bash -n scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh
bash -n scripts/wait-package-antelume-seed-probe-harvest-v1.sh
/home/ubuntu/venv/bin/python -m py_compile research/temporal_contrastive/score_real_division_seed_ensemble_probe.py
/home/ubuntu/venv/bin/python -m py_compile research/temporal_contrastive/overnight_seed_policy.py
test ! -e /home/ubuntu/biohub-results/competition-real-division-seed-ensemble-probe-v1
nohup bash scripts/run-antelume-real-division-seed-probe-after-ensemble-v1.sh >/home/ubuntu/biohub-logs/real-division-seed-probe-controller-v1.log 2>&1 < /dev/null &
echo PROBE_CONTROLLER_PID=$!
'@
    Invoke-External "ssh" @(
        "-i", $SshKey,
        "-o", "StrictHostKeyChecking=no",
        "${RemoteUser}@${RemoteHost}",
        $remoteCommand
    )

    Write-Terminal @{
        schema_version = 1
        status = "deployed"
        run_id = "antelume-seed-probe-deploy-v1"
        credential_became_valid = $true
        remote_mutation_performed = $true
        remote_scope = "/home/ubuntu/biohub and /home/ubuntu/biohub-results only"
        competition_submission_performed = $false
    }
}
catch {
    Write-Terminal @{
        schema_version = 1
        status = "deployment_failed"
        run_id = "antelume-seed-probe-deploy-v1"
        credential_became_valid = $true
        remote_mutation_performed = $false
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    throw
}
