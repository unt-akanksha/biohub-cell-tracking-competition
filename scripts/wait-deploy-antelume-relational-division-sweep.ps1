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
    throw "Invalid relational deployment wait bounds"
}
$kernelRef = "indarkarhana/biohub-relational-division-patches-v3"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-relational-division-sweep-v1"
$downloadRoot = Join-Path $stateRoot "kaggle-output"
$archivePath = Join-Path $downloadRoot "biohub_relational_division_patches_v3.tar.gz"
$verificationPath = Join-Path $stateRoot "archive-verification.json"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-relational-division-patch-archive.py"
$trainer = Join-Path $RepositoryRoot "research/temporal_contrastive/train_relational_division_sweep.py"
$model = Join-Path $RepositoryRoot "research/temporal_contrastive/relational_division_model.py"
$inference = Join-Path $RepositoryRoot "research/temporal_contrastive/relational_division_inference.py"
$probeScorer = Join-Path $RepositoryRoot "research/temporal_contrastive/score_relational_division_development_probe.py"
$developmentInventory = Join-Path $RepositoryRoot ".biohub/results/competition-relational-division-development-inventory-v1.json"
$runnerTemplate = Join-Path $RepositoryRoot "scripts/run-antelume-relational-division-sweep-v1.sh"
$renderedRunner = Join-Path $stateRoot "run-antelume-relational-division-sweep-v1.sh"

function Write-Terminal([hashtable]$Payload) {
    $Payload.schema_version = 1
    $Payload.run_id = "antelume-relational-division-sweep-deploy-v1"
    $Payload.competition_submission_performed = $false
    $Payload.recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Append-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (([DateTimeOffset]::Now.ToString("o")) + " " + $Message)
}

function Publish-Key {
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to publish the relational deployment SSH key"
    }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
if ($ValidateOnly) {
    foreach ($path in @($verifier, $trainer, $model, $inference, $probeScorer, $developmentInventory, $runnerTemplate)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Relational deployment input is missing: $path"
        }
    }
    & python -m py_compile $verifier $trainer $model $inference $probeScorer
    if ($LASTEXITCODE -ne 0) { throw "Relational Python validation failed" }
    Push-Location -LiteralPath $RepositoryRoot
    try {
        & bash -n "scripts/run-antelume-relational-division-sweep-v1.sh"
        if ($LASTEXITCODE -ne 0) { throw "Relational runner validation failed" }
    }
    finally {
        Pop-Location
    }
    @{ status = "validated"; stage = "antelume_relational_division_sweep_deploy" } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Relational deployment already reached a terminal state"
}

try {
    $kernelComplete = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
        Append-Log "kaggle_status poll=$poll output=$statusOutput"
        if ($statusOutput -match "COMPLETE") { $kernelComplete = $true; break }
        if ($statusOutput -match "ERROR|CANCEL") { throw "Relational extractor failed: $statusOutput" }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $kernelComplete) { throw "Timed out waiting for relational extraction" }
    New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
    $downloaded = Test-Path -LiteralPath $archivePath -PathType Leaf
    if ($downloaded) { Append-Log "reusing_existing_relational_archive path=$archivePath" }
    for ($retry = 1; -not $downloaded -and $retry -le 12; $retry++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $downloadOutput = (& kaggle kernels output $kernelRef `
            -p $downloadRoot `
            --file-pattern "biohub_relational_division_patches_v3.tar.gz" `
            --force 2>&1) -join "`n"
        $downloadExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        Append-Log "kaggle_output retry=$retry exit=$downloadExitCode output=$downloadOutput"
        if ($downloadExitCode -eq 0) {
            $downloaded = $true
            break
        }
        if ($downloadOutput -notmatch "429|Too Many Requests") {
            throw "Relational output download failed: $downloadOutput"
        }
        if ($retry -lt 12) { Start-Sleep -Seconds ([Math]::Min(300, 30 * $retry)) }
    }
    if (-not $downloaded) { throw "Kaggle continued throttling relational output download" }
    if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Relational patch archive was not downloaded"
    }
    & python $verifier --archive $archivePath --report $verificationPath *> $null
    if ($LASTEXITCODE -ne 0) { throw "Relational patch archive verification failed" }
    $verification = Get-Content -Raw -LiteralPath $verificationPath | ConvertFrom-Json
    $archiveSha256 = [string]$verification.archive_sha256
    $runnerText = (Get-Content -Raw -LiteralPath $runnerTemplate).Replace(
        "__RELATIONAL_ARCHIVE_SHA256__", $archiveSha256
    )
    if ($runnerText -match "__RELATIONAL_ARCHIVE_SHA256__") {
        throw "Relational runner archive binding failed"
    }
    Set-Content -LiteralPath $renderedRunner -Encoding utf8 -Value $runnerText

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
    if (-not $credentialReady) { throw "Timed out waiting for refreshed AWS credentials" }

    $scpBase = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no")
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($trainer, $model, $inference, $probeScorer, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/research/temporal_contrastive/"))
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($renderedRunner, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/scripts/run-antelume-relational-division-sweep-v1.sh"))
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($archivePath, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub-relational-division-patches-v3.tar.gz"))
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($developmentInventory, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub-relational-development-inventory-v1.json"))
    Publish-Key
    $remoteCommand = @'
cd /home/ubuntu/biohub
chmod +x scripts/run-antelume-relational-division-sweep-v1.sh
bash -n scripts/run-antelume-relational-division-sweep-v1.sh
/home/ubuntu/venv/bin/python -m py_compile research/temporal_contrastive/relational_division_model.py research/temporal_contrastive/relational_division_inference.py research/temporal_contrastive/train_relational_division_sweep.py research/temporal_contrastive/score_relational_division_development_probe.py
test ! -e /home/ubuntu/biohub-results/competition-relational-division-sweep-v1
nohup bash scripts/run-antelume-relational-division-sweep-v1.sh >/home/ubuntu/biohub-logs/relational-division-sweep-controller-v1.log 2>&1 < /dev/null &
echo RELATIONAL_CONTROLLER_PID=$!
'@
    Invoke-Checked "ssh" @("-i", $SshKey, "-o", "StrictHostKeyChecking=no", "${RemoteUser}@${RemoteHost}", $remoteCommand)
    Write-Terminal @{
        status = "deployed_and_queued"
        kernel_ref = $kernelRef
        archive_sha256 = $archiveSha256
        archive_bytes = [int64]$verification.archive_bytes
        planned_model_count = 8
        steps_per_model = 15000
        relational_patch_encodings_per_step = 3
        remote_gpu = "one Antelume A10G"
        waits_for_current_probe_chain = $true
        audit_opened_during_training = $false
        final_probe_opened = $false
        development_probe_runs_only_after_audit_acceptance = $true
        authorized_for_submission = $false
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        planned_model_count = 8
        remote_mutation_performed = $false
        authorized_for_submission = $false
    }
    throw
}
