param(
    [string]$AwsProfile = "148971207977_InventoryOptimization-EC2-Access",
    [string]$AwsRegion = "us-east-1",
    [string]$SourceInstanceId = "i-0d12195df0d3558f3",
    [string]$InstanceType = "g5.12xlarge",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$PublicKey = "C:/Users/IndarKumar/.ssh/rsna_ec2.pub",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumCredentialPolls = 1440,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($PollSeconds -lt 15 -or $MaximumCredentialPolls -lt 1) {
    throw "Invalid AWS temporal-localizer controller bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/aws-4gpu-temporal-localizer-v1"
$syntheticRoot = Join-Path $RepositoryRoot ".biohub/cache/public-research-20260830/synthetic16"
$archivePath = Join-Path $stateRoot "biohub-synthetic16-temporal-localizer-v1.tar.gz"
$developmentArchivePath = Join-Path $stateRoot "biohub-temporal-localizer-development-v1.tar.gz"
$developmentStage = Join-Path $stateRoot "development-stage"
$probeRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-division-probe-frames-v1"
$controlRoot = Join-Path $RepositoryRoot ".biohub/cache/public-frontier-outputs-20260829/biohub-ct-0940-ema/tracking_repo/predictions/unknown/unet_transformer_val/split_0"
$truthRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-truth/public-node-acceptance-v1"
$runnerTemplate = Join-Path $RepositoryRoot "scripts/run-aws-4gpu-temporal-localizer-v1.sh"
$renderedRunner = Join-Path $stateRoot "run-aws-4gpu-temporal-localizer-v1.sh"
$launchTerminal = Join-Path $stateRoot "launch-terminal.json"
$harvestTerminal = Join-Path $stateRoot "harvest-terminal.json"
$logPath = Join-Path $stateRoot "controller.log"
$localResult = Join-Path $stateRoot "biohub-synthetic16-temporal-node-localizer-v1-results.tar.gz"
$requiredCode = @(
    (Join-Path $RepositoryRoot "research/synthetic_pretrain/data.py"),
    (Join-Path $RepositoryRoot "research/temporal_contrastive/patch_model.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/__init__.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/consensus.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/inference.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/model.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/score_real_development_probe.py"),
    (Join-Path $RepositoryRoot "research/temporal_localization/train_synthetic_localizer.py")
)

function Write-AtomicJson([string]$Path, [hashtable]$Payload) {
    $Payload.schema_version = 1
    $Payload.recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    $Payload.competition_submission_performed = $false
    $temporary = "$Path.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Append-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

function Wait-Credentials {
    for ($poll = 1; $poll -le $MaximumCredentialPolls; $poll++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & aws sts get-caller-identity --profile $AwsProfile --region $AwsRegion 2> $null | Out-Null
        $status = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($status -eq 0) { return }
        if ($poll -lt $MaximumCredentialPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    throw "Timed out waiting for refreshed AWS credentials"
}

function Publish-Key([string]$InstanceId, [string]$AvailabilityZone) {
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    if ($LASTEXITCODE -ne 0) { throw "Failed to publish the temporal-localizer SSH key" }
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
foreach ($required in @($runnerTemplate) + $requiredCode) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "AWS temporal-localizer deployment input is missing: $required"
    }
}
foreach ($requiredDirectory in @($probeRoot, $controlRoot, $truthRoot)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "Temporal-localizer development input is missing: $requiredDirectory"
    }
}
$sequenceFiles = @(0..15 | ForEach-Object {
    Join-Path $syntheticRoot ("sequences/seq_{0:D4}.npz" -f $_)
})
foreach ($sequence in $sequenceFiles) {
    if (-not (Test-Path -LiteralPath $sequence -PathType Leaf)) {
        throw "Synthetic16 source is incomplete: $sequence"
    }
}
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Push-Location -LiteralPath $syntheticRoot
    try {
        & tar -czf $archivePath README.md source_metadata.json source_manifest_full.json sequences
        if ($LASTEXITCODE -ne 0) { throw "Synthetic16 archive creation failed" }
    }
    finally { Pop-Location }
}
$probeManifest = Get-Content -Raw -LiteralPath (Join-Path $probeRoot "probe_cache_manifest.json") | ConvertFrom-Json
if (
    $probeManifest.status -ne "complete" -or
    $probeManifest.run_id -ne "competition-division-probe-frame-cache-v1" -or
    [int]$probeManifest.summary.files -ne 23 -or
    [int]$probeManifest.summary.frames -ne 15 -or
    [int]$probeManifest.summary.movies -ne 4 -or
    [int64]$probeManifest.summary.bytes -ne 66682915 -or
    $probeManifest.competition_test_data_read -ne $false -or
    $probeManifest.public_leaderboard_used_for_selection -ne $false
) {
    throw "Temporal-localizer real development manifest changed"
}
if (-not (Test-Path -LiteralPath $developmentArchivePath -PathType Leaf)) {
    New-Item -ItemType Directory -Path $developmentStage -Force | Out-Null
    Copy-Item -LiteralPath $probeRoot -Destination (Join-Path $developmentStage "probe") -Recurse
    Copy-Item -LiteralPath $controlRoot -Destination (Join-Path $developmentStage "control") -Recurse
    Copy-Item -LiteralPath $truthRoot -Destination (Join-Path $developmentStage "truth") -Recurse
    Push-Location -LiteralPath $developmentStage
    try {
        & tar -czf $developmentArchivePath probe control truth
        if ($LASTEXITCODE -ne 0) { throw "Temporal-localizer development archive creation failed" }
    }
    finally { Pop-Location }
}
$archiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
$developmentArchiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $developmentArchivePath).Hash.ToLowerInvariant()
$runnerText = (Get-Content -Raw -LiteralPath $runnerTemplate).Replace(
    "__SYNTHETIC16_ARCHIVE_SHA256__", $archiveSha256
).Replace(
    "__DEVELOPMENT_ARCHIVE_SHA256__", $developmentArchiveSha256
)
if ($runnerText -match "__(SYNTHETIC16|DEVELOPMENT)_ARCHIVE_SHA256__") {
    throw "Temporal-localizer runner hash binding failed"
}
Set-Content -LiteralPath $renderedRunner -Encoding utf8 -Value $runnerText
& python -m py_compile @requiredCode
if ($LASTEXITCODE -ne 0) { throw "Temporal-localizer Python validation failed" }
Push-Location -LiteralPath $RepositoryRoot
try {
    & bash -n "scripts/run-aws-4gpu-temporal-localizer-v1.sh"
    if ($LASTEXITCODE -ne 0) { throw "Temporal-localizer runner validation failed" }
}
finally { Pop-Location }
if ($ValidateOnly) {
    @{
        status = "validated"
        instance_type = $InstanceType
        planned_gpu_count = 4
        planned_model_count = 4
        parameters_per_model = 71249805
        steps_per_model = 20000
        archive_sha256 = $archiveSha256
        archive_bytes = (Get-Item -LiteralPath $archivePath).Length
        development_archive_sha256 = $developmentArchiveSha256
        development_archive_bytes = (Get-Item -LiteralPath $developmentArchivePath).Length
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $harvestTerminal) {
    throw "AWS temporal-localizer controller already reached a harvest terminal"
}

$newInstanceId = $null
try {
    Wait-Credentials
    $source = (& aws ec2 describe-instances `
        --profile $AwsProfile --region $AwsRegion --instance-ids $SourceInstanceId `
        --output json 2>> $logPath) -join "`n" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the Antelume source instance" }
    $sourceInstance = $source.Reservations[0].Instances[0]
    $imageId = [string]$sourceInstance.ImageId
    $subnetId = [string]$sourceInstance.SubnetId
    $securityGroups = @($sourceInstance.SecurityGroups | ForEach-Object { [string]$_.GroupId })
    if (-not $imageId -or -not $subnetId -or $securityGroups.Count -lt 1) {
        throw "Antelume source networking or AMI inventory is incomplete"
    }
    $launchArguments = @(
        "ec2", "run-instances",
        "--profile", $AwsProfile,
        "--region", $AwsRegion,
        "--image-id", $imageId,
        "--instance-type", $InstanceType,
        "--count", "1",
        "--subnet-id", $subnetId,
        "--security-group-ids"
    ) + $securityGroups + @(
        "--associate-public-ip-address",
        "--instance-initiated-shutdown-behavior", "stop",
        "--tag-specifications", "ResourceType=instance,Tags=[{Key=Name,Value=Biohub-Temporal-Localizer-4GPU},{Key=Project,Value=Biohub},{Key=AutoStop,Value=true}]",
        "--output", "json"
    )
    $launch = (& aws @launchArguments 2>> $logPath) -join "`n" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "AWS rejected the four-GPU temporal-localizer launch" }
    $newInstanceId = [string]$launch.Instances[0].InstanceId
    if (-not $newInstanceId) { throw "AWS launch returned no instance identifier" }
    Invoke-Checked "aws" @(
        "ec2", "wait", "instance-status-ok", "--profile", $AwsProfile,
        "--region", $AwsRegion, "--instance-ids", $newInstanceId
    )
    $description = (& aws ec2 describe-instances `
        --profile $AwsProfile --region $AwsRegion --instance-ids $newInstanceId `
        --output json 2>> $logPath) -join "`n" | ConvertFrom-Json
    $instance = $description.Reservations[0].Instances[0]
    $remoteHost = [string]$instance.PublicIpAddress
    $availabilityZone = [string]$instance.Placement.AvailabilityZone
    if (-not $remoteHost -or -not $availabilityZone) {
        throw "The four-GPU instance has no reachable public endpoint"
    }
    Publish-Key $newInstanceId $availabilityZone
    $sshBase = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20")
    $ready = $false
    for ($retry = 1; $retry -le 20; $retry++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & ssh @sshBase "${RemoteUser}@${remoteHost}" "echo READY" *> $null
        $status = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($status -eq 0) { $ready = $true; break }
        if ($retry -lt 20) {
            Start-Sleep -Seconds 15
            Publish-Key $newInstanceId $availabilityZone
        }
    }
    if (-not $ready) { throw "The four-GPU instance did not become SSH-ready" }
    Invoke-Checked "ssh" ($sshBase + @("${RemoteUser}@${remoteHost}", "mkdir -p /home/ubuntu/biohub/research/{synthetic_pretrain,temporal_contrastive,temporal_localization} /home/ubuntu/biohub/scripts /home/ubuntu/biohub-results"))
    Invoke-Checked "scp" ($sshBase + @($PublicKey, "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub-deploy.pub"))
    Invoke-Checked "ssh" ($sshBase + @("${RemoteUser}@${remoteHost}", "cat /home/ubuntu/biohub-deploy.pub >> /home/ubuntu/.ssh/authorized_keys"))
    Invoke-Checked "scp" ($sshBase + @(
        $requiredCode[0], "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub/research/synthetic_pretrain/data.py"
    ))
    Invoke-Checked "scp" ($sshBase + @(
        $requiredCode[1], "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub/research/temporal_contrastive/patch_model.py"
    ))
    Invoke-Checked "scp" ($sshBase + @(
        $requiredCode[2], $requiredCode[3], $requiredCode[4], $requiredCode[5],
        $requiredCode[6], $requiredCode[7],
        "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub/research/temporal_localization/"
    ))
    Invoke-Checked "scp" ($sshBase + @(
        $renderedRunner, "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub/scripts/run-aws-4gpu-temporal-localizer-v1.sh"
    ))
    Invoke-Checked "scp" ($sshBase + @(
        $archivePath, "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub-synthetic16-temporal-localizer-v1.tar.gz"
    ))
    Invoke-Checked "scp" ($sshBase + @(
        $developmentArchivePath, "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub-temporal-localizer-development-v1.tar.gz"
    ))
    $remoteCommand = "chmod +x /home/ubuntu/biohub/scripts/run-aws-4gpu-temporal-localizer-v1.sh && bash -n /home/ubuntu/biohub/scripts/run-aws-4gpu-temporal-localizer-v1.sh && nohup bash /home/ubuntu/biohub/scripts/run-aws-4gpu-temporal-localizer-v1.sh >/home/ubuntu/biohub-temporal-localizer-v1-controller.log 2>&1 < /dev/null & echo TEMPORAL_LOCALIZER_PID=`$!"
    Invoke-Checked "ssh" ($sshBase + @("${RemoteUser}@${remoteHost}", $remoteCommand))
    Write-AtomicJson $launchTerminal @{
        run_id = "aws-4gpu-temporal-localizer-launch-v1"
        status = "launched"
        instance_id = $newInstanceId
        instance_type = $InstanceType
        public_ip = $remoteHost
        availability_zone = $availabilityZone
        gpu_count = 4
        model_count = 4
        parameters_per_model = 71249805
        steps_per_model = 20000
        synthetic_archive_sha256 = $archiveSha256
        development_archive_sha256 = $developmentArchiveSha256
        development_probe_runs_only_after_member_audits = $true
        auto_stop_after_harvest_window = $true
        authorized_for_submission = $false
    }

    $remoteChecksum = "/home/ubuntu/biohub-synthetic16-temporal-node-localizer-v1-results.tar.gz.sha256"
    $resultReady = $false
    for ($poll = 1; $poll -le 480; $poll++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & ssh @sshBase "${RemoteUser}@${remoteHost}" "test -f $remoteChecksum" *> $null
        $status = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($status -eq 0) { $resultReady = $true; break }
        Start-Sleep -Seconds 120
    }
    if (-not $resultReady) { throw "Timed out waiting for the four-GPU temporal-localizer result" }
    Invoke-Checked "scp" ($sshBase + @(
        "${RemoteUser}@${remoteHost}:/home/ubuntu/biohub-synthetic16-temporal-node-localizer-v1-results.tar.gz",
        $localResult
    ))
    $checksumText = (& ssh @sshBase "${RemoteUser}@${remoteHost}" "cat $remoteChecksum") -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Could not read the remote temporal-localizer checksum" }
    $expectedSha256 = ($checksumText -split "\s+")[0].ToLowerInvariant()
    $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $localResult).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) { throw "Temporal-localizer harvest checksum mismatch" }
    Invoke-Checked "ssh" ($sshBase + @(
        "${RemoteUser}@${remoteHost}", "touch /home/ubuntu/biohub-temporal-localizer-v1-harvest-complete"
    ))
    Write-AtomicJson $harvestTerminal @{
        run_id = "aws-4gpu-temporal-localizer-harvest-v1"
        status = "harvested"
        instance_id = $newInstanceId
        result_sha256 = $actualSha256
        result_bytes = (Get-Item -LiteralPath $localResult).Length
        instance_auto_stop_requested = $true
        authorized_for_submission = $false
    }
}
catch {
    Write-AtomicJson $harvestTerminal @{
        run_id = "aws-4gpu-temporal-localizer-harvest-v1"
        status = "failed"
        instance_id = $newInstanceId
        error = $_.Exception.Message
        recoverable_instance_preserved = [bool]$newInstanceId
        authorized_for_submission = $false
    }
    throw
}
