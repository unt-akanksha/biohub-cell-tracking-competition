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
    throw "Invalid graph-context deployment wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$archiveSha256 = "efa3b5af80c75b2bc091ecda1e86064660dcfffe197e2a4950a12981248b0c3e"
$manifestSha256 = "2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9"
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-division-sweep-v1"
$priorDeployTerminal = Join-Path $RepositoryRoot ".biohub/cache/antelume-relational-division-sweep-v1/deployment-terminal.json"
$archivePath = Join-Path $RepositoryRoot ".biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1.tar.gz"
$manifestPath = Join-Path $RepositoryRoot ".biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1/graph_context_relational_patch_manifest.json"
$model = Join-Path $RepositoryRoot "research/temporal_contrastive/graph_context_division_model.py"
$trainer = Join-Path $RepositoryRoot "research/temporal_contrastive/train_graph_context_division_sweep.py"
$runnerTemplate = Join-Path $RepositoryRoot "scripts/run-antelume-graph-context-division-sweep-v1.sh"
$renderedRunner = Join-Path $stateRoot "run-antelume-graph-context-division-sweep-v1.sh"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"

function Write-Terminal([hashtable]$Payload) {
    $Payload.schema_version = 1
    $Payload.run_id = "antelume-graph-context-division-sweep-deploy-v1"
    $Payload.competition_submission_performed = $false
    $Payload.recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    $temporary = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Append-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Publish-Key {
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    if ($LASTEXITCODE -ne 0) { throw "Failed to publish graph-context SSH key" }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
foreach ($required in @($archivePath, $manifestPath, $model, $trainer, $runnerTemplate)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Graph-context deployment input is missing: $required"
    }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant() -ne $archiveSha256) {
    throw "Graph-context archive changed"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $manifestPath).Hash.ToLowerInvariant() -ne $manifestSha256) {
    throw "Graph-context manifest changed"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if (
    $manifest.status -ne "complete" -or
    $manifest.run_id -ne "competition-graph-context-relational-patches-v1" -or
    [int]$manifest.summary.rows -ne 3013 -or
    [int]$manifest.summary.positives -ne 134 -or
    [int]$manifest.summary.hard_negatives -ne 2879 -or
    $manifest.context_edges_read -ne $false -or
    $manifest.context_labels_used -ne $false -or
    $manifest.audit_labels_scored -ne $false -or
    $manifest.competition_test_data_read -ne $false -or
    $manifest.public_leaderboard_used_for_selection -ne $false
) {
    throw "Graph-context deployment manifest is ineligible"
}
$runnerText = (Get-Content -Raw -LiteralPath $runnerTemplate).Replace(
    "__GRAPH_CONTEXT_ARCHIVE_SHA256__", $archiveSha256
).Replace(
    "__GRAPH_CONTEXT_MANIFEST_SHA256__", $manifestSha256
)
if ($runnerText -match "__GRAPH_CONTEXT_(ARCHIVE|MANIFEST)_SHA256__") {
    throw "Graph-context runner hash binding failed"
}
Set-Content -LiteralPath $renderedRunner -Encoding utf8 -Value $runnerText
& python -m py_compile $model $trainer
if ($LASTEXITCODE -ne 0) { throw "Graph-context Python validation failed" }
Push-Location -LiteralPath $RepositoryRoot
try {
    & bash -n "scripts/run-antelume-graph-context-division-sweep-v1.sh"
    if ($LASTEXITCODE -ne 0) { throw "Graph-context runner validation failed" }
}
finally {
    Pop-Location
}
if ($ValidateOnly) {
    @{
        status = "validated"
        stage = "antelume_graph_context_division_sweep_deploy"
        archive_sha256 = $archiveSha256
        manifest_sha256 = $manifestSha256
        planned_model_count = 8
        parameters_per_model = 74732308
        steps_per_model = 20000
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    throw "Graph-context deployment already reached a terminal state"
}

try {
    $priorReady = $false
    for ($poll = 1; $poll -le $MaximumPolls; $poll++) {
        if (Test-Path -LiteralPath $priorDeployTerminal -PathType Leaf) {
            $priorReady = $true
            break
        }
        if ($poll -lt $MaximumPolls) { Start-Sleep -Seconds $PollSeconds }
    }
    if (-not $priorReady) { throw "Timed out waiting for relational deployment" }
    $prior = Get-Content -Raw -LiteralPath $priorDeployTerminal | ConvertFrom-Json
    if ($prior.status -ne "deployed_and_queued") {
        Write-Terminal @{
            status = "skipped_after_relational_deploy_failure"
            prior_status = $prior.status
            remote_mutation_performed = $false
            authorized_for_submission = $false
        }
        exit 0
    }

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
    Invoke-Checked "scp" ($scpBase + @($model, $trainer, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/research/temporal_contrastive/"))
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($renderedRunner, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub/scripts/run-antelume-graph-context-division-sweep-v1.sh"))
    Publish-Key
    Invoke-Checked "scp" ($scpBase + @($archivePath, "${RemoteUser}@${RemoteHost}:/home/ubuntu/biohub-graph-context-relational-patches-v1.tar.gz"))
    Publish-Key
    $remoteCommand = @'
cd /home/ubuntu/biohub
chmod +x scripts/run-antelume-graph-context-division-sweep-v1.sh
bash -n scripts/run-antelume-graph-context-division-sweep-v1.sh
/home/ubuntu/venv/bin/python -m py_compile research/temporal_contrastive/graph_context_division_model.py research/temporal_contrastive/train_graph_context_division_sweep.py
test ! -e /home/ubuntu/biohub-results/competition-graph-context-division-sweep-v1
nohup bash scripts/run-antelume-graph-context-division-sweep-v1.sh >/home/ubuntu/biohub-logs/graph-context-division-sweep-controller-v1.log 2>&1 < /dev/null &
echo GRAPH_CONTEXT_CONTROLLER_PID=$!
'@
    Invoke-Checked "ssh" @("-i", $SshKey, "-o", "StrictHostKeyChecking=no", "${RemoteUser}@${RemoteHost}", $remoteCommand)
    Write-Terminal @{
        status = "deployed_and_queued"
        archive_sha256 = $archiveSha256
        archive_bytes = (Get-Item -LiteralPath $archivePath).Length
        manifest_sha256 = $manifestSha256
        planned_model_count = 8
        parameters_per_model = 74732308
        steps_per_model = 20000
        remote_gpu = "one Antelume A10G"
        waits_for_relational_sweep_terminal = $true
        waits_for_gpu_idle = $true
        ensemble_members_precommitted_before_audit = $true
        model_subset_searched_on_audit = $false
        final_probe_opened = $false
        remote_mutation_performed = $true
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
