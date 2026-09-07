param(
    [string]$AwsProfile = "148971207977_InventoryOptimization-EC2-Access",
    [string]$AwsRegion = "us-east-1",
    [string]$InstanceId = "i-0d12195df0d3558f3",
    [string]$AvailabilityZone = "us-east-1b",
    [string]$RemoteUser = "ubuntu",
    [string]$SshKey = "C:/Users/IndarKumar/.ssh/rsna_ec2",
    [string]$PublicKey = "C:/Users/IndarKumar/.ssh/rsna_ec2.pub",
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$CredentialPollSeconds = 60,
    [int]$MaximumCredentialPolls = 14400,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($CredentialPollSeconds -lt 15 -or $MaximumCredentialPolls -lt 1) {
    throw "Invalid graph-context v2 credential wait bounds"
}
Set-Location -LiteralPath $RepositoryRoot
$stateRoot = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2-deploy"
$terminalPath = Join-Path $stateRoot "deployment-terminal.json"
$logPath = Join-Path $stateRoot "deployment.log"
$archive = Join-Path $RepositoryRoot ".biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1.tar.gz"
$manifest = Join-Path $RepositoryRoot ".biohub/cache/graph-context-relational-v1/biohub_graph_context_relational_patches_v1/graph_context_relational_patch_manifest.json"
$inventory = Join-Path $RepositoryRoot ".biohub/results/competition-graph-context-development-inventory-v1.json"
$trainer = Join-Path $RepositoryRoot "research/temporal_contrastive/train_graph_context_division_sweep.py"
$scorer = Join-Path $RepositoryRoot "research/temporal_contrastive/score_graph_context_division_development_probe.py"
$model = Join-Path $RepositoryRoot "research/temporal_contrastive/graph_context_division_model.py"
$inference = Join-Path $RepositoryRoot "research/temporal_contrastive/graph_context_division_inference.py"
$runnerTemplate = Join-Path $RepositoryRoot "scripts/run-antelume-graph-context-frozen-ensemble-v2.sh"
$renderedRunner = Join-Path $stateRoot "run-antelume-graph-context-frozen-ensemble-v2.sh"
$archiveSha = "efa3b5af80c75b2bc091ecda1e86064660dcfffe197e2a4950a12981248b0c3e"
$manifestSha = "2e5c4c46b11ff1480194c29e88c0e16b0ebf2f0fd65b2fc29548d26fee599bd9"
$inventorySha = "c8883e77abcf76c5a837c5fd0b21afdfefb2e51cb8e550e3a81f69d70562f311"

function Hash([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Write-Terminal([hashtable]$Payload) {
    $Payload["schema_version"] = 1
    $Payload["run_id"] = "antelume-graph-context-frozen-ensemble-v2-deploy"
    $Payload["recorded_at"] = [DateTimeOffset]::UtcNow.ToString("o")
    $Payload["competition_submission_performed"] = $false
    $Payload["authorized_for_submission"] = $false
    $partial = "$terminalPath.partial"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $partial -Encoding utf8
    Move-Item -LiteralPath $partial -Destination $terminalPath -Force
}

function Publish-Key {
    & aws ec2-instance-connect send-ssh-public-key `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-id $InstanceId `
        --availability-zone $AvailabilityZone `
        --instance-os-user $RemoteUser `
        --ssh-public-key "file://$PublicKey" *> $null
    if ($LASTEXITCODE -ne 0) { throw "Failed to publish Antelume SSH key" }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
foreach ($required in @($archive, $manifest, $inventory, $trainer, $scorer, $model, $inference, $runnerTemplate)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Graph-context v2 input is missing: $required"
    }
}
if ((Hash $archive) -ne $archiveSha -or (Hash $manifest) -ne $manifestSha -or (Hash $inventory) -ne $inventorySha) {
    throw "Graph-context v2 immutable data input changed"
}
& python -m py_compile $trainer $scorer $model $inference
if ($LASTEXITCODE -ne 0) { throw "Graph-context v2 Python preflight failed" }
& bash -n "scripts/run-antelume-graph-context-frozen-ensemble-v2.sh"
if ($LASTEXITCODE -ne 0) { throw "Graph-context v2 runner preflight failed" }

$trainerSha = Hash $trainer
$scorerSha = Hash $scorer
$modelSha = Hash $model
$inferenceSha = Hash $inference
$runnerText = (Get-Content -Raw -LiteralPath $runnerTemplate).Replace(
    "__GRAPH_CONTEXT_ARCHIVE_SHA256__", $archiveSha
).Replace(
    "__GRAPH_CONTEXT_MANIFEST_SHA256__", $manifestSha
).Replace(
    "__GRAPH_CONTEXT_TRAINER_SHA256__", $trainerSha
).Replace(
    "__GRAPH_CONTEXT_SCORER_SHA256__", $scorerSha
).Replace(
    "__GRAPH_CONTEXT_MODEL_SHA256__", $modelSha
).Replace(
    "__GRAPH_CONTEXT_INFERENCE_SHA256__", $inferenceSha
)
if ($runnerText -match "__GRAPH_CONTEXT_[A-Z_]+__") {
    throw "Graph-context v2 runner hash binding failed"
}
$runnerText | Set-Content -LiteralPath $renderedRunner -Encoding utf8
& bash -n ".biohub/cache/antelume-graph-context-frozen-ensemble-v2-deploy/run-antelume-graph-context-frozen-ensemble-v2.sh"
if ($LASTEXITCODE -ne 0) { throw "Rendered graph-context v2 runner is invalid" }

if ($ValidateOnly) {
    @{
        status = "validated"
        planned_model_count = 8
        parameters_per_model = 74732308
        steps_per_model = 20000
        policy_contract = "all-selection-admitted-equal-rank-ensemble-v2"
        trainer_sha256 = $trainerSha
        scorer_sha256 = $scorerSha
    } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) {
    $existing = Get-Content -Raw -LiteralPath $terminalPath | ConvertFrom-Json
    if ($existing.status -eq "failed") {
        Remove-Item -LiteralPath $terminalPath
    }
    else {
        throw "Graph-context v2 deployment already reached a terminal state"
    }
}

try {
    $credentialReady = $false
    for ($poll = 1; $poll -le $MaximumCredentialPolls; $poll++) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & aws sts get-caller-identity --profile $AwsProfile --region $AwsRegion 2> $null | Out-Null
        $credentialStatus = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($credentialStatus -eq 0) {
            $credentialReady = $true
            break
        }
        if ($poll -lt $MaximumCredentialPolls) {
            Start-Sleep -Seconds $CredentialPollSeconds
        }
    }
    if (-not $credentialReady) { throw "Timed out waiting for refreshed AWS credentials" }
    $remoteHost = (& aws ec2 describe-instances `
        --profile $AwsProfile `
        --region $AwsRegion `
        --instance-ids $InstanceId `
        --query "Reservations[0].Instances[0].PublicIpAddress" `
        --output text).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remoteHost -or $remoteHost -eq "None") {
        throw "Antelume public address is unavailable"
    }
    $remote = "${RemoteUser}@${remoteHost}"
    $sshArgs = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no")
    Publish-Key
    Invoke-Checked "ssh.exe" ($sshArgs + @($remote, "set -euo pipefail; test ! -e /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run.sh; test ! -e /home/ubuntu/biohub-graph-context-frozen-ensemble-v2-results.tar.gz; mkdir -p /home/ubuntu/biohub-graph-context-frozen-ensemble-v2 /home/ubuntu/biohub/research/temporal_contrastive /home/ubuntu/biohub/scripts /home/ubuntu/biohub-logs"))
    foreach ($source in @($trainer, $scorer, $model, $inference)) {
        Publish-Key
        Invoke-Checked "scp.exe" ($sshArgs + @($source, "${remote}:/home/ubuntu/biohub/research/temporal_contrastive/"))
    }
    Publish-Key
    Invoke-Checked "scp.exe" ($sshArgs + @($renderedRunner, "${remote}:/home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run.sh"))
    Publish-Key
    Invoke-Checked "scp.exe" ($sshArgs + @($inventory, "${remote}:/home/ubuntu/biohub-graph-context-development-inventory-v1.json"))
    Publish-Key
    Invoke-Checked "scp.exe" ($sshArgs + @($archive, "${remote}:/home/ubuntu/biohub-graph-context-relational-patches-v1.tar.gz"))
    Publish-Key
    $launch = "set -euo pipefail; chmod +x /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run.sh; bash -n /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run.sh; nohup bash /home/ubuntu/biohub-graph-context-frozen-ensemble-v2/run.sh >/home/ubuntu/biohub-logs/graph-context-frozen-ensemble-v2.log 2>&1 < /dev/null & echo PID=`$!"
    Invoke-Checked "ssh.exe" ($sshArgs + @($remote, $launch))
    Write-Terminal @{
        status = "deployed_and_queued"
        remote_host = $remoteHost
        waits_for_v27_verified_harvest = $true
        waits_for_gpu_idle = $true
        planned_model_count = 8
        parameters_per_model = 74732308
        steps_per_model = 20000
        worst_case_gpu_hours = 5
        policy_contract = "all-selection-admitted-equal-rank-ensemble-v2"
        policy_unit_audited = $true
        constituent_audit_gate_required = $false
        trainer_sha256 = $trainerSha
        scorer_sha256 = $scorerSha
    }
}
catch {
    Write-Terminal @{
        status = "failed"
        error = $_.Exception.Message
        remote_mutation_performed = $false
    }
    throw
}
