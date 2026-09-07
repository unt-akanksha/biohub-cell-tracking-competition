param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 480.0,
    [int]$PollSeconds = 120,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($MaximumWaitHours -le 0 -or $PollSeconds -lt 30) { throw "Invalid V31 conditional wait bounds" }
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot
$automationRoot = Join-Path $RepositoryRoot ".biohub/automation"
$terminalPath = Join-Path $automationRoot "peak-rank-xl-hard-mined-conditional-training-controller-v31.json"
$logPath = Join-Path $automationRoot "peak-rank-xl-hard-mined-conditional-training-controller-v31.log"
$v21ValidationPath = Join-Path $automationRoot "peak-rank-expanded-real-xl-balanced-validation-controller-v21.json"
$v27ValidationPath = Join-Path $automationRoot "peak-rank-hard-mined-temporal-snr-validation-controller-v27.json"
$graphHarvestPath = Join-Path $RepositoryRoot ".biohub/cache/antelume-graph-context-frozen-ensemble-v2/harvest-terminal.json"
$deployScript = Join-Path $RepositoryRoot "scripts/deploy-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.ps1"
$harvestScript = Join-Path $RepositoryRoot "scripts/wait-harvest-antelume-peak-rank-xl-hard-mined-temporal-snr-v31.ps1"
$validationScript = Join-Path $RepositoryRoot "scripts/wait-launch-evaluate-peak-rank-validation-v1.ps1"
$candidateScript = Join-Path $RepositoryRoot "scripts/wait-build-verify-submit-peak-rank-candidate.ps1"

function Write-Terminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = "peak-rank-xl-hard-mined-conditional-training-controller-v31"
        status = $Status
        competition_submission_performed = $false
        authorized_for_submission = $false
        public_leaderboard_used_for_selection = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    foreach ($key in $Evidence.Keys) { $payload[$key] = $Evidence[$key] }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        ([DateTimeOffset]::Now.ToString("o")) + " " + $Message
    )
}

function Is-Cleanly-Promoted($Payload) {
    return (
        $Payload.status -eq "completed" -and
        $Payload.promotion_passed -eq $true -and
        $Payload.accepted_for_candidate_integration -eq $true -and
        $Payload.competition_submission_performed -eq $false
    )
}

New-Item -ItemType Directory -Path $automationRoot -Force | Out-Null
foreach ($required in @($deployScript, $harvestScript, $validationScript, $candidateScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "V31 conditional input is missing: $required"
    }
}
if ($ValidateOnly) {
    $errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile(
        $PSCommandPath, [ref]$null, [ref]$errors
    )
    if ($errors.Count -ne 0) { throw "V31 conditional controller syntax is invalid" }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $deployScript -ValidateOnly *> $null
    if ($LASTEXITCODE -ne 0) { throw "V31 deploy preflight failed" }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $harvestScript -ValidateOnly *> $null
    if ($LASTEXITCODE -ne 0) { throw "V31 harvest preflight failed" }
    @{ status = "validated"; conditional_on = @("v21_clean_promotion", "v27_clean_promotion", "graph_v2_verified_harvest") } | ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $terminalPath) { throw "V31 conditional controller already reached a terminal state" }

$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
try {
    $dependencies = @($v21ValidationPath, $v27ValidationPath, $graphHarvestPath)
    while (@($dependencies | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -gt 0) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) { throw "Timed out waiting for V31 evidence gates" }
        Start-Sleep -Seconds $PollSeconds
    }
    $v21 = Get-Content -Raw -LiteralPath $v21ValidationPath | ConvertFrom-Json
    $v27 = Get-Content -Raw -LiteralPath $v27ValidationPath | ConvertFrom-Json
    $graph = Get-Content -Raw -LiteralPath $graphHarvestPath | ConvertFrom-Json
    if (-not (Is-Cleanly-Promoted $v21) -or -not (Is-Cleanly-Promoted $v27)) {
        Write-Terminal "skipped_after_parent_rejection" @{
            v21_status = $v21.status
            v21_promotion_passed = $v21.promotion_passed
            v27_status = $v27.status
            v27_promotion_passed = $v27.promotion_passed
            graph_status = $graph.status
            aws_gpu_job_launched = $false
        }
        exit 0
    }
    if (-not (
        @("accepted", "scientifically_rejected") -contains $graph.status -and
        $graph.remote_harvest_acknowledged -eq $true -and
        $graph.competition_submission_performed -eq $false
    )) {
        Write-Terminal "skipped_after_graph_harvest_failure" @{
            v21_promotion_passed = $true
            v27_promotion_passed = $true
            graph_status = $graph.status
            graph_remote_harvest_acknowledged = $graph.remote_harvest_acknowledged
            aws_gpu_job_launched = $false
        }
        exit 0
    }

    Write-Log "all clean parent gates passed; deploying precommitted V31"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $deployScript
    if ($LASTEXITCODE -ne 0) { throw "V31 deploy failed" }
    $harvestProcess = Start-Process powershell.exe -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $harvestScript
    ) -WindowStyle Hidden -PassThru
    $validationProcess = Start-Process powershell.exe -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $validationScript,
        "-RepositoryRoot", $RepositoryRoot, "-Variant", "xl-hard-mined-temporal-snr-v31",
        "-MaximumWaitHours", "180", "-PollSeconds", "120"
    ) -WindowStyle Hidden -PassThru
    $candidateProcess = Start-Process powershell.exe -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $candidateScript,
        "-RepositoryRoot", $RepositoryRoot, "-Variant", "xl-hard-mined-temporal-snr-v31",
        "-MaximumWaitHours", "180", "-PollSeconds", "120"
    ) -WindowStyle Hidden -PassThru
    Write-Terminal "deployed_and_controllers_started" @{
        v21_validation_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $v21ValidationPath).Hash.ToLowerInvariant()
        v27_validation_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $v27ValidationPath).Hash.ToLowerInvariant()
        graph_harvest_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $graphHarvestPath).Hash.ToLowerInvariant()
        parameter_count = 129761606
        steps = 6000
        maximum_wall_seconds = 108000
        aws_gpu_job_launched = $true
        harvest_waiter_pid = $harvestProcess.Id
        validation_controller_pid = $validationProcess.Id
        candidate_controller_pid = $candidateProcess.Id
    }
}
catch {
    Write-Terminal "failed" @{ error = $_.Exception.Message; aws_gpu_job_launched = $false }
    Write-Log "failed error=$($_.Exception.Message)"
    throw
}
