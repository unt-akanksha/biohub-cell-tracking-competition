param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $workspace
$reports = Join-Path $workspace 'reports/experiments'
$pythonExe = Join-Path $workspace '.biohub/cache/graph-analysis-venv/Scripts/python.exe'
$evaluator = Join-Path $PSScriptRoot 'evaluate-trajectory-event-ensemble-selection-v1.py'
$contractPath = Join-Path $reports 'trajectory-event-ensemble-selection-v1-contract.json'
$fitPath = Join-Path $reports 'trajectory-event-fork16-fit-6bba-v1.json'
$receiptPath = Join-Path $reports 'trajectory-event-ensemble-selection-v1-queue.json'
$lockPath = Join-Path $reports 'trajectory-event-ensemble-selection-v1-queue.lock'
function Get-Digest([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$expectedContract = '57dde40d18ceadf25dcaf4bca9de1631c0663d751aa83b31914e8a1ed7dfae20'
if ((Get-Digest $contractPath) -ne $expectedContract) { throw 'Frozen comparison contract changed' }
$contract = (Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json).contract
function Assert-Inputs {
    if ((Get-Digest $contractPath) -ne $expectedContract -or (Get-Digest $evaluator) -ne $contract.source_sha256) { throw 'Frozen comparison code changed' }
    if ((Get-Digest (Join-Path $workspace 'research/trajectory_event_ensemble_selection_v1.py')) -ne $contract.helper_sha256) { throw 'Frozen selection gates changed' }
    if ((Get-Digest (Join-Path $reports 'trajectory-event-ensemble-selection-v1-design.md')) -ne $contract.design_sha256) { throw 'Frozen comparison design changed' }
}
Assert-Inputs
$fit = Get-Content -LiteralPath $fitPath -Raw | ConvertFrom-Json
$expectedPid = 36456
if ($fit.status -ne 'running' -or $fit.pid -ne $expectedPid) { throw 'Expected live second fit is absent; inspect before scheduling' }
$owned = Get-Process -Id $expectedPid
$null = $owned.Handle
$identity = Get-CimInstance Win32_Process -Filter "ProcessId = $expectedPid"
if ($owned.HasExited -or $identity.CommandLine -notlike '*train-trajectory-event-fork16-source-v1.py*--embryo 6bba --resume*') { throw 'Training process identity mismatch' }
foreach ($path in @($receiptPath, $lockPath, (Join-Path $reports 'trajectory-event-ensemble-selection-v1-smoke.json'), (Join-Path $reports 'trajectory-event-ensemble-selection-v1.json'))) {
    if (Test-Path -LiteralPath $path) { throw 'Existing comparison or owner; do not duplicate' }
}
if ($ValidateOnly) {
    [ordered]@{status='comparison_successor_preflight_passed';training_pid=$expectedPid;
        training_start_utc=$owned.StartTime.ToUniversalTime().ToString('o');writes_performed=$false} | ConvertTo-Json -Compress
    exit 0
}
$lockStream = [IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$receipt = [ordered]@{status='waiting_for_exact_final_fit';pid=$PID;training_pid=$expectedPid;
    training_start_utc=$owned.StartTime.ToUniversalTime().ToString('o');started_utc=[DateTime]::UtcNow.ToString('o');
    contract_sha256=$expectedContract;gpu_used=$false;existing_held_evaluation_controller_unchanged=$true;
    submission_allowed=$false;restart_allowed=$false;stages=@()}
function Save-Receipt {
    $receipt.updated_utc = [DateTime]::UtcNow.ToString('o')
    $temporary = "$receiptPath.tmp"
    [IO.File]::WriteAllText($temporary,($receipt | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
}
try {
    Save-Receipt
    $waitStart = [DateTime]::UtcNow
    while (-not $owned.WaitForExit(30000)) {
        if (([DateTime]::UtcNow - $waitStart).TotalMinutes -gt 90) { throw 'Wait cap reached; training untouched, inspect same handle' }
    }
    $fit = Get-Content -LiteralPath $fitPath -Raw | ConvertFrom-Json
    if ($fit.status -ne 'source_event_training_complete' -or $fit.steps -ne 13671 -or -not $fit.model_fitted) { throw "Fit ended as $($fit.status); no comparison/retry" }
    $receipt.final_fit_sha256 = Get-Digest $fitPath
    Assert-Inputs
    foreach ($stage in @('smoke', 'full')) {
        $receipt.status = "running_$stage"; Save-Receipt
        & $pythonExe $evaluator $stage
        if ($LASTEXITCODE -ne 0) { throw "$stage comparison failed; inspect, no retry" }
        $suffix = if ($stage -eq 'smoke') {'-smoke'} else {''}
        $resultPath = Join-Path $reports "trajectory-event-ensemble-selection-v1$suffix.json"
        $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        $expectedStatus = if ($stage -eq 'smoke') {'selection_inference_smoke_passed'} else {'fixed_ensemble_selection_scoring_complete'}
        if ($result.status -ne $expectedStatus) { throw 'Unexpected comparison terminal status' }
        $receipt.stages += [ordered]@{stage=$stage;status=$result.status;sha256=(Get-Digest $resultPath)}
        Save-Receipt
    }
    $receipt.status = 'comparison_completed_no_automatic_promotion'; Save-Receipt
} catch {
    $receipt.status = 'stopped_requires_inspection'; $receipt.error = $_.Exception.Message; Save-Receipt
    throw
} finally {
    $lockStream.Dispose()
}
