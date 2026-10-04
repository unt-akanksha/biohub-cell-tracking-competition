param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$reports = Join-Path $workspace 'reports/experiments'
$fitName = 'trajectory-event-fork16-fit-6bba-v1'
$fitPath = Join-Path $reports "$fitName.json"
$fitRoot = Join-Path $workspace ".biohub/cache/$fitName"
$fitContract = Join-Path $fitRoot 'CONTRACT.json'
$trainer = Join-Path $PSScriptRoot 'train-trajectory-event-fork16-source-v1.py'
$evaluator = Join-Path $PSScriptRoot 'evaluate-trajectory-event-fork16-held-v1.py'
$pythonExe = Join-Path $workspace '.biohub/cache/graph-analysis-venv/Scripts/python.exe'
$name = 'trajectory-event-fork16-6bba-resume-evaluation-v1'
$receiptPath = Join-Path $reports "$name.json"
$lockPath = Join-Path $reports "$name.lock"
$expectedContract = 'ea0bcdfe8c7bb18ca56e55a929b32451eeafa96cf92791dce70f6798dd2ce111'
$expectedTrainer = 'd13c1798ad4327e2d095c392605d481fc546b02569ad201f88e672a9e2ab796f'
$expectedEvaluator = '0d6716829382e06ae5e476d8e0789104cd9f3ac0663722c040f7f898f6390f37'
function Get-Digest([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Assert-Inputs {
    if ((Get-Digest $fitContract) -ne $expectedContract -or
        (Get-Digest $trainer) -ne $expectedTrainer -or
        (Get-Digest $evaluator) -ne $expectedEvaluator) { throw 'Frozen code/contract changed' }
    $control = Get-Content -LiteralPath (Join-Path $reports 'trajectory-event-fork16-held-44b6-v1-control-preflight.json') -Raw | ConvertFrom-Json
    if ($control.status -ne 'untrained_control_preflight_passed' -or $control.ground_truth_opened) { throw 'Control preflight required' }
}
Assert-Inputs
$launch = Get-Content -LiteralPath (Join-Path $reports 'trajectory-event-fork16-full-fit-launch-v1.json') -Raw | ConvertFrom-Json
$expectedPid = [int]$launch.workers.'6bba'.actual_python_pid
if ($expectedPid -ne 43524) { throw 'Original process identity changed' }
$fit = Get-Content -LiteralPath $fitPath -Raw | ConvertFrom-Json
if ($fit.status -ne 'running' -or $fit.pid -ne $expectedPid) { throw 'Require original live fit; inspect any terminal state separately' }
$owned = Get-Process -Id $expectedPid
$null = $owned.Handle
$identity = Get-CimInstance Win32_Process -Filter "ProcessId = $expectedPid"
if ($owned.HasExited -or $identity.CommandLine -notlike '*train-trajectory-event-fork16-source-v1.py*--embryo 6bba --resume*') { throw 'Live training command did not match' }
foreach ($suffix in @('','-smoke','-queue')) {
    if (Test-Path -LiteralPath (Join-Path $reports "trajectory-event-fork16-held-44b6-v1$suffix.json")) { throw 'Existing evaluation or owner; do not duplicate' }
}
if ((Test-Path -LiteralPath $receiptPath) -or (Test-Path -LiteralPath $lockPath)) { throw 'Existing resume controller; inspect it' }
if ($ValidateOnly) {
    [ordered]@{status='resume_successor_preflight_passed';training_pid=$expectedPid;
      training_start_utc=$owned.StartTime.ToUniversalTime().ToString('o');writes_performed=$false} | ConvertTo-Json -Compress
    exit 0
}
$lockStream = [IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$receipt = [ordered]@{status='waiting_for_original_safe_pause';pid=$PID;original_training_pid=$expectedPid;
    original_start_utc=$owned.StartTime.ToUniversalTime().ToString('o');started_utc=[DateTime]::UtcNow.ToString('o');
    fit_contract_sha256=$expectedContract;trainer_sha256=$expectedTrainer;evaluator_sha256=$expectedEvaluator;
    gpu_used=$false;submission_allowed=$false;maximum_resumes=1;stages=@()}
function Save-Receipt {
    $receipt.updated_utc=[DateTime]::UtcNow.ToString('o')
    $temporary="$receiptPath.tmp"
    [IO.File]::WriteAllText($temporary,($receipt | ConvertTo-Json -Depth 16),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
}
try {
    Save-Receipt
    $waitStart=[DateTime]::UtcNow
    while (-not $owned.WaitForExit(30000)) {
        if (([DateTime]::UtcNow-$waitStart).TotalMinutes -gt 20) { throw 'Original pause wait exceeded20minutes; training untouched' }
    }
    $fit=Get-Content -LiteralPath $fitPath -Raw | ConvertFrom-Json
    if ($fit.status -ne 'paused_at_wall_limit' -or $fit.pid -ne $expectedPid) { throw "Original ended as $($fit.status); no automatic retry" }
    Assert-Inputs
    $pointer=Get-Content -LiteralPath (Join-Path $fitRoot 'latest-checkpoint.json') -Raw | ConvertFrom-Json
    $checkpoint=[IO.Path]::GetFullPath((Join-Path $fitRoot $pointer.path))
    if ([IO.Path]::GetDirectoryName($checkpoint) -ne [IO.Path]::GetFullPath($fitRoot) -or
        (Get-Digest $checkpoint) -ne $pointer.sha256 -or $fit.checkpoint_sha256 -ne $pointer.sha256) { throw 'Checkpoint containment/hash failed' }
    $saved=Get-Content -LiteralPath $checkpoint -Raw | ConvertFrom-Json
    if ($saved.contract_sha256 -ne $expectedContract -or $saved.optimizer.steps -ne $fit.steps) { throw 'Checkpoint state/contract mismatch' }
    $receipt.paused_fit=$fit
    $receipt.paused_fit_sha256=Get-Digest $fitPath
    $receipt.resume_checkpoint=$pointer
    $receipt.status='resuming_exact_checkpoint';Save-Receipt
    & $pythonExe $trainer --embryo 6bba --resume
    if ($LASTEXITCODE -ne 0) { throw 'Resumed training failed; inspect checkpoint, no retry' }
    $fit=Get-Content -LiteralPath $fitPath -Raw | ConvertFrom-Json
    if ($fit.status -ne 'source_event_training_complete' -or $fit.steps -ne 13671 -or -not $fit.model_fitted) { throw "Resumed fit ended as $($fit.status); no evaluation" }
    $receipt.completed_fit_sha256=Get-Digest $fitPath
    Assert-Inputs
    foreach ($stage in @('smoke','full')) {
        $receipt.status="running_$stage";Save-Receipt
        $arguments=@($evaluator,'--fit-embryo','6bba')
        if ($stage -eq 'smoke') { $arguments+='--smoke' }
        & $pythonExe @arguments
        if ($LASTEXITCODE -ne 0) { throw "$stage evaluation failed; no retry" }
        $suffix=if ($stage -eq 'smoke') {'-smoke'} else {''}
        $outputPath=Join-Path $reports "trajectory-event-fork16-held-44b6-v1$suffix.json"
        $output=Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
        $expectedStatus=if ($stage -eq 'smoke') {'complete_movie_inference_smoke_passed'} else {'held_embryo_complete_movie_scoring_finished'}
        if ($output.status -ne $expectedStatus) { throw 'Unexpected evaluation terminal status' }
        $receipt.stages += [ordered]@{stage=$stage;status=$output.status;sha256=(Get-Digest $outputPath)}
        Save-Receipt
    }
    $receipt.status='held_evaluation_completed_no_automatic_promotion';Save-Receipt
} catch {
    $receipt.status='stopped_requires_inspection';$receipt.error=$_.Exception.Message;Save-Receipt
    throw
} finally {
    $lockStream.Dispose()
}
