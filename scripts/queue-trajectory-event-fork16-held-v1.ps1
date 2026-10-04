param(
    [Parameter(Mandatory=$true)][ValidateSet('44b6','6bba')][string]$FitEmbryo,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonExe = Join-Path $workspace '.biohub/cache/graph-analysis-venv/Scripts/python.exe'
$reportRoot = Join-Path $workspace 'reports/experiments'
$fitName = "trajectory-event-fork16-fit-$FitEmbryo-v1"
$fitReportPath = Join-Path $reportRoot "$fitName.json"
$fitContractPath = Join-Path $workspace ".biohub/cache/$fitName/CONTRACT.json"
$launch = Get-Content -LiteralPath (Join-Path $reportRoot 'trajectory-event-fork16-full-fit-launch-v1.json') -Raw | ConvertFrom-Json
$expected = $launch.workers.$FitEmbryo
$fit = Get-Content -LiteralPath $fitReportPath -Raw | ConvertFrom-Json
$evaluator = Join-Path $PSScriptRoot 'evaluate-trajectory-event-fork16-held-v1.py'
$evaluatorHash = '0d6716829382e06ae5e476d8e0789104cd9f3ac0663722c040f7f898f6390f37'
$held = if ($FitEmbryo -eq '44b6') {'6bba'} else {'44b6'}
$queueName = "trajectory-event-fork16-held-$held-v1-queue"
$queuePath = Join-Path $reportRoot "$queueName.json"
$queueLock = Join-Path $reportRoot "$queueName.lock"

function Assert-Inputs {
    if ((Get-FileHash -LiteralPath $fitContractPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.contract_sha256) {
        throw 'Frozen fit contract changed'
    }
    if ((Get-FileHash -LiteralPath $evaluator -Algorithm SHA256).Hash.ToLowerInvariant() -ne $evaluatorHash) {
        throw 'Frozen evaluator changed'
    }
    $controlPath = Join-Path $reportRoot "trajectory-event-fork16-held-$held-v1-control-preflight.json"
    $control = Get-Content -LiteralPath $controlPath -Raw | ConvertFrom-Json
    if ($control.status -ne 'untrained_control_preflight_passed' -or $control.ground_truth_opened -or $control.source_sha256 -ne $evaluatorHash) {
        throw 'Matching completed control preflight required'
    }
}

Assert-Inputs
if ($fit.status -ne 'running' -or $fit.pid -ne $expected.actual_python_pid) { throw 'Queue only the verified original live fit' }
$ownedProcess = Get-Process -Id $fit.pid
$null = $ownedProcess.Handle # Hold this process identity, not a future recycled PID.
$identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($fit.pid)"
if ($ownedProcess.HasExited -or $identity.CommandLine -notlike '*train-trajectory-event-fork16-source-v1.py*' -or
    $identity.CommandLine -notlike "*--embryo $FitEmbryo*" -or $identity.CommandLine -notlike '*--resume*') {
    throw 'Training process identity did not match'
}
foreach ($suffix in @('','-smoke')) {
    if (Test-Path -LiteralPath (Join-Path $reportRoot "trajectory-event-fork16-held-$held-v1$suffix.json")) {
        throw 'An evaluation already exists; inspect it rather than duplicate'
    }
}
if ((Test-Path -LiteralPath $queuePath) -or (Test-Path -LiteralPath $queueLock)) { throw 'Preserve existing successor' }
if ($ValidateOnly) {
    [ordered]@{status='successor_preflight_passed';fit_embryo=$FitEmbryo;held_embryo=$held;training_pid=$fit.pid;
        training_start_utc=$ownedProcess.StartTime.ToUniversalTime().ToString('o');writes_performed=$false} | ConvertTo-Json -Compress
    exit 0
}
$lockStream = [IO.File]::Open($queueLock,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$receipt = [ordered]@{status='waiting_for_exact_training_process';pid=$PID;fit_embryo=$FitEmbryo;held_embryo=$held;
    training_pid=$fit.pid;training_start_utc=$ownedProcess.StartTime.ToUniversalTime().ToString('o');
    started_utc=[DateTime]::UtcNow.ToString('o');evaluator_sha256=$evaluatorHash;fit_contract_sha256=$expected.contract_sha256;
    gpu_used=$false;training_restart_allowed=$false;submission_allowed=$false;stages=@()}
function Save-Receipt {
    $receipt.updated_utc = [DateTime]::UtcNow.ToString('o')
    $temporary = "$queuePath.tmp"
    [IO.File]::WriteAllText($temporary,($receipt | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $queuePath -Force
}
try {
    Save-Receipt
    $waitStart = [DateTime]::UtcNow
    while (-not $ownedProcess.WaitForExit(30000)) {
        if (([DateTime]::UtcNow - $waitStart).TotalHours -gt 3) {
            throw 'Successor wait exceeded three hours; training is untouched'
        }
    }
    $fit = Get-Content -LiteralPath $fitReportPath -Raw | ConvertFrom-Json
    if ($fit.status -ne 'source_event_training_complete' -or -not $fit.model_fitted) {
        throw "Training ended as $($fit.status); inspect/resume separately, no evaluation launched"
    }
    Assert-Inputs
    foreach ($stage in @('smoke','full')) {
        $receipt.status = "running_$stage";Save-Receipt
        $arguments = @($evaluator,'--fit-embryo',$FitEmbryo)
        if ($stage -eq 'smoke') { $arguments += '--smoke' }
        & $pythonExe @arguments
        if ($LASTEXITCODE -ne 0) { throw "$stage evaluator failed with exit $LASTEXITCODE" }
        $suffix = if ($stage -eq 'smoke') {'-smoke'} else {''}
        $outputPath = Join-Path $reportRoot "trajectory-event-fork16-held-$held-v1$suffix.json"
        $output = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
        $expectedStatus = if ($stage -eq 'smoke') {'complete_movie_inference_smoke_passed'} else {'held_embryo_complete_movie_scoring_finished'}
        if ($output.status -ne $expectedStatus) { throw 'Evaluator did not produce the required terminal state' }
        $receipt.stages += [ordered]@{stage=$stage;status=$output.status;sha256=(Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()}
        Save-Receipt
    }
    $receipt.status='held_evaluation_completed_no_automatic_promotion';Save-Receipt
} catch {
    $receipt.status='stopped_requires_inspection';$receipt.error=$_.Exception.Message;Save-Receipt
    throw
} finally {
    $lockStream.Dispose()
    # Preserve the lock marker as an ownership receipt; do not auto-relaunch.
}
