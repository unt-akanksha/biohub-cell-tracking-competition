param(
    [double]$MaximumWaitHours = 48.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$kernelDir = Join-Path $projectRoot 'kaggle/biohub-zebrahub-multiscale-pretrain-v1'
$metadataPath = Join-Path $kernelDir 'kernel-metadata.json'
$notebookPath = Join-Path $kernelDir 'biohub-zebrahub-multiscale-pretrain-v1.ipynb'
$verifierPath = Join-Path $projectRoot 'research/temporal_contrastive/verify_multiscale_pretraining_output.py'
$kernelStateScript = Join-Path $projectRoot 'scripts/get-kaggle-kernel-state.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$kernelRef = 'indarkarhana/biohub-zebrahub-multiscale-pretrain-v1'
$runtimeRef = 'indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4'
$shardsRef = 'indarkarhana/biohub-zebrahub-contextual-shards-v1'
$downloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/zebrahub-multiscale-pretrain-v1-recovery-20260829'
$logPath = Join-Path $automationDir 'zebrahub-multiscale-pretrain-recovery.log'
$terminalPath = Join-Path $automationDir 'zebrahub-multiscale-pretrain-recovery.json'
$verificationOutput = Join-Path $automationDir 'zebrahub-multiscale-pretrain-recovery-verification.json'
$verificationError = Join-Path $automationDir 'zebrahub-multiscale-pretrain-recovery-verification.stderr.log'
$expectedVerifierSha256 = '150176d062679c0094a169be04b16638a89a45631dd9d70cf8209fd82a1dbf30'
$expectedMetadataSha256 = 'd7edfb27e13d11c7cc3d9e85d3683417193e6acafc4ed3c0eb13362ef39504c6'
$expectedNotebookSha256 = 'c15f7e1d9080067bec78e7567b1aba8c95132b135854cedd56951954083ed4ec'

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label changed: $actual"
    }
}

function Write-RecoveryLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-RecoveryTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'zebrahub-multiscale-contextual-pretrain-v1-recovery'
        status = $Status
        kernel_ref = $kernelRef
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        metadata_sha256 = $expectedMetadataSha256
        notebook_sha256 = $expectedNotebookSha256
        verifier_sha256 = $expectedVerifierSha256
        competition_data_read = $false
        public_leaderboard_used_for_selection = $false
        downstream_kernel_launched = $false
        submission_created = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString('o')
    }
    foreach ($key in $Evidence.Keys) {
        $payload[$key] = $Evidence[$key]
    }
    $temporary = $terminalPath + '.tmp'
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $terminalPath -Force
}

function Assert-LocalContract {
    Require-Hash $verifierPath $expectedVerifierSha256 'Multiscale pretraining verifier'
    Require-Hash $metadataPath $expectedMetadataSha256 'Multiscale pretraining metadata'
    Require-Hash $notebookPath $expectedNotebookSha256 'Multiscale pretraining notebook'
    foreach ($required in @($evaluationPython, $kernelStateScript)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required recovery input is missing: $required"
        }
    }
    $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
    if (
        $metadata.id -ne $kernelRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.dataset_sources -notcontains $shardsRef -or
        $metadata.kernel_sources -notcontains 'indarkarhana/biohub-zebrahub-contextual-pretrain-v1' -or
        $metadata.kernel_sources -notcontains 'indarkarhana/biohub-zsns001-contextual-gate-v1' -or
        @($metadata.competition_sources).Count -ne 0
    ) {
        throw 'Multiscale pretraining metadata contract is invalid'
    }
}

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null
Assert-LocalContract
if ($ValidateOnly) {
    [ordered]@{
        status = 'valid'
        kernel_ref = $kernelRef
        download_root = $downloadRoot
        downstream_kernel_launched = $false
        submission_created = $false
    } | ConvertTo-Json
    exit 0
}

if (Test-Path -LiteralPath $terminalPath) {
    Write-RecoveryLog "refused_existing_terminal path=$terminalPath"
    exit 2
}

try {
    $stateText = (& python $kernelStateScript --kernel-slug $kernelRef 2>&1) -join "`n"
    $initialState = $stateText | ConvertFrom-Json
    $kernelVersion = [int]$initialState.current_version_number
    if (
        $initialState.present -ne $true -or
        $kernelVersion -lt 1 -or
        $initialState.is_private -ne $true -or
        $initialState.enable_gpu -ne $true -or
        $initialState.enable_tpu -ne $false -or
        $initialState.enable_internet -ne $false -or
        $initialState.dataset_sources -notcontains $runtimeRef -or
        $initialState.dataset_sources -notcontains $shardsRef -or
        @($initialState.competition_sources).Count -ne 0
    ) {
        throw "Initial remote pretraining state is invalid: $stateText"
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    $poll = 0
    $completed = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($poll -gt 0) {
            for ($minute = 0; $minute -lt 5; $minute++) {
                Start-Sleep -Seconds 60
            }
        }
        $poll += 1
        $statusOutput = (& kaggle kernels status $kernelRef 2>&1) -join "`n"
        Write-RecoveryLog "status poll=$poll version=$kernelVersion output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            $completed = $true
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Multiscale pretraining did not complete: $statusOutput"
        }
    }
    if (-not $completed) {
        throw 'Timed out waiting for multiscale pretraining completion'
    }

    $completedStateText = (& python $kernelStateScript --kernel-slug $kernelRef 2>&1) -join "`n"
    $completedState = $completedStateText | ConvertFrom-Json
    if (
        $completedState.present -ne $true -or
        [int]$completedState.current_version_number -ne $kernelVersion -or
        $completedState.enable_gpu -ne $true -or
        $completedState.enable_tpu -ne $false -or
        $completedState.enable_internet -ne $false -or
        $completedState.dataset_sources -notcontains $runtimeRef -or
        $completedState.dataset_sources -notcontains $shardsRef -or
        @($completedState.competition_sources).Count -ne 0
    ) {
        throw "Completed remote pretraining state is invalid: $completedStateText"
    }

    if (Test-Path -LiteralPath $downloadRoot) {
        throw "Refusing to reuse recovery download target: $downloadRoot"
    }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $downloadOutput = (& kaggle kernels output $kernelRef -p $downloadRoot --force 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "Multiscale pretraining output download failed: $downloadOutput"
    }
    Write-RecoveryLog "download output=$downloadOutput"

    if (Test-Path -LiteralPath $verificationOutput) {
        throw "Refusing to overwrite recovery verification evidence: $verificationOutput"
    }
    & $evaluationPython $verifierPath --root $downloadRoot 1> $verificationOutput 2> $verificationError
    if ($LASTEXITCODE -ne 0) {
        throw "Multiscale pretraining verification failed; see $verificationError"
    }
    $verified = Get-Content -Raw -LiteralPath $verificationOutput | ConvertFrom-Json
    if (
        $verified.status -ne 'verified' -or
        $verified.run_id -ne 'zebrahub-multiscale-contextual-pretrain-v1' -or
        $verified.appearance_family -ne 'temporal_multiscale_contextual_pair_fusion_v4' -or
        $verified.gpu_count -ne 2 -or
        $verified.strict_checkpoint_loaded -ne $true -or
        $verified.folds.PSObject.Properties.Count -ne 2 -or
        $verified.competition_data_read -ne $false -or
        $verified.public_leaderboard_used_for_selection -ne $false -or
        $verified.submission_created -ne $false
    ) {
        throw 'Multiscale pretraining verification evidence is invalid'
    }

    Write-RecoveryTerminal 'verified' @{
        kernel_version = $kernelVersion
        poll_count = $poll
        download_root = $downloadRoot
        pretraining_terminal_sha256 = $verified.pretraining_terminal_sha256
        launcher_terminal_sha256 = $verified.launcher_terminal_sha256
        verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verificationOutput).Hash.ToLowerInvariant()
    }
    Write-RecoveryLog "verified version=$kernelVersion polls=$poll"
    exit 0
} catch {
    Write-RecoveryTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-RecoveryLog "failed error=$($_.Exception.Message)"
    exit 1
}
