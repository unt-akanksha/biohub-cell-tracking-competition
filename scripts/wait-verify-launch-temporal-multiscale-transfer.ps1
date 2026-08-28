param(
    [double]$MinimumQuotaHours = 11.0,
    [double]$MaximumWaitHours = 504.0
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$automationDir = Join-Path $projectRoot '.biohub/automation'
$pretrainLaunchTerminal = Join-Path $automationDir 'zebrahub-multiscale-pretrain-launch.json'
$logPath = Join-Path $automationDir 'temporal-multiscale-transfer-launch.log'
$terminalPath = Join-Path $automationDir 'temporal-multiscale-transfer-launch.json'
$verificationOutput = Join-Path $automationDir 'zebrahub-multiscale-pretrain-verification.json'
$verificationError = Join-Path $automationDir 'zebrahub-multiscale-pretrain-verification.stderr.log'
$pretrainRef = 'indarkarhana/biohub-zebrahub-multiscale-pretrain-v1'
$transferRef = 'indarkarhana/biohub-temporal-multiscale-transfer-v4'
$runtimeRef = 'indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4'
$pretrainDownloadRoot = Join-Path $projectRoot '.biohub/cache/kernel-outputs/zebrahub-multiscale-pretrain-v1-autochain'
$transferDir = Join-Path $projectRoot 'kaggle/biohub-temporal-multiscale-transfer-v4'
$transferMetadata = Join-Path $transferDir 'kernel-metadata.json'
$transferNotebook = Join-Path $transferDir 'biohub-temporal-multiscale-transfer-v4.ipynb'
$verifier = Join-Path $projectRoot 'research/temporal_contrastive/verify_multiscale_pretraining_output.py'
$evaluationPython = Join-Path $projectRoot '.biohub/evaluation-venv/Scripts/python.exe'
$kernelStateScript = Join-Path $projectRoot 'scripts/get-kaggle-kernel-state.py'
$expectedVerifierSha256 = '150176d062679c0094a169be04b16638a89a45631dd9d70cf8209fd82a1dbf30'
$expectedMetadataSha256 = 'ec785539b7006808266cc2e81ee2017da5ab77dd1974d93dabb63195dbe36e83'
$expectedNotebookSha256 = 'fd749af69a332f06c012983dd8cb588e4363a7e3d3e9d07103ca6f544e14d71c'

New-Item -ItemType Directory -Force -Path $automationDir | Out-Null

function Write-ChainLog([string]$Message) {
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value (
        ([DateTimeOffset]::Now.ToString('o')) + ' ' + $Message
    )
}

function Write-ChainTerminal([string]$Status, [hashtable]$Evidence) {
    $payload = @{
        schema_version = 1
        run_id = 'temporal-multiscale-contextual-pair-fusion-v4'
        status = $Status
        pretrain_ref = $pretrainRef
        transfer_ref = $transferRef
        minimum_quota_hours = $MinimumQuotaHours
        quota_reserve_hours = 0.0
        expected_gpu_count = 2
        machine_shape = 'NvidiaTeslaT4'
        verifier_sha256 = $expectedVerifierSha256
        transfer_metadata_sha256 = $expectedMetadataSha256
        transfer_notebook_sha256 = $expectedNotebookSha256
        public_leaderboard_used_for_selection = $false
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

function Require-Hash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Label changed: $actual"
    }
}

if (Test-Path -LiteralPath $terminalPath) {
    Write-ChainLog "refused_existing_terminal path=$terminalPath"
    exit 2
}

try {
    Require-Hash $verifier $expectedVerifierSha256 'Multiscale pretraining verifier'
    Require-Hash $transferMetadata $expectedMetadataSha256 'Multiscale transfer metadata'
    Require-Hash $transferNotebook $expectedNotebookSha256 'Multiscale transfer notebook'
    foreach ($required in @($evaluationPython, $kernelStateScript)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required transfer-chain input is missing: $required"
        }
    }
    $metadata = Get-Content -Raw -LiteralPath $transferMetadata | ConvertFrom-Json
    if (
        $metadata.id -ne $transferRef -or
        $metadata.is_private -ne $true -or
        $metadata.enable_gpu -ne $true -or
        $metadata.enable_tpu -ne $false -or
        $metadata.enable_internet -ne $false -or
        $metadata.machine_shape -ne 'NvidiaTeslaT4' -or
        $metadata.dataset_sources -notcontains $runtimeRef -or
        $metadata.kernel_sources -notcontains $pretrainRef -or
        $metadata.kernel_sources -notcontains 'indarkarhana/biohub-zsns001-contextual-gate-v1' -or
        $metadata.competition_sources -notcontains 'biohub-cell-tracking-during-development'
    ) {
        throw 'Multiscale transfer metadata contract is invalid'
    }

    $deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
    while (-not (Test-Path -LiteralPath $pretrainLaunchTerminal)) {
        if ([DateTimeOffset]::UtcNow -ge $deadline) {
            throw 'Timed out waiting for multiscale pretraining launch evidence'
        }
        Start-Sleep -Seconds 60
    }
    $pretrainLaunch = Get-Content -Raw -LiteralPath $pretrainLaunchTerminal | ConvertFrom-Json
    $pretrainVersion = [int]$pretrainLaunch.kernel_version
    if (
        $pretrainLaunch.status -ne 'launched' -or
        $pretrainLaunch.kernel_ref -ne $pretrainRef -or
        $pretrainLaunch.expected_gpu_count -ne 2 -or
        $pretrainVersion -lt 1 -or
        $pretrainLaunch.quota_reserve_hours -ne 0.0 -or
        $pretrainLaunch.public_leaderboard_used_for_selection -ne $false -or
        $pretrainLaunch.submission_created -ne $false
    ) {
        throw 'Multiscale pretraining launch evidence is invalid'
    }

    $externalPoll = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($externalPoll -gt 0) {
            for ($wait = 0; $wait -lt 5; $wait++) {
                Start-Sleep -Seconds 60
            }
        }
        $externalPoll += 1
        $statusOutput = (& kaggle kernels status $pretrainRef 2>&1) -join "`n"
        Write-ChainLog "pretrain_status poll=$externalPoll version=$pretrainVersion output=$statusOutput"
        if ($statusOutput -match 'COMPLETE') {
            break
        }
        if ($statusOutput -match 'ERROR|CANCEL') {
            throw "Multiscale pretraining did not complete: $statusOutput"
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw 'Timed out waiting for multiscale pretraining completion'
    }
    $pretrainStateText = (& python $kernelStateScript --kernel-slug $pretrainRef 2>&1) -join "`n"
    $pretrainState = $pretrainStateText | ConvertFrom-Json
    if (
        $pretrainState.present -ne $true -or
        [int]$pretrainState.current_version_number -ne $pretrainVersion -or
        $pretrainState.enable_gpu -ne $true -or
        $pretrainState.enable_tpu -ne $false -or
        $pretrainState.enable_internet -ne $false
    ) {
        throw "Completed multiscale pretraining remote state is invalid: $pretrainStateText"
    }

    if (Test-Path -LiteralPath $pretrainDownloadRoot) {
        if (-not (Get-ChildItem -LiteralPath $pretrainDownloadRoot -Recurse -Filter launcher_terminal.json -ErrorAction SilentlyContinue)) {
            throw "Pretraining download target exists without launcher evidence: $pretrainDownloadRoot"
        }
    } else {
        New-Item -ItemType Directory -Path $pretrainDownloadRoot | Out-Null
        $downloadOutput = (& kaggle kernels output $pretrainRef -p $pretrainDownloadRoot --force 2>&1) -join "`n"
        if ($LASTEXITCODE -ne 0) {
            throw "Multiscale pretraining output download failed: $downloadOutput"
        }
        Write-ChainLog "pretrain_download output=$downloadOutput"
    }
    if (Test-Path -LiteralPath $verificationOutput) {
        throw "Refusing to overwrite multiscale verification evidence: $verificationOutput"
    }
    & $evaluationPython $verifier --root $pretrainDownloadRoot 1> $verificationOutput 2> $verificationError
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

    $runtimeText = (& kaggle datasets status $runtimeRef --format json 2>&1) -join "`n"
    $runtimeStatus = $runtimeText | ConvertFrom-Json
    if ($runtimeStatus.status -ne 'ready' -or [int]$runtimeStatus.current_version_number -ne 3) {
        throw "Multiscale runtime version changed: $runtimeText"
    }
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $quotaText = (& kaggle quota --format json 2>&1) -join "`n"
        $resources = $quotaText | ConvertFrom-Json
        $gpu = $resources | Where-Object resource -eq 'GPU'
        $remaining = [double](($gpu.remaining -replace 'h', ''))
        if ($remaining -lt $MinimumQuotaHours) {
            Write-ChainLog "quota_not_ready remaining=$remaining"
            Start-Sleep -Seconds 60
            continue
        }
        $pushOutput = (& kaggle kernels push -p $transferDir 2>&1) -join "`n"
        Write-ChainLog "transfer_push remaining=$remaining output=$pushOutput"
        if ($pushOutput -match 'successfully pushed') {
            Start-Sleep -Seconds 5
            $transferStatus = (& kaggle kernels status $transferRef 2>&1) -join "`n"
            if ($transferStatus -notmatch 'QUEUED|RUNNING|COMPLETE') {
                throw "Multiscale transfer push succeeded but status is invalid: $transferStatus"
            }
            $transferStateText = (& python $kernelStateScript --kernel-slug $transferRef 2>&1) -join "`n"
            $transferState = $transferStateText | ConvertFrom-Json
            $transferVersion = [int]$transferState.current_version_number
            if (
                $transferState.kernel_slug -ne $transferRef -or
                $transferState.present -ne $true -or
                $transferVersion -lt 1 -or
                $transferState.is_private -ne $true -or
                $transferState.enable_gpu -ne $true -or
                $transferState.enable_tpu -ne $false -or
                $transferState.enable_internet -ne $false -or
                $transferState.dataset_sources -notcontains $runtimeRef -or
                $transferState.kernel_sources -notcontains $pretrainRef -or
                $transferState.competition_sources -notcontains 'biohub-cell-tracking-during-development'
            ) {
                throw "Remote multiscale transfer state is invalid: $transferStateText"
            }
            Write-ChainTerminal 'launched' @{
                pretrain_kernel_version = $pretrainVersion
                pretraining_terminal_sha256 = $verified.pretraining_terminal_sha256
                pretraining_launcher_sha256 = $verified.launcher_terminal_sha256
                verification_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $verificationOutput).Hash.ToLowerInvariant()
                quota_before_hours = $remaining
                kernel_version = $transferVersion
                kaggle_status = $transferStatus
                push_output = $pushOutput
            }
            Write-ChainLog "transfer_launched version=$transferVersion status=$transferStatus"
            exit 0
        }
        if ($pushOutput -notmatch 'Maximum batch GPU session count|quota|accelerator') {
            throw "Multiscale transfer push failed: $pushOutput"
        }
        Start-Sleep -Seconds 60
    }
    throw 'Timed out waiting to launch multiscale transfer'
} catch {
    Write-ChainTerminal 'failed' @{
        error = $_.Exception.Message
        competition_submission_performed = $false
    }
    Write-ChainLog "failed error=$($_.Exception.Message)"
    exit 1
}
