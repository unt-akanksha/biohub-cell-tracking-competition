param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [string]$KernelRef = "indarkarhana/biohub-real-localization-expanded-replay-cache-v2",
    [int]$ExpectedKernelVersion = 2,
    [int]$PollSeconds = 120,
    [double]$MaximumWaitHours = 8.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ($ExpectedKernelVersion -lt 1 -or $PollSeconds -lt 60 -or $MaximumWaitHours -le 0) {
    throw "Invalid expanded replay harvest bounds"
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $RepositoryRoot
$cacheRoot = Join-Path $RepositoryRoot ".biohub/cache/competition-real-localization-expanded-replay-v2"
$downloadRoot = Join-Path $cacheRoot "kernel-v$ExpectedKernelVersion"
$terminalPath = Join-Path $cacheRoot "harvest-v$ExpectedKernelVersion-terminal.json"
$logPath = Join-Path $cacheRoot "harvest-v$ExpectedKernelVersion.log"
$verifier = Join-Path $RepositoryRoot "scripts/verify-real-localization-expanded-replay-archive.py"
$deadline = [DateTimeOffset]::UtcNow.AddHours($MaximumWaitHours)
$versionedRef = "$KernelRef/$ExpectedKernelVersion"

if (-not (Test-Path -LiteralPath $verifier -PathType Leaf)) { throw "Expanded replay verifier is missing" }
& python -m py_compile $verifier
if ($LASTEXITCODE -ne 0) { throw "Expanded replay verifier does not compile" }
if ($ValidateOnly) {
    @{status="validated"; kernel_ref=$versionedRef; submission_performed=$false} | ConvertTo-Json
    exit 0
}
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
if (Test-Path -LiteralPath $terminalPath) { throw "Expanded replay harvest terminal already exists" }

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $logPath -Value "$([DateTimeOffset]::UtcNow.ToString('o')) $Message" -Encoding utf8
}

function Native([scriptblock]$Command) {
    $output = (& $Command 2>&1) -join "`n"
    return @{exit_code=$LASTEXITCODE; output=$output}
}

try {
    $complete = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        $result = Native { & kaggle kernels status $KernelRef }
        Write-Log "status exit=$($result.exit_code) output=$($result.output)"
        if ($result.exit_code -eq 0 -and $result.output -match "(?i)complete") {
            $complete = $true
            break
        }
        if ($result.output -match "(?i)(error|failed|cancel)") {
            throw "Expanded replay kernel failed: $($result.output)"
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $complete) { throw "Timed out waiting for expanded replay kernel" }
    if (Test-Path -LiteralPath $downloadRoot) { throw "Expanded replay download root already exists" }
    New-Item -ItemType Directory -Path $downloadRoot | Out-Null
    $pattern = '^(biohub-real-localization-expanded-shards-v2\.tar(\.sha256)?|launcher_terminal\.json)$'
    $result = Native {
        & kaggle kernels output $versionedRef -p $downloadRoot --force `
            --file-pattern $pattern --page-size 20
    }
    if ($result.exit_code -ne 0) { throw "Expanded replay download failed: $($result.output)" }
    Write-Log "downloaded output=$($result.output)"
    $archive = Join-Path $downloadRoot "biohub-real-localization-expanded-shards-v2.tar"
    $shaPath = "$archive.sha256"
    $kernelTerminal = Join-Path $downloadRoot "launcher_terminal.json"
    $verificationText = (& python $verifier --archive $archive --sha256-file $shaPath --terminal $kernelTerminal 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Expanded replay verification failed: $verificationText" }
    $verification = $verificationText | ConvertFrom-Json
    $payload = @{
        schema_version = 1
        run_id = "competition-real-localization-expanded-replay-harvest-v2"
        status = "verified"
        kernel_ref = $versionedRef
        archive = $archive
        verification = $verification
        competition_submission_performed = $false
        authorized_for_submission = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
}
catch {
    $payload = @{
        schema_version = 1
        run_id = "competition-real-localization-expanded-replay-harvest-v2"
        status = "failed"
        error = $_.Exception.Message
        competition_submission_performed = $false
        authorized_for_submission = $false
        recorded_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $temporary = "$terminalPath.partial"
    $payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $terminalPath
    throw
}
