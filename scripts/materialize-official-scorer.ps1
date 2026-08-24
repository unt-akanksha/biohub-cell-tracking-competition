[CmdletBinding()]
param(
    [Parameter()]
    [string]$ProjectRoot = "",

    [Parameter()]
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$OrganizerRepository = "https://github.com/royerlab/kaggle-cell-tracking-competition.git"
$OrganizerCommit = "075fc5f5a52d11077f9dc2b074644618f26939e2"
$TracksDataRepository = "https://github.com/royerlab/tracksdata.git"
$TracksDataCommit = "39dccf3a243e44274759468cb31b2ad9e7fc1d09"

$ResolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$VendorRoot = Join-Path $ResolvedRoot ".biohub/vendor"
$OrganizerCheckout = Join-Path $VendorRoot "kaggle-cell-tracking-competition"
$TracksDataCheckout = Join-Path $VendorRoot "tracksdata"
$EnvironmentPath = Join-Path $ResolvedRoot ".biohub/evaluation-venv"
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts/python.exe"
$RequirementsLock = Join-Path $ResolvedRoot "requirements/evaluation-lock.txt"

New-Item -ItemType Directory -Force -Path $VendorRoot | Out-Null

function Invoke-Git {
    param([string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git failed with exit code $LASTEXITCODE"
    }
}

function ConvertTo-NormalizedRepository {
    param([string]$Value)
    return ($Value.TrimEnd("/") -replace '\.git$', '').ToLowerInvariant()
}

function Assert-Or-CreateCheckout {
    param(
        [string]$Repository,
        [string]$Commit,
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Destination) {
        if (-not (Test-Path -LiteralPath (Join-Path $Destination ".git"))) {
            throw "refusing non-git checkout at $Destination"
        }
        $Remote = (& git -C $Destination remote get-url origin).Trim()
        if ($LASTEXITCODE -ne 0 -or (ConvertTo-NormalizedRepository $Remote) -ne (ConvertTo-NormalizedRepository $Repository)) {
            throw "refusing checkout with unexpected origin at $Destination"
        }
        $Head = (& git -C $Destination rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or $Head -ne $Commit) {
            throw "refusing checkout at unexpected commit: $Destination"
        }
        $Dirty = & git -C $Destination status --porcelain --untracked-files=no
        if ($LASTEXITCODE -ne 0 -or $Dirty) {
            throw "refusing modified checkout at $Destination"
        }
        return
    }

    # Keep the small public source history so isolated regression tests can
    # compare the pre-patch commit without a second network checkout.
    Invoke-Git -Arguments @("clone", "--no-checkout", $Repository, $Destination)
    Invoke-Git -Arguments @("-C", $Destination, "checkout", "--detach", $Commit)
}

Assert-Or-CreateCheckout -Repository $OrganizerRepository -Commit $OrganizerCommit -Destination $OrganizerCheckout
Assert-Or-CreateCheckout -Repository $TracksDataRepository -Commit $TracksDataCommit -Destination $TracksDataCheckout

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    & uv venv $EnvironmentPath --python $PythonExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "failed to create scorer environment"
    }
}

& uv pip install --python $EnvironmentPython --requirement $RequirementsLock
if ($LASTEXITCODE -ne 0) {
    throw "failed to install the scorer environment lock"
}
& uv pip install --python $EnvironmentPython --no-deps --editable $TracksDataCheckout --editable $OrganizerCheckout --editable $ResolvedRoot
if ($LASTEXITCODE -ne 0) {
    throw "failed to install verified local source checkouts"
}

& $EnvironmentPython -m biohub_tracker --root $ResolvedRoot scorer verify --lock (Join-Path $ResolvedRoot "config/official-scorer.lock.json") --checkout $OrganizerCheckout --tracksdata-checkout $TracksDataCheckout
if ($LASTEXITCODE -ne 0) {
    throw "official scorer verification failed"
}
