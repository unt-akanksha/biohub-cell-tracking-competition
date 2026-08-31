param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [int]$PollSeconds = 60,
    [int]$MaximumPolls = 1800,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$controller = Join-Path $PSScriptRoot "wait-evaluate-graph-context-division-development.ps1"
$arguments = @{
    RepositoryRoot = $RepositoryRoot
    PollSeconds = $PollSeconds
    MaximumPolls = $MaximumPolls
    HarvestRootRelative = ".biohub/cache/antelume-graph-context-recovery-v1"
    StateRootRelative = ".biohub/cache/graph-context-division-recovery-development-v1"
    ArchiveFileName = "graph-context-recovery-results.tar.gz"
    ControllerRunId = "graph-context-division-recovery-development-controller-v1"
}
if ($ValidateOnly) { $arguments["ValidateOnly"] = $true }
& $controller @arguments
exit $LASTEXITCODE
