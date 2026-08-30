param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 72.0,
    [int]$PollSeconds = 60,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "wait-launch-relational-consensus-candidate.ps1"
$arguments = @{
    RepositoryRoot = $RepositoryRoot
    MaximumWaitHours = $MaximumWaitHours
    PollSeconds = $PollSeconds
    RuntimeDatasetRef = "indarkarhana/biohub-graph-context-consensus-division-v1"
    CandidateKernelRef = "indarkarhana/biohub-ema-graph-context-consensus-v1"
    DevelopmentTerminalRelative = ".biohub/cache/graph-context-division-development-v1/development-terminal.json"
    CandidateDirectoryRelative = "kaggle/biohub-ema-graph-context-consensus-v1"
    CandidateBuilderRelative = "scripts/build-graph-context-consensus-submission-candidate.py"
    CandidateNotebookName = "biohub-ema-graph-context-consensus-v1.ipynb"
    VerifierRelative = "scripts/verify-graph-context-consensus-submission-candidate.py"
    SubmitterRelative = "scripts/submit-graph-context-consensus-candidate.py"
    RuntimeSearchTerm = "biohub-graph-context-consensus-division-v1"
    DatasetVersionMessage = "Development-admitted graph-context division consensus v1"
    StatePrefix = "graph-context-consensus-candidate"
    LaunchRunId = "graph-context-consensus-candidate-launch-v1"
    PromotionRunId = "graph-context-consensus-candidate-controller-v1"
    FamilyLabel = "Graph-context"
}
if ($ValidateOnly) { $arguments["ValidateOnly"] = $true }
& $launcher @arguments
exit $LASTEXITCODE
