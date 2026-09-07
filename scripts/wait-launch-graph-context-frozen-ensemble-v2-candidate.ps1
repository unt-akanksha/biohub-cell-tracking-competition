param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 240.0,
    [int]$PollSeconds = 300,
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
    DevelopmentTerminalRelative = ".biohub/cache/graph-context-frozen-ensemble-development-v2/development-terminal.json"
    CandidateDirectoryRelative = "kaggle/biohub-ema-graph-context-consensus-v1"
    CandidateBuilderRelative = "scripts/build-graph-context-consensus-submission-candidate.py"
    CandidateNotebookName = "biohub-ema-graph-context-consensus-v1.ipynb"
    VerifierRelative = "scripts/verify-graph-context-consensus-submission-candidate.py"
    SubmitterRelative = "scripts/submit-graph-context-consensus-candidate.py"
    RuntimeSearchTerm = "biohub-graph-context-consensus-division-v1"
    DatasetVersionMessage = "Frozen-policy graph-context division ensemble v2"
    StatePrefix = "graph-context-frozen-ensemble-v2-candidate"
    LaunchRunId = "graph-context-frozen-ensemble-v2-candidate-launch"
    PromotionRunId = "graph-context-frozen-ensemble-v2-candidate-controller"
    FamilyLabel = "Graph-context frozen ensemble v2"
}
if ($ValidateOnly) { $arguments["ValidateOnly"] = $true }
& $launcher @arguments
exit $LASTEXITCODE
