param(
    [string]$RepositoryRoot = "C:/Users/IndarKumar/Documents/Comp/Biohub",
    [double]$MaximumWaitHours = 96.0,
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
    DevelopmentTerminalRelative = ".biohub/cache/graph-context-division-recovery-development-v1/development-terminal.json"
    CandidateDirectoryRelative = "kaggle/biohub-ema-graph-context-consensus-v1"
    CandidateBuilderRelative = "scripts/build-graph-context-consensus-submission-candidate.py"
    CandidateNotebookName = "biohub-ema-graph-context-consensus-v1.ipynb"
    VerifierRelative = "scripts/verify-graph-context-consensus-submission-candidate.py"
    SubmitterRelative = "scripts/submit-graph-context-consensus-candidate.py"
    RuntimeSearchTerm = "biohub-graph-context-consensus-division-v1"
    DatasetVersionMessage = "Recovered five-member plus completed graph-context consensus v1"
    StatePrefix = "graph-context-recovery-consensus-candidate"
    LaunchRunId = "graph-context-recovery-consensus-candidate-launch-v1"
    PromotionRunId = "graph-context-recovery-consensus-candidate-controller-v1"
    FamilyLabel = "Recovered graph-context"
}
if ($ValidateOnly) { $arguments["ValidateOnly"] = $true }
& $launcher @arguments
exit $LASTEXITCODE
