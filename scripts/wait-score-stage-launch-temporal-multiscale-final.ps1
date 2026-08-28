param(
    [double]$MinimumQuotaHours = 12.0,
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$base = Join-Path $PSScriptRoot 'wait-score-stage-launch-temporal-contextual-final.ps1'
$expectedBaseSha256 = 'e4944ad149285730d2973e655ee940ec12a1fd3184aa66fc05d35fe28e71bfb5'
$observedBaseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $base).Hash.ToLowerInvariant()
if ($observedBaseSha256 -ne $expectedBaseSha256) {
    throw "Base final watcher changed: $observedBaseSha256"
}

$source = Get-Content -Raw -LiteralPath $base
$quotedRoot = $projectRoot.Replace("'", "''")
$replacements = [ordered]@{
    '$projectRoot = Split-Path -Parent $PSScriptRoot' = "`$projectRoot = '$quotedRoot'"
    'temporal-contextual-processed-launch.json' = 'temporal-multiscale-processed-launch.json'
    'temporal-contextual-final-launch.log' = 'temporal-multiscale-final-launch.log'
    'temporal-contextual-final-launch.json' = 'temporal-multiscale-final-launch.json'
    'indarkarhana/biohub-temporal-contextual-processed-acceptance-v3' = 'indarkarhana/biohub-multiscale-processed-v4'
    'indarkarhana/biohub-temporal-contextual-exact-acceptance-v3' = 'indarkarhana/biohub-multiscale-exact-acceptance-v4'
    'indarkarhana/biohub-temporal-contextual-final-runtime-v1' = 'indarkarhana/biohub-multiscale-contextual-final-v4'
    'indarkarhana/biohub-temporal-contextual-submission-candidate-v3' = 'indarkarhana/biohub-multiscale-submission-candidate-v4'
    '.biohub/cache/kernel-outputs/temporal-contextual-processed-v3-autochain' = '.biohub/cache/kernel-outputs/temporal-multiscale-processed-v4-autochain'
    '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain' = '.biohub/cache/kernel-outputs/temporal-multiscale-transfer-v4-autochain'
    'temporal-contextual-exact-acceptance.json' = 'temporal-multiscale-exact-acceptance.json'
    'temporal-contextual-exact-acceptance.stdout.log' = 'temporal-multiscale-exact-acceptance.stdout.log'
    'temporal-contextual-exact-acceptance.stderr.log' = 'temporal-multiscale-exact-acceptance.stderr.log'
    'temporal-contextual-exact-staging' = 'temporal-multiscale-exact-staging'
    'biohub-temporal-contextual-exact-acceptance-v3' = 'biohub-multiscale-exact-acceptance-v4'
    'temporal-contextual-final-preflight.json' = 'temporal-multiscale-final-preflight.json'
    'temporal-contextual-final-preflight.stdout.log' = 'temporal-multiscale-final-preflight.stdout.log'
    'temporal-contextual-final-preflight.stderr.log' = 'temporal-multiscale-final-preflight.stderr.log'
    'run-temporal-contextual-exact-acceptance.py' = 'run-temporal-multiscale-contextual-exact-acceptance.py'
    'stage-temporal-contextual-kaggle-artifacts.py' = 'stage-temporal-multiscale-kaggle-artifacts.py'
    'build-temporal-contextual-submission-candidate-preflight.py' = 'build-temporal-multiscale-submission-preflight.py'
    '.biohub/cache/dataset-redownloads/biohub-temporal-contextual-final-runtime-v1-version2' = '.biohub/cache/dataset-redownloads/biohub-multiscale-contextual-final-v4-version1'
    'kaggle/biohub-temporal-contextual-submission-candidate-v3' = 'kaggle/biohub-multiscale-submission-candidate-v4'
    'biohub-temporal-contextual-submission-candidate-v3.ipynb' = 'biohub-multiscale-submission-candidate-v4.ipynb'
    'dafd6fb978f1abc8ad56612113ec480a54cbd490bc12ff2ba88c4dae8133012f' = '4679591f6fb59edcdf91f5f38a8cfb04282549aff27268e89fa6f50ee6d7ad64'
    '1bd2319310c2e24c637b0d2b0d909fe11f756af7d2e5e450cafc614fee4f98fa' = '8677dcae28cec2cabdeee64872012c04db64d5ffeae320408b761b2f75c404e8'
    'b785a75f088c68ede25359a652510126a49a1758c9fb729ce956183576147b0f' = 'f4851d3bd4036430792a2d2b2af5ef900ed00032a3ee9b4a551791cfe1c04cc0'
    'ef010e45a6a10d1f00efee2d696a8c5a218c52b29e039673b32aa128ff248f43' = '484b6aa301e8bfd4769a5a6d1b5d9a848c2d096ea14a0d34537007218b819ad9'
    'f5a5e40827c0cd1b846dad0702faa7ae3697288f59c7ce24e8da7797d179db01' = '019c6c438bf9570e3765fd80f9904e0019c9e756b47aa94da734371e9d8384ee'
    'final_runtime_version = 2' = 'final_runtime_version = 1'
    'indarkarhana/biohub-temporal-contextual-transfer-v3' = 'indarkarhana/biohub-temporal-multiscale-transfer-v4'
    'temporal-contextual-pair-fusion-candidate-v3' = 'temporal-multiscale-contextual-pair-fusion-candidate-v4'
    'Exact clean processed acceptance for contextual v3' = 'Exact clean processed acceptance for multiscale v4'
}
foreach ($entry in $replacements.GetEnumerator()) {
    if (-not $source.Contains($entry.Key)) {
        throw "Base final watcher lost replacement token: $($entry.Key)"
    }
    $source = $source.Replace($entry.Key, $entry.Value)
}
if (
    $source.Contains('biohub-temporal-contextual-submission-candidate-v3') -or
    $source.Contains('biohub-temporal-contextual-exact-acceptance-v3')
) {
    throw 'Transformed multiscale final watcher retains a v3 execution reference'
}
$script = [ScriptBlock]::Create($source)
if ($ValidateOnly) {
    [pscustomobject]@{
        status = 'validated'
        stage = 'multiscale_final'
        transformed_characters = $source.Length
    } | ConvertTo-Json
    exit 0
}

& $script -MinimumQuotaHours $MinimumQuotaHours -MaximumWaitHours $MaximumWaitHours
exit $LASTEXITCODE
