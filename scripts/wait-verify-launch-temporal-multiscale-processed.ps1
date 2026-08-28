param(
    [double]$MinimumQuotaHours = 6.0,
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$base = Join-Path $PSScriptRoot 'wait-verify-launch-temporal-contextual-processed.ps1'
$expectedBaseSha256 = 'f34ac3baa41d380134f68d387406abad149682c5a8b14ea103c6ab1c322f8409'
$observedBaseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $base).Hash.ToLowerInvariant()
if ($observedBaseSha256 -ne $expectedBaseSha256) {
    throw "Base processed watcher changed: $observedBaseSha256"
}

$source = Get-Content -Raw -LiteralPath $base
$quotedRoot = $projectRoot.Replace("'", "''")
$replacements = [ordered]@{
    '$projectRoot = Split-Path -Parent $PSScriptRoot' = "`$projectRoot = '$quotedRoot'"
    'temporal-contextual-calibration-launch.json' = 'temporal-multiscale-calibration-launch.json'
    'temporal-contextual-processed-launch.log' = 'temporal-multiscale-processed-launch.log'
    'temporal-contextual-processed-launch.json' = 'temporal-multiscale-processed-launch.json'
    'indarkarhana/biohub-temporal-contextual-calibration-v3' = 'indarkarhana/biohub-multiscale-calibration-v4'
    'indarkarhana/biohub-temporal-contextual-processed-acceptance-v3' = 'indarkarhana/biohub-multiscale-processed-v4'
    'kaggle/biohub-temporal-contextual-processed-acceptance-v3' = 'kaggle/biohub-multiscale-processed-v4'
    'biohub-temporal-contextual-processed-acceptance-v3.ipynb' = 'biohub-multiscale-processed-v4.ipynb'
    '.biohub/cache/kernel-outputs/temporal-contextual-calibration-v3-autochain' = '.biohub/cache/kernel-outputs/temporal-multiscale-calibration-v4-autochain'
    'temporal_contextual_calibration_v3' = 'temporal_multiscale_contextual_calibration_v4'
    '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain/temporal_contextual_transfer_v3' = '.biohub/cache/kernel-outputs/temporal-multiscale-transfer-v4-autochain/temporal_multiscale_contextual_transfer_v4'
    '624a14a202cde15a1bfc4cf942241df1f80801094e60e8886182c8dffc826125' = '0dace5ddc203a028459b030e6be98a0f08f69d77ec15332e0a782daee2a44355'
    '08c0316da23940e21490d56909d3bb88c690b9103dfb0d2cf57c40de96e8665b' = 'dda2a9b160b2ccbf61e47cb5985da37a44a77c4a99e5764ee272a9c2b6480564'
    'indarkarhana/biohub-temporal-contextual-transfer-runtime-v1' = 'indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4'
    'indarkarhana/biohub-temporal-contextual-transfer-v3' = 'indarkarhana/biohub-temporal-multiscale-transfer-v4'
    'temporal-contextual-pair-fusion-processed-acceptance-v3' = 'temporal-multiscale-contextual-pair-fusion-processed-acceptance-v4'
    'temporal-contextual-pair-fusion-blend-v3' = 'temporal-multiscale-contextual-pair-fusion-blend-v4'
    'temporal_contextual_pair_fusion_v3' = 'temporal_multiscale_contextual_pair_fusion_v4'
}
foreach ($entry in $replacements.GetEnumerator()) {
    if (-not $source.Contains($entry.Key)) {
        throw "Base processed watcher lost replacement token: $($entry.Key)"
    }
    $source = $source.Replace($entry.Key, $entry.Value)
}
if (
    $source.Contains('temporal-contextual-pair-fusion-processed-acceptance-v3') -or
    $source.Contains('biohub-temporal-contextual-processed-acceptance-v3') -or
    $source.Contains('biohub-temporal-contextual-calibration-v3')
) {
    throw 'Transformed multiscale processed watcher retains a v3 execution reference'
}
$script = [ScriptBlock]::Create($source)
if ($ValidateOnly) {
    [pscustomobject]@{
        status = 'validated'
        stage = 'multiscale_processed'
        transformed_characters = $source.Length
    } | ConvertTo-Json
    exit 0
}

& $script -MinimumQuotaHours $MinimumQuotaHours -MaximumWaitHours $MaximumWaitHours
exit $LASTEXITCODE
