param(
    [double]$MinimumQuotaHours = 6.0,
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$base = Join-Path $PSScriptRoot 'wait-verify-launch-temporal-contextual-calibration.ps1'
$expectedBaseSha256 = '5095862f4b3a758848e3e44173b7a3867a7852e2780466da156ebb4bc75750d9'
$observedBaseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $base).Hash.ToLowerInvariant()
if ($observedBaseSha256 -ne $expectedBaseSha256) {
    throw "Base calibration watcher changed: $observedBaseSha256"
}

$source = Get-Content -Raw -LiteralPath $base
$quotedRoot = $projectRoot.Replace("'", "''")
$replacements = [ordered]@{
    '$projectRoot = Split-Path -Parent $PSScriptRoot' = "`$projectRoot = '$quotedRoot'"
    'temporal-contextual-transfer-launch.json' = 'temporal-multiscale-transfer-launch.json'
    'temporal-contextual-calibration-launch.log' = 'temporal-multiscale-calibration-launch.log'
    'temporal-contextual-calibration-launch.json' = 'temporal-multiscale-calibration-launch.json'
    'indarkarhana/biohub-temporal-contextual-transfer-v3' = 'indarkarhana/biohub-temporal-multiscale-transfer-v4'
    'indarkarhana/biohub-temporal-contextual-calibration-v3' = 'indarkarhana/biohub-multiscale-calibration-v4'
    'kaggle/biohub-temporal-contextual-calibration-v3' = 'kaggle/biohub-multiscale-calibration-v4'
    'biohub-temporal-contextual-calibration-v3.ipynb' = 'biohub-multiscale-calibration-v4.ipynb'
    '.biohub/cache/dataset-redownloads/biohub-temporal-contextual-transfer-runtime-v1-version4' = '.biohub/cache/dataset-redownloads/biohub-temporal-multiscale-contextual-runtime-v4-version3'
    '.biohub/cache/kernel-outputs/temporal-contextual-transfer-v3-autochain' = '.biohub/cache/kernel-outputs/temporal-multiscale-transfer-v4-autochain'
    'temporal_contextual_transfer_v3' = 'temporal_multiscale_contextual_transfer_v4'
    'cbe5fe27639155746c95a98d91702d5fbe595172b058e0e9db330374ecfff25d' = 'ac1c32a70f3dcc699806d18bde487d5d9154ba6774f06c7a812773c0e0efb8b3'
    '46f0b73396f50d1e3bebb40ce2f811b20d791347dbf51b0b512e2e71e492b955' = '30f29003ddc6fda3bace2c564695fe1e2da74be5e67588736f6fb3795dcb40f3'
    '4596cd9fd7211c27f6b437268c8e719847f0e7cce91438cc86195e79aba497e1' = '6c45093936444fe0e4a84dbbca38028dbc4c9c52bfc534d2703eadaeb73444cf'
    'temporal-contextual-pair-fusion-blend-v3' = 'temporal-multiscale-contextual-pair-fusion-blend-v4'
    'temporal-contextual-pair-fusion-v3' = 'temporal-multiscale-contextual-pair-fusion-v4'
    'temporal_contextual_pair_fusion_v3' = 'temporal_multiscale_contextual_pair_fusion_v4'
}
foreach ($entry in $replacements.GetEnumerator()) {
    if (-not $source.Contains($entry.Key)) {
        throw "Base calibration watcher lost replacement token: $($entry.Key)"
    }
    $source = $source.Replace($entry.Key, $entry.Value)
}
if (
    $source.Contains('temporal-contextual-pair-fusion-v3') -or
    $source.Contains('biohub-temporal-contextual-calibration-v3') -or
    $source.Contains('biohub-temporal-contextual-transfer-v3')
) {
    throw 'Transformed multiscale calibration watcher retains a v3 execution reference'
}
$script = [ScriptBlock]::Create($source)
if ($ValidateOnly) {
    [pscustomobject]@{
        status = 'validated'
        stage = 'multiscale_calibration'
        transformed_characters = $source.Length
    } | ConvertTo-Json
    exit 0
}

& $script -MinimumQuotaHours $MinimumQuotaHours -MaximumWaitHours $MaximumWaitHours
exit $LASTEXITCODE
