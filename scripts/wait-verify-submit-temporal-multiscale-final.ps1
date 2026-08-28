param(
    [double]$MaximumWaitHours = 504.0,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$base = Join-Path $PSScriptRoot 'wait-verify-submit-temporal-contextual-final.ps1'
$expectedBaseSha256 = 'f0a3a84a4c94e95756134812cf5be1e34a47e3283e66efe66a5837a04714218b'
$observedBaseSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $base).Hash.ToLowerInvariant()
if ($observedBaseSha256 -ne $expectedBaseSha256) {
    throw "Base verified-submit watcher changed: $observedBaseSha256"
}

$source = Get-Content -Raw -LiteralPath $base
$quotedRoot = $projectRoot.Replace("'", "''")
$replacements = [ordered]@{
    '$projectRoot = Split-Path -Parent $PSScriptRoot' = "`$projectRoot = '$quotedRoot'"
    'temporal-contextual-final-launch.json' = 'temporal-multiscale-final-launch.json'
    'temporal-contextual-submission.log' = 'temporal-multiscale-submission.log'
    'temporal-contextual-submission.json' = 'temporal-multiscale-submission.json'
    'temporal-contextual-submission-receipt.json' = 'temporal-multiscale-submission-receipt.json'
    'indarkarhana/biohub-temporal-contextual-submission-candidate-v3' = 'indarkarhana/biohub-multiscale-submission-candidate-v4'
    'kaggle/biohub-temporal-contextual-submission-candidate-v3' = 'kaggle/biohub-multiscale-submission-candidate-v4'
    'biohub-temporal-contextual-submission-candidate-v3.ipynb' = 'biohub-multiscale-submission-candidate-v4.ipynb'
    'submit-temporal-contextual-kernel.py' = 'submit-temporal-multiscale-contextual-kernel.py'
    '.biohub/cache/kernel-outputs/temporal-contextual-submission-v3-autochain' = '.biohub/cache/kernel-outputs/temporal-multiscale-submission-v4-autochain'
    'temporal-contextual-submission.stdout.log' = 'temporal-multiscale-submission.stdout.log'
    'temporal-contextual-submission.stderr.log' = 'temporal-multiscale-submission.stderr.log'
    '7b43627a89b2d557c69665b3324da02ed8361a9f9a8b32f8df195725269ba6d6' = '0b771c93a3a66dc7e9431e6ef311e07acb830c9b07862e5995fe898a0fe8339c'
    'ef010e45a6a10d1f00efee2d696a8c5a218c52b29e039673b32aa128ff248f43' = '484b6aa301e8bfd4769a5a6d1b5d9a848c2d096ea14a0d34537007218b819ad9'
    'f5a5e40827c0cd1b846dad0702faa7ae3697288f59c7ce24e8da7797d179db01' = '019c6c438bf9570e3765fd80f9904e0019c9e756b47aa94da734371e9d8384ee'
    'temporal-contextual-pair-fusion-candidate-v3' = 'temporal-multiscale-contextual-pair-fusion-candidate-v4'
    'Contextual v3: clean reciprocal exact-gated non-replica candidate' = 'Multiscale v4: clean reciprocal exact-gated non-replica candidate'
}
foreach ($entry in $replacements.GetEnumerator()) {
    if (-not $source.Contains($entry.Key)) {
        throw "Base submit watcher lost replacement token: $($entry.Key)"
    }
    $source = $source.Replace($entry.Key, $entry.Value)
}
if ($source.Contains('biohub-temporal-contextual-submission-candidate-v3')) {
    throw 'Transformed multiscale submit watcher retains a v3 execution reference'
}
$script = [ScriptBlock]::Create($source)
if ($ValidateOnly) {
    [pscustomobject]@{
        status = 'validated'
        stage = 'multiscale_submit'
        transformed_characters = $source.Length
    } | ConvertTo-Json
    exit 0
}

& $script -MaximumWaitHours $MaximumWaitHours
exit $LASTEXITCODE
