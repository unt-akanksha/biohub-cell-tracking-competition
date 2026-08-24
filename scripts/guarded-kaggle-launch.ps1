param(
    [string]$RunId,
    [string]$KernelDir,
    [string]$KernelRef,
    [string]$PreflightReport,
    [string]$AuthorizationId,
    [string]$Nonce,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'

if ($Execute) {
    if (-not $AuthorizationId -or -not $Nonce) {
        throw '-Execute requires -AuthorizationId and -Nonce'
    }
    & python -m biohub_tracker launch execute --authorization-id $AuthorizationId --nonce $Nonce --live --execute
    exit $LASTEXITCODE
}

if (-not $RunId -or -not $KernelDir -or -not $KernelRef -or -not $PreflightReport) {
    throw 'Authorization requires -RunId, -KernelDir, -KernelRef, and -PreflightReport'
}

& python -m biohub_tracker launch authorize --run-id $RunId --kernel-dir $KernelDir --kernel-ref $KernelRef --preflight-report $PreflightReport --live
exit $LASTEXITCODE
