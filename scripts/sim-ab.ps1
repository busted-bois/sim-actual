param(
    [ValidateSet("main", "branch", "jacobian", "main-jacobian", "branch-legacy")]
    [string]$Profile = "jacobian"
)

$env:PILOT_AB = $Profile
Write-Host "[sim-ab] PILOT_AB=$Profile" -ForegroundColor Cyan
Write-Host "  main            = main dynamics, P-only vision"
Write-Host "  branch          = main flight + active_gate telemetry, P-only"
Write-Host "  jacobian        = main flight + yaw-only Jacobian (opt-in)"
Write-Host "  main-jacobian   = same as jacobian"
Write-Host "  branch-legacy   = old low-thrust angle-P (known gate-1 miss)"
Set-Location $PSScriptRoot\..
uv run main.py
