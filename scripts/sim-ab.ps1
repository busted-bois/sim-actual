param(
    [ValidateSet("main", "branch", "jacobian", "main-jacobian", "branch-legacy")]
    [string]$Profile = "main"
)

$env:PILOT_AB = $Profile
Write-Host "[sim-ab] PILOT_AB=$Profile" -ForegroundColor Cyan
Set-Location $PSScriptRoot\..
uv run main.py
