# Diagnose GP flight logs for corner-clip / COMMIT bias.
# Usage: make diagnose-gp
#        pwsh scripts/diagnose_gp_log.ps1 [path\to\gp_log.csv]

param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dataDir = Join-Path $root "rl\data"

if (-not $LogPath) {
    $latest = Get-ChildItem -Path $dataDir -Filter "gp_log_*.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        Write-Host "No rl/data/gp_log_*.csv found."
        Write-Host "Default diagnosis (no log): opening-centre / vertical bias + COMMIT while off-axis."
        Write-Host "Re-run make control-flight, then make diagnose-gp."
        exit 0
    }
    $LogPath = $latest.FullName
}

Write-Host "Log: $LogPath"
$rows = Import-Csv $LogPath
if (-not $rows) {
    Write-Host "Empty log."
    exit 1
}

$hasPhase = $rows[0].PSObject.Properties.Name -contains "phase"
$thru = if ($hasPhase) {
    $rows | Where-Object { $_.phase -match "ALIGN|COMMIT|BRAKE" }
} else {
    @()
}

Write-Host ("Total rows: {0}; ALIGN/COMMIT/BRAKE rows: {1}" -f $rows.Count, $thru.Count)

function Summarize($subset, $label) {
    if (-not $subset -or $subset.Count -eq 0) {
        Write-Host "${label}: (none)"
        return
    }
    $bys = @($subset | ForEach-Object { [double]$_.by })
    $bzs = @($subset | ForEach-Object { [double]$_.bz })
    $srcs = $subset | Group-Object source | ForEach-Object { "$($_.Name)=$($_.Count)" }
    Write-Host ("{0}: n={1} by_mean={2:n3} by_abs_max={3:n3} bz_mean={4:n3} bz_abs_max={5:n3} src=[{6}]" -f `
        $label, $subset.Count,
        ($bys | Measure-Object -Average).Average,
        ($bys | ForEach-Object { [math]::Abs($_) } | Measure-Object -Maximum).Maximum,
        ($bzs | Measure-Object -Average).Average,
        ($bzs | ForEach-Object { [math]::Abs($_) } | Measure-Object -Maximum).Maximum,
        ($srcs -join ", "))
}

if ($hasPhase) {
    Summarize ($rows | Where-Object { $_.phase -eq "ALIGN" }) "ALIGN"
    Summarize ($rows | Where-Object { $_.phase -eq "COMMIT" }) "COMMIT"
    Summarize ($rows | Where-Object { $_.phase -eq "BRAKE" }) "BRAKE"
} else {
    Write-Host "No phase column — summarize last 60 rows near gate (bx<6)."
    $near = $rows | Where-Object {
        try { [double]$_.bx -lt 6.0 -and [double]$_.bx -gt 0.1 } catch { $false }
    } | Select-Object -Last 60
    Summarize $near "NEAR"
}

Write-Host ""
Write-Host "Clip heuristic:"
Write-Host "  |by| large at COMMIT -> lateral / late bank"
Write-Host "  |bz| large at COMMIT -> vertical / opening-centre bias (top/bottom)"
Write-Host "  source!=yolo at COMMIT -> dashed on outer frame (bad)"
