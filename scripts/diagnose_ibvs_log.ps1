# Summarize latest (or given) IBVS nav log for Try-1 pass criteria:
# gate pass (n_passed>=1) + COMMIT with small |ex|/|ey|.
param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $LogPath) {
    $LogPath = Get-ChildItem (Join-Path $root "rl\data\nav_log_ibvs_*.csv") |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $LogPath -or -not (Test-Path $LogPath)) {
    Write-Host "No nav_log_ibvs_*.csv found."
    exit 1
}

Write-Host "log: $LogPath"
$rows = Import-Csv $LogPath
Write-Host "rows: $($rows.Count)"
$rows | Group-Object phase | Sort-Object Count -Descending | Format-Table Name, Count

$commit = $rows | Where-Object { $_.phase -eq "COMMIT" }
$nMax = ($rows | ForEach-Object { [int]$_.n_passed } | Measure-Object -Maximum).Maximum
Write-Host "n_passed_max: $nMax"
Write-Host "COMMIT rows: $($commit.Count)"

if ($commit.Count -gt 0) {
    $ex = $commit | ForEach-Object { [math]::Abs([double]$_.ex) }
    $ey = $commit | ForEach-Object { [math]::Abs([double]$_.ey) }
    $exAvg = ($ex | Measure-Object -Average).Average
    $eyAvg = ($ey | Measure-Object -Average).Average
    $exMax = ($ex | Measure-Object -Maximum).Maximum
    $eyMax = ($ey | Measure-Object -Maximum).Maximum
    Write-Host ("COMMIT |ex| avg={0:N3} max={1:N3}  (target <0.08)" -f $exAvg, $exMax)
    Write-Host ("COMMIT |ey| avg={0:N3} max={1:N3}  (target <0.14)" -f $eyAvg, $eyMax)
    $small = ($exMax -lt 0.08) -and ($eyMax -lt 0.14)
    Write-Host "COMMIT small ex/ey: $small"
} else {
    Write-Host "COMMIT small ex/ey: False (no COMMIT)"
}

$pass = ($nMax -ge 1) -and ($commit.Count -gt 0)
Write-Host "Try1 gate1/pass criteria (n_passed>=1 + COMMIT present): $pass"
if (-not ($rows | Where-Object { $_.phase -eq "TRACK" })) {
    Write-Host "diagnose: no TRACK — YOLO pose likely missing (HSV GATE != data.pose)"
}
$spreads = $rows | ForEach-Object { [double]$_.spread } | Sort-Object -Unique
if ($spreads.Count -eq 1) {
    Write-Host "diagnose: spread frozen at $($spreads[0]) — no fresh detections ingested"
}
