# Launch FlightSim.exe from a complete AI-GP install.
# Unreal breaks on paths with spaces, so we always run via C:\AIGP_SIM junction.
$ErrorActionPreference = "Stop"

$Junction = "C:\AIGP_SIM"
$minPakBytes = 1GB

function Get-PakSize([string]$root) {
    $pak = Join-Path $root "FlightSim\Content\Paks\FlightSim-WindowsNoEditor.pak"
    if (Test-Path $pak) { return (Get-Item $pak).Length }
    return 0
}

function Find-AigpRoot {
    if ($env:AIGP_ROOT -and (Test-Path $env:AIGP_ROOT)) {
        return (Resolve-Path $env:AIGP_ROOT).Path
    }

    $repo = Resolve-Path (Join-Path $PSScriptRoot "..")
    $candidates = @()

    # Prefer complete installs under Downloads (newest first), then repo-local.
    $downloads = Join-Path $env:USERPROFILE "Downloads"
    $candidates += Get-ChildItem $downloads -Directory -Filter "AI-GP Simulator v*" -EA SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Get-ChildItem $_.FullName -Directory -Filter "AIGP_*" -EA SilentlyContinue } |
        ForEach-Object { $_.FullName }

    $candidates += Get-ChildItem $repo -Directory -Filter "AIGP_*" -EA SilentlyContinue |
        ForEach-Object { $_.FullName }

    foreach ($c in $candidates) {
        $exe = Join-Path $c "FlightSim.exe"
        $pakSize = Get-PakSize $c
        if ((Test-Path $exe) -and ($pakSize -ge $minPakBytes)) {
            Write-Host "Using install: $c (pak $([math]::Round($pakSize/1GB, 2)) GB)"
            return $c
        }
        if ((Test-Path $exe) -and ($pakSize -gt 0)) {
            Write-Warning "Skipping incomplete install (pak $([math]::Round($pakSize/1MB, 1)) MB, need >= 1 GB): $c"
        }
    }

    throw "No complete AI-GP install found (FlightSim.exe + pak >= 1GB). Set AIGP_ROOT or extract the full simulator zip."
}

$root = Find-AigpRoot
$exe = Join-Path $root "FlightSim.exe"

if (Test-Path $Junction) {
    cmd /c rmdir "$Junction" | Out-Null
}
cmd /c mklink /J "$Junction" "$root" | Out-Null

Write-Host "Launching $Junction\FlightSim.exe ..."
Start-Process -FilePath (Join-Path $Junction "FlightSim.exe") -WorkingDirectory $Junction
Write-Host "Started. Then run: uv run main.py"
