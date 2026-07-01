param([int]$Port = 14550)

$ErrorActionPreference = "Stop"

$procIds = @()
try {
    $procIds = Get-NetUDPEndpoint -LocalPort $Port -ErrorAction Stop |
        Select-Object -ExpandProperty OwningProcess -Unique
} catch {
    $procIds = netstat -ano -p UDP |
        Select-String ":$Port\s" |
        ForEach-Object { ($_ -split '\s+')[-1] } |
        Sort-Object -Unique
}

$procIds = $procIds | Where-Object { $_ -and $_ -ne 0 }

if (-not $procIds) {
    Write-Host "UDP $Port is free."
    exit 0
}

foreach ($procId in $procIds) {
    try {
        $p = Get-Process -Id $procId -ErrorAction Stop
        Stop-Process -Id $procId -Force
        Write-Host "Killed $($p.ProcessName) (PID $procId) holding UDP $Port."
    } catch {
        Write-Host "Could not kill PID ${procId}: $_"
    }
}
