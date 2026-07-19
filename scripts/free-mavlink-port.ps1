param(
    # MAVLink + FPV camera (stale listeners on 5600 block IBVS vision_rx bind).
    [int[]]$Ports = @(14550, 5600)
)

$ErrorActionPreference = "Stop"

function Get-UdpHolders([int]$Port) {
    $ids = @()
    try {
        $ids = Get-NetUDPEndpoint -LocalPort $Port -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique
    } catch {
        $ids = netstat -ano -p UDP |
            Select-String ":$Port\s" |
            ForEach-Object { ($_ -split '\s+')[-1] } |
            Sort-Object -Unique
    }
    return @($ids | Where-Object { $_ -and $_ -ne 0 })
}

$any = $false
foreach ($Port in $Ports) {
    $procIds = Get-UdpHolders $Port
    if (-not $procIds) {
        Write-Host "UDP $Port is free."
        continue
    }
    $any = $true
    foreach ($procId in $procIds) {
        try {
            $p = Get-Process -Id $procId -ErrorAction Stop
            Stop-Process -Id $procId -Force
            Write-Host "Killed $($p.ProcessName) (PID $procId) holding UDP $Port."
        } catch {
            Write-Host "Could not kill PID ${procId}: $_"
        }
    }
}
if (-not $any) { exit 0 }
