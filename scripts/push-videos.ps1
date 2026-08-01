# Publish local run recordings (runs/videos/*.mp4) to the shared `videos`
# branch via Git LFS. All work happens in a throwaway worktree, so the current
# working tree and branch are never touched.

param(
    [string]$Source,
    [string]$TelemetryDir,
    [string]$Branch = "videos",
    [string]$Remote = "origin",
    # An overnight run can leave hundreds of MB of CSV; publish a sane slice.
    [int]$MaxTelemetryMb = 25,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Source) { $Source = Join-Path $Root "runs/videos" }
if (-not $TelemetryDir) { $TelemetryDir = Join-Path $Root "rl/data" }
Set-Location $Root

# git chats on stderr (progress, "Preparing worktree"). Windows PowerShell turns
# that into a terminating NativeCommandError under -ErrorAction Stop, so drop to
# Continue for the call and judge success by exit code alone.
function Test-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & git @GitArgs 2>&1
    } finally {
        $ErrorActionPreference = $prev
    }
    return New-Object psobject -Property @{ Code = $LASTEXITCODE; Output = (($out | ForEach-Object { "$_" }) -join "`n") }
}

# Same call, but a non-zero exit is fatal.
function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $r = Test-Git @GitArgs
    if ($r.Code -ne 0) {
        throw "git $($GitArgs -join ' ') failed:`n$($r.Output)"
    }
    return $r.Output
}

# --- Is there anything to push? ----------------------------------------------

if (-not (Test-Path $Source)) {
    Write-Host "No recordings yet: $Source does not exist. Fly a run first."
    exit 0
}

$videos = @(Get-ChildItem -Path $Source -Filter *.mp4 -File | Sort-Object Name)
if ($videos.Count -eq 0) {
    Write-Host "No .mp4 files in $Source. Fly a run first."
    exit 0
}

if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Error "git-lfs is not installed. Get it from https://git-lfs.com (or 'choco install git-lfs'), then re-run."
    exit 1
}
Invoke-Git lfs install --local | Out-Null

$member = (Test-Git config user.name).Output.Trim()
if (-not $member) { $member = $env:USERNAME }
# The member name ends up in filenames, so keep it boring.
$member = ($member -replace '[^A-Za-z0-9]+', '-').Trim('-').ToLower()
if (-not $member) { $member = "unknown" }

# --- Check out the videos branch somewhere harmless ---------------------------

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("push-videos-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$remoteRef = "refs/remotes/$Remote/$Branch"
$tempBranch = ""
$remoteHas = (Test-Git ls-remote --exit-code --heads $Remote $Branch).Code -eq 0

# Videos already on the branch stay as LFS pointers instead of being downloaded.
# This has to hold for every checkout below -- the rebase in the retry path
# touches teammates' files too, and pushing must never cost a re-download of
# everyone else's runs. It only disables the smudge (checkout) direction; our
# own files still go through the clean filter into LFS on `git add`.
$env:GIT_LFS_SKIP_SMUDGE = "1"

if ($remoteHas) {
    Invoke-Git fetch $Remote "+${Branch}:$remoteRef" | Out-Null
    Invoke-Git worktree add --detach $work $remoteRef | Out-Null
} else {
    Write-Host "Branch '$Branch' does not exist on $Remote yet -- creating it."
    Invoke-Git worktree add --no-checkout --detach $work | Out-Null
    # An orphan needs a branch name; it is local and thrown away in the finally
    # below, since the push targets $Branch by refspec.
    $tempBranch = "push-videos-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    Invoke-Git -C $work switch --orphan $tempBranch | Out-Null
}

try {
    $dest = Join-Path $work "videos"
    $telem = Join-Path $work "telemetry"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    New-Item -ItemType Directory -Path $telem -Force | Out-Null

    # Only the mp4s are LFS payload. Sidecars and CSVs deliberately fall through
    # to regular git objects: small, and that keeps them greppable with
    # `git grep` and readable with `git show` without spending LFS bandwidth.
    $attrPath = Join-Path $work ".gitattributes"
    $attrLines = @(
        'videos/*.mp4 filter=lfs diff=lfs merge=lfs -text',
        # gp_pilot opens its CSV with newline="", so it writes CRLF everywhere.
        'telemetry/*.csv text eol=lf'
    )
    $attrs = @()
    if (Test-Path $attrPath) { $attrs = @(Get-Content $attrPath) }
    $missing = @($attrLines | Where-Object { $attrs -notcontains $_ })
    if ($missing.Count -gt 0) {
        $attrs = @($attrs | Where-Object { $_ -ne "" }) + $missing
        [System.IO.File]::WriteAllText($attrPath, (($attrs -join "`n") + "`n"))
    }

    $readme = Join-Path $dest "README.md"
    if (-not (Test-Path $readme)) {
        Copy-Item (Join-Path $PSScriptRoot "push-videos-README.md") $readme
    }

    # --- Copy in whatever is new ---------------------------------------------

    # Each artifact is judged on its own. Keying the whole run off the mp4 would
    # mean a video published before sidecars existed could never gain one.
    $added = @()          # mp4s -> LFS
    $side = @()           # sidecars + telemetry -> regular git
    $lfsBytes = 0
    $regBytes = 0
    $telemBytes = 0
    $skipped = 0
    $capped = 0
    $maxTelemBytes = $MaxTelemetryMb * 1MB

    foreach ($v in $videos) {
        $name = $v.BaseName
        # Pre-timestamp recordings were all called vision.mp4; stamp those from
        # mtime so two different runs cannot collide on one name and vanish.
        if ($name -notmatch '_\d{8}_\d{6}$') {
            $name = "{0}_{1}" -f $name, $v.LastWriteTime.ToString("yyyyMMdd_HHmmss")
        }
        $stem = "{0}_{1}" -f $member, $name

        $target = Join-Path $dest ("{0}.mp4" -f $stem)
        if (Test-Path $target) {
            $skipped++
        } else {
            if (-not $DryRun) { Copy-Item $v.FullName $target }
            $added += [System.IO.Path]::GetFileName($target)
            $lfsBytes += $v.Length
        }

        # Sidecar: what the run actually did. run_meta writes it beside the mp4.
        $srcJson = Join-Path $v.DirectoryName ($v.BaseName + ".json")
        $dstJson = Join-Path $dest ("{0}.json" -f $stem)
        if ((Test-Path $srcJson) -and -not (Test-Path $dstJson)) {
            if (-not $DryRun) { Copy-Item $srcJson $dstJson }
            $side += "videos/$([System.IO.Path]::GetFileName($dstJson))"
            $regBytes += (Get-Item $srcJson).Length
        }

        # Telemetry: the CSVs this run wrote. Same run id as the video, so the
        # glob is exact rather than a nearest-timestamp guess.
        $runid = $name -replace '^vision_', ''
        $csvs = @(Get-ChildItem -Path $TelemetryDir -Filter ("gp_log_{0}_a*.csv" -f $runid) -File -ErrorAction SilentlyContinue)
        foreach ($c in $csvs) {
            $dstCsv = Join-Path $telem ("{0}_{1}" -f $member, $c.Name)
            if (Test-Path $dstCsv) { continue }
            if (($telemBytes + $c.Length) -gt $maxTelemBytes) { $capped++; continue }
            $telemBytes += $c.Length
            if (-not $DryRun) { Copy-Item $c.FullName $dstCsv }
            $side += "telemetry/$([System.IO.Path]::GetFileName($dstCsv))"
            $regBytes += $c.Length
        }
    }

    if ($added.Count -eq 0 -and $side.Count -eq 0) {
        Write-Host "Already published: all $($videos.Count) local recording(s) are on '$Branch'."
        exit 0
    }

    $mb = [math]::Round($lfsBytes / 1MB, 1)
    $regMb = [math]::Round($regBytes / 1MB, 1)
    Write-Host "Publishing as '$member' ($skipped recording(s) already on the branch):"
    if ($added.Count -gt 0) {
        Write-Host "  $($added.Count) recording(s), ~$mb MB -> Git LFS"
        $added | ForEach-Object { Write-Host "    $_" }
    }
    if ($side.Count -gt 0) {
        Write-Host "  $($side.Count) sidecar/telemetry file(s), ~$regMb MB -> regular git (no LFS quota)"
        $side | ForEach-Object { Write-Host "    $_" }
    }
    if ($capped -gt 0) {
        Write-Host "  ($capped CSV(s) skipped: over -MaxTelemetryMb $MaxTelemetryMb)"
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "-DryRun: nothing was copied, committed, or pushed."
        exit 0
    }

    Invoke-Git -C $work add -A -- videos telemetry .gitattributes | Out-Null
    if (-not (Test-Git -C $work status --porcelain).Output.Trim()) {
        Write-Host "Nothing to commit."
        exit 0
    }
    $msg = if ($added.Count -eq 0) {
        "Add sidecars/telemetry for $($side.Count) file(s) from $member"
    } else {
        "Add $($added.Count) run recording(s) from $member"
    }
    Invoke-Git -C $work commit -m $msg | Out-Null

    Write-Host ""
    Write-Host "Uploading to $Remote/$Branch ($mb MB through LFS -- this takes a while)..."
    $push = Test-Git -C $work push $Remote "HEAD:refs/heads/$Branch"
    if ($push.Code -ne 0) {
        # Someone else published while we were packing.
        Write-Host "Push rejected, rebasing onto the latest '$Branch' and retrying..."
        Invoke-Git fetch $Remote "+${Branch}:$remoteRef" | Out-Null
        Invoke-Git -C $work rebase $remoteRef | Out-Null
        $push = Test-Git -C $work push $Remote "HEAD:refs/heads/$Branch"
        if ($push.Code -ne 0) { throw "Push failed:`n$($push.Output)" }
    }

    Write-Host ""
    if ($added.Count -eq 0) {
        Write-Host "Done -- $($side.Count) sidecar/telemetry file(s) now on '$Branch'."
    } else {
        Write-Host "Done -- $($added.Count) recording(s) (+$($side.Count) sidecar/telemetry) now on '$Branch'."
    }
} finally {
    Set-Location $Root
    Test-Git worktree remove --force $work | Out-Null
    Test-Git worktree prune | Out-Null
    # Only after the worktree is gone, or the branch is still checked out.
    # Long flags on purpose: PowerShell binds a bare -D to the common -Debug
    # parameter and never passes it to git, which silently turns the delete
    # into a create.
    if ($tempBranch) { Test-Git branch --delete --force $tempBranch | Out-Null }
}
