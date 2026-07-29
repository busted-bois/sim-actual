# Publish local run recordings (runs/videos/*.mp4) to the shared `videos`
# branch via Git LFS. All work happens in a throwaway worktree, so the current
# working tree and branch are never touched.

param(
    [string]$Source,
    [string]$Branch = "videos",
    [string]$Remote = "origin",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Source) { $Source = Join-Path $Root "runs/videos" }
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
    New-Item -ItemType Directory -Path $dest -Force | Out-Null

    # The videos are LFS payload; the branch is useless without this rule.
    $attrPath = Join-Path $work ".gitattributes"
    $attrLine = 'videos/*.mp4 filter=lfs diff=lfs merge=lfs -text'
    $attrs = @()
    if (Test-Path $attrPath) { $attrs = @(Get-Content $attrPath) }
    if ($attrs -notcontains $attrLine) {
        $attrs = @($attrs | Where-Object { $_ -ne "" }) + $attrLine
        [System.IO.File]::WriteAllText($attrPath, (($attrs -join "`n") + "`n"))
    }

    $readme = Join-Path $dest "README.md"
    if (-not (Test-Path $readme)) {
        Copy-Item (Join-Path $PSScriptRoot "push-videos-README.md") $readme
    }

    # --- Copy in whatever is new ---------------------------------------------

    $added = @()
    $addedBytes = 0
    $skipped = 0
    foreach ($v in $videos) {
        $name = $v.BaseName
        # Pre-timestamp recordings were all called vision.mp4; stamp those from
        # mtime so two different runs cannot collide on one name and vanish.
        if ($name -notmatch '_\d{8}_\d{6}$') {
            $name = "{0}_{1}" -f $name, $v.LastWriteTime.ToString("yyyyMMdd_HHmmss")
        }
        $target = Join-Path $dest ("{0}_{1}{2}" -f $member, $name, $v.Extension)
        if (Test-Path $target) { $skipped++; continue }
        if (-not $DryRun) { Copy-Item $v.FullName $target }
        $added += [System.IO.Path]::GetFileName($target)
        $addedBytes += $v.Length
    }

    if ($added.Count -eq 0) {
        Write-Host "Already published: all $($videos.Count) local recording(s) are on '$Branch'."
        exit 0
    }

    $mb = [math]::Round($addedBytes / 1MB, 1)
    Write-Host "Publishing $($added.Count) recording(s) as '$member' (~$mb MB; $skipped already on the branch):"
    $added | ForEach-Object { Write-Host "  $_" }

    if ($DryRun) {
        Write-Host ""
        Write-Host "-DryRun: nothing was copied, committed, or pushed."
        exit 0
    }

    Invoke-Git -C $work add -A -- videos .gitattributes | Out-Null
    if (-not (Test-Git -C $work status --porcelain).Output.Trim()) {
        Write-Host "Nothing to commit."
        exit 0
    }
    Invoke-Git -C $work commit -m "Add $($added.Count) run recording(s) from $member" | Out-Null

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
    Write-Host "Done -- $($added.Count) recording(s) now on '$Branch'."
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
