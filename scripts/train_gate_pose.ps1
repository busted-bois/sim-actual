# Train / refresh YOLO-pose for gate opening corners (inner-weighted).
#
# Stay in pose family (NOT detect-only bbox). Prefer YOLO11s-pose.
# Weight inner-4 keypoints over outer-4; train on real FPV JPEGs from this sim.
#
# Usage:
#   make capture-fpv          # while FlightSim + make view (or control-flight) runs
#   # label keypoints (inner TL,TR,BL,BR then outer) into YOLO-pose dataset
#   make train-gate-pose      # needs CUDA for practical s-model training
#
# Dataset layout (Ultralytics pose):
#   rl/data/gate_pose_ds/
#     images/{train,val}/...
#     labels/{train,val}/...   # class cx cy w h + 8*(x y v)
#     data.yaml
#
# data.yaml skeleton is written if missing. Training is skipped until images exist.

param(
    [string]$Model = "yolo11s-pose.pt",
    [int]$Epochs = 100,
    [int]$Imgsz = 640,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$ds = Join-Path $root "rl\data\gate_pose_ds"
$yaml = Join-Path $ds "data.yaml"
$imgTrain = Join-Path $ds "images\train"
$names = @"
# Gate pose — 1 class, 8 keypoints (inner then outer).
# Inner KPs are the opening (flight target); outer are scale helpers.
path: $($ds -replace '\\','/')
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
"@

New-Item -ItemType Directory -Force -Path $imgTrain | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ds "images\val") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ds "labels\train") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ds "labels\val") | Out-Null
if (-not (Test-Path $yaml)) {
    Set-Content -Path $yaml -Value $names -Encoding utf8
    Write-Host "Wrote $yaml"
}

$nTrain = @(Get-ChildItem $imgTrain -File -ErrorAction SilentlyContinue).Count
if ($nTrain -lt 10) {
    Write-Host "Only $nTrain train images in $imgTrain (need >=10 labeled)."
    Write-Host "1) make capture-fpv while sim FPV is streaming"
    Write-Host "2) Label 8 KPs (inner first) with a YOLO-pose tool"
    Write-Host "3) Re-run make train-gate-pose"
    Write-Host ""
    Write-Host "Train recipe (when ready):"
    Write-Host "  uv run yolo pose train model=$Model data=$yaml epochs=$Epochs imgsz=$Imgsz"
    Write-Host "  Copy best.pt -> simulator/models/gate_pose.pt"
    Write-Host ""
    Write-Host "Inner-weight tip: duplicate inner-KP labels in loss via custom trainer,"
    Write-Host "or train inner-4-only pose model and keep outer for CV/PnP scale."
    exit 0
}

if (-not $Device) {
    $Device = "0"
}

Write-Host "Training $Model on $yaml ($nTrain train images)..."
uv run yolo pose train model=$Model data=$yaml epochs=$Epochs imgsz=$Imgsz device=$Device
Write-Host "Copy runs/pose/train/weights/best.pt to simulator/models/gate_pose.pt when validated."
