# Capture FPV JPEGs for YOLO-pose retrain.
# Prefer env-gated dump inside GatePoseRunner (simulator logic):
#
#   set GATE_CAPTURE_DIR=rl\data\gate_pose_ds\images\train
#   set GATE_CAPTURE_EVERY=5
#   make view
#   # or: make control-flight
#
# Then label 8 keypoints (inner TL,TR,BL,BR then outer) and run:
#   make train-gate-pose

Write-Host "Use GATE_CAPTURE_DIR with make view / make control-flight (see Makefile capture-fpv)."
Write-Host "Images land in rl/data/gate_pose_ds/images/train/"
Write-Host "Label inner-4 first, then: make train-gate-pose"
