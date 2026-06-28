# Vision pilot A/B evaluation
#
# Baseline (main dynamics, P-only vision):
#   make sim
#
# Jacobian yaw blend (conservative, opt-in):
#   make sim-ab-jacobian
#   powershell -File scripts/sim-ab.ps1 -Profile jacobian
#
# Reliable full course (odometry, no pixel steering):
#   make auto
#
# Pass criteria (make sim / sim-ab-jacobian):
#   - Gate 1 forward pass, no 180 spin
#   - No false fly-through (gates_passed matches sim active_gate_index)
#   - Live window shows gate bbox (runs/vision.mp4 recorded)
#
# Pass criteria (make auto):
#   - Terminal: [RACE] OUTCOME=success
