"""VQ2 real-simulator reinforcement-learning pipeline.

A gate-racing PPO policy trained *in the real VQ2 simulator* (not a headless
model) using only VQ2-available signals: camera (YOLO 8-corner + PnP gate pose),
HIGHRES_IMU, and the classical GPEstimation ego-state. No ground-truth
position/velocity/attitude/odometry is used anywhere.

Modules (see .claude/plans/polymorphic-wishing-puddle.md):
    controller  -- policy action -> attitude-rate MAVLink command
    observation -- vision + GPEstimation -> stacked observation vector
    reward      -- reward + termination
    reset       -- episodic reset harness (teleport + re-align + gate tracker)
    gym_env     -- VQ2RealEnv(gym.Env) wrapping the real sim
    train       -- SB3 PPO trainer (single real env, frame-stack, BC warm-start)
    eval        -- evaluation
"""
