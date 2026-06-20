"""Module 8 (deploy) — run the trained policy on the live simulator.

Closes the loop end-to-end on the real sim:
    IMU + odometry --EKF--> filtered state --Module6--> 24-D obs
    --policy.pt--> action --scale_action--> attitude-rate + thrust --> MAVLink

Gate progression mirrors the training env (signed-distance plane crossing),
so the obs the policy sees in deployment matches what it saw in training.

State estimation: the EKF predicts on IMU and updates on the sim's odometry
position + attitude (loosely-coupled). If a trained gatenet.pt is present, the
GateNet->PnP vision pose can be fused too (Modules 3-4); odometry is the
robust default since the sim provides it directly.

ROUND-2 RISKS (read before trusting this on the real sim):
  * SIGN CHECK IS MANDATORY. The training env is canonical; the real sim has
    INVERTED roll + yaw command signs (measured; see rl.fly2). We apply
    SIGN_ROLL/SIGN_YAW below to match, but a flipped sign passes every offline
    test and then crashes on the first real flight -- before any full policy
    flight, run a low-altitude sign-check (command a small +roll then +yaw and
    confirm the drone banks/rotates the expected way).
  * NOT YET CAMERA-ONLY. The obs gate terms here are built from the privileged
    `capture_gate_map()` broadcast, not GateNet->PnP->EKF. For a truly
    camera-only (competition-legal) policy, replace the gate poses with the
    vision estimate. policy.pt was trained on an injected-noise proxy, so expect
    to RE-VALIDATE and likely fine-tune when the real perception is swapped in;
    do not assume it transfers directly. VALIDATE THE PROXY: once GateNet->PnP->
    EKF runs (even offline), log the real noise stats -- position error vs range,
    attitude error magnitude, FOV-dropout frequency near gate crossings -- and
    compare against the env bands (rl.env DR_*_NOISE_RANGE). Same ballpark =>
    transfer is plausible; off by 2-3x => retrain before trusting policy.pt.
  * GATE-FRAME CONVENTION. The live gate map uses local +y as the opening
    normal; this module's `_gate_normal`/`_passed` assume +x. The captured map is
    converted via Rz(90) in `_to_env_gate` (matching rl.env._real_course_map) so
    obs + pass detection match what the policy trained on.

    uv run -m rl.deploy --sign-check    # FIRST: verify roll/yaw signs on the sim
    uv run -m rl.deploy                 # then fly the policy on the live sim
    uv run -m rl.deploy --selftest      # closed-loop on internal env (no sim)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rl import spec
from rl.ekf import ESKF
from rl.env import DECISION_HZ, GateRacingEnv, _quat_mult, _quat_norm
from rl.observation import build_observation
from rl.train_ppo import POLICY_PT, StandalonePolicy

# Real-sim command-sign conventions. These match rl.fly2 (which clears the
# course) AND the direct rate-ID measurement (rl.rate_id): +roll command drives
# the roll ANGLE negative (banks left), so banking right needs -1; pitch passes
# through; yaw inverted. The earlier open-loop --sign-check read roll/yaw as +1,
# but that was fooled by the sim's ~2.7x rate overshoot + drift -- the closed-loop
# measurement wins. RE-VERIFY with --sign-check after any change.
SIGN_ROLL = -1.0
SIGN_PITCH = +1.0
SIGN_YAW = -1.0


def _to_env_gate(g):
    """Convert a live-sim gate (local +y opening normal) into the env/policy
    convention (+x normal) by post-rotating its quat by Rz(90deg) -- identical to
    rl.env._real_course_map. The policy trained on the converted frame, so the
    captured map MUST get the same rotation or every gate-normal obs term (and
    pass detection) is rotated 90deg and the policy flies the course wrong."""
    qz = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    g = dict(g)
    g["quat"] = _quat_norm(_quat_mult(np.asarray(g["quat"], float), qz)).tolist()
    return g


def load_policy(path: str = POLICY_PT, device: str = "cpu"):
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev)
    net = StandalonePolicy(
        obs_dim=ckpt.get("obs_dim", spec.OBS_DIM),
        act_dim=ckpt.get("act_dim", spec.ACTION_DIM),
        arch=ckpt.get("arch", [64, 64, 64]),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.to(dev)
    net.eval()

    @torch.no_grad()
    def act(obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, np.float32)[None]).to(dev)
        return np.clip(net(x)[0].cpu().numpy(), -1.0, 1.0)

    return act


def _gate_normal(g):
    return spec.quat_to_R(np.asarray(g["quat"], float)) @ np.array([1.0, 0, 0])


def _passed(p, prev_signed, g):
    """Return (passed, new_signed)."""
    gc = np.asarray(g["pos"], float)
    R = spec.quat_to_R(np.asarray(g["quat"], float))
    n = R[:, 0]  # gate normal (through-axis)
    signed = float(n @ (p - gc))
    if prev_signed < 0.0 <= signed:
        # Square opening: offsets along the gate's local right/down axes vs its
        # reported w/h.
        rel = p - gc
        dy = abs(float(R[:, 1] @ rel))  # width axis (right)
        dz = abs(float(R[:, 2] @ rel))  # height axis (down)
        hw = 0.5 * float(g.get("w", spec.GATE_SIZE_M))
        hh = 0.5 * float(g.get("h", spec.GATE_SIZE_M))
        return (dy < hw and dz < hh), signed
    return False, signed


def _takeoff(sim, gate_map, secs=3.0):
    """Lift off and level the drone before the policy takes over -- the policy
    trained airborne and won't initiate a ground takeoff (it commands < hover
    thrust at the grounded spawn). Reuses fly2's proven leveling gains/signs and
    climbs to ~gate-0 altitude."""
    from rl.fly2 import rpy

    k_att, rate_clip = 0.6, 0.30
    tgt_z = min(float(gate_map[0]["pos"][2]), -2.0)  # NED: at least ~2 m up
    hz = 100
    t0 = time.time()
    while time.time() - t0 < secs:
        snap = sim.snapshot()
        if snap.has_pose():
            roll, pitch, _ = rpy(snap.quat)
            z = float(snap.pos_ned[2])
            roll_cmd = float(np.clip(SIGN_ROLL * k_att * -roll, -rate_clip, rate_clip))
            pitch_cmd = float(np.clip(SIGN_PITCH * k_att * -pitch, -rate_clip, rate_clip))
            thrust = 0.45 if z > tgt_z + 0.4 else spec.HOVER_THRUST  # climb, then hold
            sim.send_attitude_rates(roll_cmd, pitch_cmd, 0.0, thrust)
        time.sleep(1.0 / hz)
    print("[deploy] takeoff complete -> handing to policy", flush=True)


class PolicyRunner:
    def __init__(self, policy_path: str = POLICY_PT):
        self.act = load_policy(policy_path)
        self.ekf = ESKF()
        self.last_action = np.zeros(spec.ACTION_DIM)
        self.gate_idx = 0
        self._prev_signed = None
        self._last_imu_t = None
        self._dbgn = 0

    def run(self):
        from rl.sim_interface import SimInterface, load_gate_map

        sim = SimInterface()
        if not sim.wait_for_telemetry():
            print("[deploy] no telemetry — is the simulator running?")
            return
        # Load the course from the saved gate_map.json (same source as rl.fly2 and
        # the training env) rather than racing the sim's one-shot start broadcast,
        # then convert each gate to the policy's +x-normal frame.
        gate_map = [_to_env_gate(g) for g in load_gate_map()]
        if not gate_map:
            print("[deploy] no gate map in gate_map.json; run `make capture-gates`")
            return
        print(f"[deploy] loaded {len(gate_map)} gates from gate_map.json", flush=True)

        # Sync to race start: the drone only actuates once the race is live and
        # airborne (same reason rl.fly2 waits on ENTER). Seed the EKF from the
        # airborne pose AFTER this, so it starts from a valid state.
        print(
            "[deploy] START THE RACE, then press ENTER once the drone is airborne...",
            flush=True,
        )
        try:
            input()
        except EOFError:
            pass

        # Confirm the sim is streaming LIVE data via the IMU clock (the drone may
        # legitimately sit static on the ground pre-takeoff, so DON'T check pose).
        i0 = sim.snapshot().imu
        time.sleep(0.4)
        i1 = sim.snapshot().imu
        if i0 is not None and i1 is not None and i0["time_us"] == i1["time_us"]:
            print(
                "[deploy] ABORT: telemetry FROZEN (IMU clock not advancing) -- "
                "start the race first, then re-run.",
                flush=True,
            )
            return

        sim.arm()
        time.sleep(0.2)
        _takeoff(sim, gate_map)  # get airborne + level before the policy flies

        snap = sim.snapshot()
        p0 = np.asarray(snap.pos_ned, float) if snap.has_pose() else np.zeros(3)
        self._prev_signed = float(
            _gate_normal(gate_map[0]) @ (p0 - np.array(gate_map[0]["pos"]))
        )
        print("[deploy] flying policy...", flush=True)

        while True:
            snap = sim.snapshot()
            if not snap.has_pose():
                time.sleep(1.0 / DECISION_HZ)
                continue

            # Read the sim's odometry directly — the same source rl.fly2 uses to
            # clear the course. (The EKF's IMU-driven velocity prediction diverges
            # to 100s of m/s here, so it's kept out of the control path; it was
            # scaffolding for the Round-2 vision fusion, not needed off odometry.)
            p = np.asarray(snap.pos_ned, float)
            v = np.asarray(snap.vel_ned, float)
            q = np.asarray(snap.quat, float)
            ang_vel = np.array(snap.ang_vel) if snap.ang_vel else np.zeros(3)

            # Gate progression.
            if self.gate_idx < len(gate_map):
                passed, self._prev_signed = _passed(
                    p, self._prev_signed, gate_map[self.gate_idx]
                )
                if passed:
                    self.gate_idx += 1
                    print(
                        f"[deploy] gate {self.gate_idx}/{len(gate_map)} passed",
                        flush=True,
                    )
                    if self.gate_idx >= len(gate_map):
                        print("[deploy] COURSE COMPLETE", flush=True)
                        sim.send_attitude_rates(0, 0, 0, spec.HOVER_THRUST)
                        return
                    g = gate_map[self.gate_idx]
                    self._prev_signed = float(
                        _gate_normal(g) @ (p - np.array(g["pos"]))
                    )

            obs = build_observation(
                p,
                v,
                q,
                ang_vel,
                gate_map,
                self.gate_idx,
                self.last_action[:3],
            )
            action = self.act(obs)
            self.last_action = action
            self._dbgn += 1
            if self._dbgn % 8 == 0:  # ~6 Hz: what the policy sees and commands
                print(
                    f"[dbg] n={self._dbgn} gi={self.gate_idx} "
                    f"p={np.round(p, 1)} "
                    f"v={np.round(v, 1)}(|{np.linalg.norm(v):.1f}|) "
                    f"q={np.round(q, 2)} w={np.round(ang_vel, 1)} "
                    f"to_gate={np.round(obs[0:3], 2)} dist={obs[3]:.1f} "
                    f"act={np.round(action, 2)}",
                    flush=True,
                )
            roll, pitch, yaw, thrust = spec.scale_action(action)
            # Canonical (training) -> real sim signs (verified via --sign-check).
            sim.send_attitude_rates(
                SIGN_ROLL * roll, SIGN_PITCH * pitch, SIGN_YAW * yaw, thrust
            )
            # Run at the policy's TRAINED decision rate. The policy is a 50 Hz
            # feedback controller; stepping it at 100 Hz doubles its effective
            # loop gain and makes it oscillate/shake instead of fly.
            time.sleep(1.0 / DECISION_HZ)


def _selftest():
    if not os.path.exists(POLICY_PT):
        print("[selftest] no policy.pt — run `uv run -m rl.train_ppo --quick` first")
        return
    act = load_policy(POLICY_PT)
    # Closed-loop on the internal env (deterministic policy) — verifies the full
    # obs->policy->action loop runs and the exported weights drive the sim model.
    for stage in (0, 2):
        env = GateRacingEnv(stage=stage, seed=7)
        o, _ = env.reset()
        assert o.shape == (spec.OBS_DIM,)
        tot, steps = 0.0, 0
        term = trunc = False
        while not (term or trunc):
            a = act(o)
            assert a.shape == (spec.ACTION_DIM,) and np.all(np.isfinite(a))
            o, r, term, trunc, info = env.step(a)
            tot += r
            steps += 1
        print(
            f"[selftest] stage {stage}: policy ran {steps} steps, "
            f"reward={tot:.1f}, gate_idx={env.gate_idx}, info={info}"
        )
    print("[selftest] OK — policy.pt loads, infers, and drives the loop")


def _sign_check(hold: float = 1.2):
    """Live command-sign check — run BEFORE any full policy flight. Hovers, then
    commands a small +roll and a small +yaw (with SIGN_* applied, as in flight),
    printing the expected motion so the operator can confirm the real sim matches.
    A flipped sign passes every offline test and crashes on the first flight."""
    from rl.sim_interface import SimInterface

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[sign-check] no telemetry — is the simulator running?")
        return
    # The drone only accepts control once the race is live (it's airborne at race
    # start). Sync to it like rl.fly2 does, since there are no gates to wait on.
    print(
        "[sign-check] START THE RACE now, then press ENTER once the drone is "
        "airborne...",
        flush=True,
    )
    try:
        input()
    except EOFError:
        pass
    sim.arm()
    hz = 100

    def drive(label, roll=0.0, pitch=0.0, yaw=0.0, secs=hold):
        print(f"[sign-check] {label}", flush=True)
        for _ in range(int(secs * hz)):
            sim.send_attitude_rates(
                SIGN_ROLL * roll, SIGN_PITCH * pitch, SIGN_YAW * yaw, spec.HOVER_THRUST
            )
            time.sleep(1.0 / hz)

    drive("HOVER — settle, watch the drone BODY (not the scenery)", secs=2.5)
    drive(">>> ROLL — drone should BANK RIGHT (right side dips)", roll=0.5, secs=0.8)
    drive("    holding — note the tilt direction", roll=0.0, secs=1.2)
    drive("    leveling", roll=-0.5, secs=0.8)
    drive("HOVER — settle", secs=2.0)
    drive(">>> PITCH — NOSE should tip UP", pitch=0.5, secs=0.7)
    drive("    holding — note nose up or down", pitch=0.0, secs=1.0)
    drive("    leveling", pitch=-0.5, secs=0.7)
    drive("HOVER — settle", secs=2.0)
    drive(">>> YAW — NOSE should swing RIGHT (toward 3 o'clock)", yaw=0.5, secs=1.5)
    drive("    holding — note nose left or right", yaw=0.0, secs=0.8)
    drive("    returning", yaw=-0.5, secs=1.5)
    drive("HOVER — settle", secs=1.0)
    sim.send_attitude_rates(0.0, 0.0, 0.0, spec.HOVER_THRUST)
    print(
        "[sign-check] done — if either motion was REVERSED, flip the matching "
        "SIGN_ROLL/SIGN_YAW in rl/deploy.py before flying the policy.",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--sign-check",
        action="store_true",
        help="live roll/yaw sign check (run before the first full flight)",
    )
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.sign_check:
        _sign_check()
    else:
        PolicyRunner().run()
