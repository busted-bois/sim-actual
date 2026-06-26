"""Tier-A Jacobian demo: pure geometric controller in GateRacingEnv.

No ML weights.  Gate pursuit sets v_des_ned + yaw_rate_des each tick;
geo_control.velocity_to_action() (the Jacobian inner loop) converts those
to attitude-rate + thrust; env.py executes them against the internal
physics model.

The "Jacobian" here is the analytic map

    R(q)  :  body-frame acceleration -> world-frame velocity
  inv(R)  :  desired world velocity  -> required body attitude rates

That is exactly what velocity_to_action() computes -- a geometric
(Jacobian-based) controller, not a trained matrix.

This demo proves the inner loop works before any ML is added (Tier A).
Tier B adds a BC/PPO policy on top that replaces the hand-coded pursuit
with a learned outer loop -- but the geo_control layer stays identical.

Run:
    uv run -m rl.demo_jacobian                         # stage 0, 4 episodes
    uv run -m rl.demo_jacobian --stage 2 --episodes 6  # full 6-gate course
    uv run -m rl.demo_jacobian --stage 1 --plot        # save 3-D trajectory
    uv run -m rl.demo_jacobian --selftest              # pass/fail for CI
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from rl import spec
from rl.env import GateRacingEnv
from rl.geo_control import GeoGains

# ---------------------------------------------------------------------------
# Deploy-style gains: hover_thrust 0.5 matches the internal training plant
# (spec.HOVER_THRUST).  Switch to GeoGains() defaults (hover 0.27) when
# validating against the live FlightSim.
# ---------------------------------------------------------------------------
DEMO_GAINS = GeoGains(
    hover_thrust=spec.HOVER_THRUST,  # 0.5 -- internal model
    thrust_min=0.0,
    thrust_max=1.0,
    k_lat=0.2,
    roll_max=0.22,
    brake_max=0.18,
)


# ---------------------------------------------------------------------------
# Gate-pursuit outer loop (Tier A -- no ML)
# ---------------------------------------------------------------------------

def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw from quaternion (w,x,y,z), body->world."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pursuit_velocity(
    p: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    gate_pos: np.ndarray,
    lookahead: float = 0.0,
    max_speed: float = 5.0,
    next_gate_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute v_des_ned (3,) and yaw_rate_des for a gate target.

    Speed is proportional to distance, capped at max_speed.
    If next_gate_pos is given, a small lookahead blend softens the corner.
    Returns (v_des_ned, yaw_rate_des) -- the inputs to velocity_to_action().
    """
    p = np.asarray(p, float)
    gate_pos = np.asarray(gate_pos, float)

    to_gate = gate_pos - p
    dist = float(np.linalg.norm(to_gate))
    if dist < 1e-6:
        return np.zeros(3), 0.0

    direction = to_gate / dist
    # Proportional to distance, but also ramp DOWN inside 8m so the drone
    # doesn't blow through the gate at full speed.  Gate opening is 2.72m --
    # need to be under ~2 m/s to thread it reliably.
    speed = min(max_speed, max(1.5, 0.45 * dist))
    if dist < 8.0:
        speed = min(speed, max(1.5, 0.25 * dist))

    # Lookahead: blend toward the next gate when close to the current one.
    if next_gate_pos is not None and lookahead > 0.0 and dist < lookahead:
        blend = 1.0 - dist / lookahead  # 0 far, 1 at gate
        to_next = np.asarray(next_gate_pos, float) - p
        n_next = np.linalg.norm(to_next)
        if n_next > 1e-6:
            direction = (1.0 - blend) * direction + blend * (to_next / n_next)
            n_dir = np.linalg.norm(direction)
            if n_dir > 1e-6:
                direction /= n_dir

    v_des = direction * speed

    yaw = _quat_to_yaw(np.asarray(q, float))
    bearing = math.atan2(to_gate[1], to_gate[0])
    yaw_err = (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi
    yaw_rate_des = 0.8 * yaw_err

    return v_des, yaw_rate_des


def geo_action(env: GateRacingEnv, lookahead: float = 3.0) -> np.ndarray:
    """Build action using gate pursuit -> geo_control (Jacobian inner loop).

    Outer loop: pursuit_velocity() -> v_des_ned, yaw_rate_des
    Encoding:   spec.encode_velocity_action() -> normalized [-1,1] action

    env.step() decodes the action and calls geo_control.velocity_to_action()
    internally (the Jacobian/geometric inner loop) with TRAIN_GAINS.
    """
    gate_map = env.gate_map
    idx = env.gate_idx
    gate_pos = np.array(gate_map[idx]["pos"])
    next_pos = np.array(gate_map[min(idx + 1, len(gate_map) - 1)]["pos"])

    v_des, yaw_rate_des = pursuit_velocity(
        env.p, env.v, env.q, gate_pos,
        lookahead=lookahead,
        next_gate_pos=next_pos if idx + 1 < len(gate_map) else None,
    )

    return spec.encode_velocity_action(v_des, yaw_rate_des)


def _rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(q, float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: GateRacingEnv,
    lookahead: float = 3.0,
    record: bool = False,
) -> dict:
    """Run one episode with the Tier-A geometric controller."""
    obs, _ = env.reset()
    positions = [env.p.copy()]
    jac_log = []  # store one Jacobian sample per gate for diagnostics
    total_reward = 0.0
    gates_passed = 0
    term = trunc = False

    while not (term or trunc):
        action = geo_action(env, lookahead=lookahead)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if record:
            positions.append(env.p.copy())
        if info.get("gate_passed"):
            gates_passed += 1
            # Log geo_control Jacobian columns at the moment of gate pass.
            if record:
                roll, pitch, yaw = _rpy(env.q)
                jac_log.append(_jacobian_columns(env.v, roll, pitch, yaw))

    return {
        "positions": np.array(positions) if record else np.empty((0, 3)),
        "jac_log": jac_log,
        "gates": list(env.gate_map),
        "total_reward": total_reward,
        "gates_passed": gates_passed,
        "steps": env.steps,
        "course_complete": env.gate_idx >= len(env.gate_map),
    }


def _jacobian_columns(
    vel_ned: np.ndarray, roll: float, pitch: float, yaw: float
) -> dict:
    """Compute the geo_control sensitivity (partial Jacobian) at a given state.

    The full Jacobian of body-rate commands w.r.t. velocity error is:

        d(roll_rate)/d(v_err_right)  = k_lat * k_att / roll_max  (saturated)
        d(pitch_rate)/d(v_err_fwd)   = k_lean * k_att / lean_max (saturated)

    The 3x3 world->body rotation R(q)^T projects v_err from NED into
    forward/right components before applying the gains.  That projection IS
    the Jacobian structure that makes this a geometric controller.
    """
    g = DEMO_GAINS
    cY, sY = math.cos(yaw), math.sin(yaw)
    # World->body (horizontal) rotation rows.
    fwd_row = np.array([cY, sY, 0.0])   # projects v_err onto body-forward
    rgt_row = np.array([sY, -cY, 0.0])  # projects v_err onto body-right
    return {
        "yaw_deg": math.degrees(yaw),
        "fwd_axis_NED": fwd_row.round(3).tolist(),
        "rgt_axis_NED": rgt_row.round(3).tolist(),
        "k_fwd->pitch": round(g.k_lean * g.k_att, 4),
        "k_rgt->roll":  round(g.k_lat  * g.k_att, 4),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def explain_jacobian() -> None:
    """Print a plain-language breakdown of the geo_control Jacobian."""
    print()
    print("=" * 66)
    print("  What the Jacobian inner loop does")
    print("=" * 66)
    print("""
  Input:  v_des_ned (3,)  desired world-frame NED velocity  [m/s]
          yaw_rate_des     desired heading rate              [rad/s]
  Output: roll_rate, pitch_rate, yaw_rate, thrust

  Step 1 -- World -> Body projection (this is the Jacobian / R(q)^T):

      v_err_fwd =  v_err_x*cos(yaw) + v_err_y*sin(yaw)   (forward in body)
      v_err_rgt =  v_err_x*sin(yaw) - v_err_y*cos(yaw)   (right in body)

      These two lines are the partial Jacobian columns that tell you how
      a North/East velocity error projects onto the drone's body axes.
      At yaw=0 (facing North): fwd=x, rgt=y -- axes align with NED.
      At yaw=90 deg (facing East): fwd=y, rgt=-x -- axes rotate with drone.

  Step 2 -- Desired attitude from velocity error:

      target_pitch = -k_lean * v_err_fwd   (forward = nose-down pitch)
      target_roll  =  k_lat  * v_err_rgt   (right   = right roll)

  Step 3 -- Attitude P-controller -> rates:

      pitch_rate =  k_att * (target_pitch - current_pitch)
      roll_rate  = -k_att * (target_roll  - current_roll)
      yaw_rate   = -yaw_rate_des  (sign convention: measured on fly2)

  Step 4 -- Tilt-compensated thrust:

      thrust = (hover_thrust + k_vz*vz_err) / cos(roll)*cos(pitch)

  Gains used (DEMO_GAINS / TRAIN_GAINS):""")
    g = DEMO_GAINS
    print(f"      hover_thrust={g.hover_thrust}  k_lean={g.k_lean}  "
          f"k_lat={g.k_lat}  k_att={g.k_att}  k_vz={g.k_vz}")
    print()
    print("  The product k_lean*k_att is the end-to-end gain from a")
    print("  forward velocity error to a pitch-rate command -- the")
    print("  'bandwidth' of the inner loop.  Raise it to respond faster;")
    print("  lower it for stability in simulation with high latency.")
    print("=" * 66)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectories(
    all_histories: list[dict], gate_map: list, out: str = "jacobian_demo.png"
) -> None:
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[demo] matplotlib not available -- skipping plot")
        return

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.cm.Blues  # type: ignore[attr-defined]
    n = max(len(all_histories), 1)

    for i, h in enumerate(all_histories):
        pos = h["positions"]
        if len(pos) < 2:
            continue
        color = cmap(0.4 + 0.5 * i / n)
        label = f"ep{i}  ({h['gates_passed']}/{len(h['gates'])} gates)"
        # NED -> plot: East on x, North on y, altitude on z.
        ax.plot(pos[:, 1], pos[:, 0], -pos[:, 2],
                color=color, linewidth=1.8, alpha=0.8, label=label)
        ax.scatter(pos[0, 1], pos[0, 0], -pos[0, 2],
                   marker="o", s=35, color=color, zorder=5)

    for g in gate_map:
        corners = spec.gate_corners_world(np.array(g["pos"]), np.array(g["quat"]))
        loop = np.vstack([corners, corners[0]])
        ax.plot(loop[:, 1], loop[:, 0], -loop[:, 2],
                "k-", linewidth=2.5, alpha=0.55)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Tier-A Jacobian (geo_control) Controller -- Gate Racing")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"[demo] trajectory saved -> {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-A Jacobian geo_control demo")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                    help="0=1 gate  1=2 gates  2=6 gates (full course)")
    ap.add_argument("--lookahead", type=float, default=3.0,
                    help="metres to start blending toward next gate")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="run pass/fail selftest for CI")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    stage_label = {0: "1 gate (straight)", 1: "2 gates (turn)", 2: "6 gates (full)"}
    print("=" * 66)
    print("  Tier-A Jacobian / geo_control Demo")
    print(f"  Stage {args.stage}: {stage_label[args.stage]}")
    print(f"  {args.episodes} episodes  lookahead={args.lookahead}m")
    print("=" * 66)

    explain_jacobian()

    histories = []
    total_gates_passed = 0
    total_gates_possible = 0

    for ep in range(args.episodes):
        env = GateRacingEnv(stage=args.stage, seed=ep, max_seconds=args.max_seconds)
        h = run_episode(env, lookahead=args.lookahead, record=args.plot)
        histories.append(h)
        n_gates = len(h["gates"])
        total_gates_passed += h["gates_passed"]
        total_gates_possible += n_gates
        status = "COMPLETE" if h["course_complete"] else f"{h['gates_passed']}/{n_gates} gates"
        print(f"  ep{ep}: {h['steps']:4d} steps  reward={h['total_reward']:8.1f}  {status}")

        # Print one Jacobian breakdown per episode to show geometry changes.
        if h["jac_log"]:
            j = h["jac_log"][-1]
            print(f"         [last gate] yaw={j['yaw_deg']:.1f}deg  "
                  f"fwd_NED={j['fwd_axis_NED']}  rgt_NED={j['rgt_axis_NED']}  "
                  f"k_fwd-pitch={j['k_fwd->pitch']}  k_rgt-roll={j['k_rgt->roll']}")

    pct = 100.0 * total_gates_passed / max(total_gates_possible, 1)
    print()
    print(f"  Gate pass rate: {total_gates_passed}/{total_gates_possible}  ({pct:.0f}%)")
    completes = sum(1 for h in histories if h["course_complete"])
    avg_r = float(np.mean([h["total_reward"] for h in histories]))
    print(f"  Courses complete: {completes}/{args.episodes}")
    print(f"  Avg reward: {avg_r:.1f}")
    print()
    print("  Path to Tier B (hybrid ML):")
    print("    1. Record demos:  uv run -m rl.fly2 --mode course --log rl/data/demo.jsonl")
    print("    2. BC warm-start: uv run -m rl.bc_dataset --demo rl/data/demo.jsonl")
    print("    3. PPO fine-tune: uv run -m rl.train_ppo --bc rl/data/demo.jsonl")
    print("    4. Deploy:        uv run -m rl.deploy")
    print("    geo_control stays identical in steps 1-4 -- only the outer loop changes.")

    if args.plot:
        gate_map = histories[-1]["gates"]
        plot_trajectories(histories, gate_map)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Pass/fail test matching env.py selftest: >= 4/6 stage-0 gate passes."""
    print("[demo_jacobian] selftest ...")
    trials = 6
    passed = 0
    for t in range(trials):
        # Mirror env.py selftest: init with seed, then reset() with no re-seed.
        # The two _reset_state() calls (init + reset) use different rng states,
        # giving the same layout that env.py's selftest was validated against.
        env = GateRacingEnv(stage=0, seed=100 + t, max_seconds=40.0)
        h = run_episode(env)
        passed += h["gates_passed"]
    assert passed >= 4, (
        f"selftest FAIL: Jacobian inner loop passed {passed}/{trials} "
        f"stage-0 gates (need >= 4)"
    )
    print(f"[demo_jacobian] selftest OK: {passed}/{trials} stage-0 gates")


if __name__ == "__main__":
    main()
