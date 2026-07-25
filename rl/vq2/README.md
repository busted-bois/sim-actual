# VQ2 real-sim RL — runbook

Reinforcement learning that flies the VQ2 course (pass gates, don't crash) using
**only VQ2-available signals**: camera (YOLO 8-corner → PnP gate pose) + IMU via
`GPEstimation`. No ground truth. Training runs **in the live VQ2 simulator**
(single real-time instance), so episodes are recycled by a teleport-based reset
harness. Warm-started by behaviour-cloning the proven GP pilot, then PPO
fine-tunes on real-sim reward (gate passes + collision avoidance).

Pipeline lives in `rl/vq2/`. All commands are `make` targets (raw `uv run`
equivalents given too, in case `make` isn't handy).

---

## 0. One-time setup

```
uv sync                     # install deps  (== make i)
```

## Prerequisites for EVERY command below

1. The **VQ2 simulator is running** and reachable on MAVLink UDP `127.0.0.1:14550`.
2. The sim is in a **Training session on the VQ2 course** (Training exposes the
   `active_gate_index` + `COLLISION` events and lets us arm/teleport; the
   Qualification block would starve the harness).
3. Only **one** MAVLink client at a time on 14550. If a stale client is holding
   the port: `make free-port`.

> The policy still only *sees* vision + IMU (no odometry/ground truth) — Training
> mode is just so the harness can reset episodes and read gate/collision events.

---

## 1. De-risk: benchmark episode reset (run this first)

Confirms automated teleport-reset is fast + reliable enough for unattended
training. If this is slow/flaky, nothing downstream works.

```
make rl2-reset-bench                       # 10 cycles
make rl2-reset-bench CYCLES=20             # more
# raw: uv run -m rl.vq2.reset --cycles 10
```
Expect: reset latency a few seconds, re-align success ~10/10.

## 2. Collect demos (fly the GP pilot, record its actions)

Taps the proven GP pilot flying the real course and logs (observation, action)
pairs to `rl/data/vq2/demos.npz`. **Appends** — run a few good flights to build a
set. Start the race in the sim while this is connected.

```
make rl2-log-demos
# raw: uv run -m rl.vq2.log_demos
```

## 3. Behaviour-clone a starting policy (no sim needed)

Supervised-fits a PPO policy to the demos → `rl/data/vq2/policy_bc.zip`. This is
the warm start so PPO doesn't begin from noise.

```
make rl2-train-bc
# raw: uv run -m rl.vq2.train_bc
```

## 4. Train PPO in the real sim (long-running, unattended)

Fine-tunes from the BC policy on real-sim reward. Real-time single env → expect
hours. The reset harness recycles episodes automatically; leave it running.

```
make rl2-train ARGS="--resume rl/data/vq2/policy_bc.zip"
# raw: uv run -m rl.vq2.train --resume rl/data/vq2/policy_bc.zip
```
Options: `--steps 200000` (default), `--seconds 30` (episode cap), `--gates 6`.
Outputs:
- checkpoints every ~5k steps → `rl/data/vq2/ckpts/vq2_ppo_<N>_steps.zip`
- final policy → `rl/data/vq2/vq2_ppo.zip`
- episode log → `rl/data/vq2/episodes.jsonl`
- TensorBoard → `rl/data/vq2/tb/`

> Start fresh from `policy_bc.zip`, not an old `vq2_ppo_*` checkpoint — earlier
> checkpoints were trained under a since-fixed collision bug.

## 5. Monitor training (while step 4 runs)

**Gate passes + reward breakdown** (re-run anytime):
```
make rl2-log                       # summary + last 15 episodes
make rl2-log ARGS="--tail 40"
make rl2-log ARGS="--all"
# raw: uv run -m rl.vq2.read_log
```
Read it as: `gate 1+ cleared X/N` = is it passing gate 1; the per-term table
shows whether `progress`/`fwd`/`gate_bonus` fire (reward working) vs `terminal`
dominating (dying early).

**Live trends:**
```
tensorboard --logdir rl/data/vq2/tb      # then open http://localhost:6006
```
Watch `episode/gates_passed`, `episode/min_gate_range`, `reward/*`.

**Raw tail (PowerShell):**
```
Get-Content rl/data/vq2/episodes.jsonl -Wait -Tail 20
```

## 6. Evaluate a policy (deterministic, in the sim)

```
make rl2-eval ARGS="--model rl/data/vq2/vq2_ppo.zip --episodes 3"
# a checkpoint instead:
make rl2-eval ARGS="--model rl/data/vq2/ckpts/vq2_ppo_50000_steps.zip --episodes 3"
# raw: uv run -m rl.vq2.eval --model rl/data/vq2/vq2_ppo.zip --episodes 3
```
Each episode prints: gates passed, steps, end reason, reward, closest-gate
range, altitude range, and collision counts (`hard`/`soft`, `threat`/`delta`).

---

## Offline self-tests (no sim)

```
uv run -m rl.vq2.reward         # reward + termination logic
uv run -m rl.vq2.controller     # action → attitude-rate mapping
uv run -m rl.vq2.observation    # obs vector shapes
```

## Typical end-to-end order

```
uv sync
make rl2-reset-bench                                  # de-risk
make rl2-log-demos                                    # (fly GP pilot a few times)
make rl2-train-bc                                     # BC warm start
make rl2-train ARGS="--resume rl/data/vq2/policy_bc.zip"   # PPO (leave running)
make rl2-log                                          # check progress
make rl2-eval ARGS="--model rl/data/vq2/vq2_ppo.zip --episodes 3"
```

## Tuning knobs (env vars / args)

- `VQ2_HARD_THREAT=2` — `COLLISION.threat_level` at/above which an episode is a
  real crash (terminates). Lower it if real crashes come through as low threat;
  check the `threat`/`delta` values printed by `rl2-eval`.
- Reward weights: `rl/vq2/reward.py` (top of file).
- Decision rate / episode length: `DECISION_HZ`, `--seconds` in `rl/vq2/train.py`.
