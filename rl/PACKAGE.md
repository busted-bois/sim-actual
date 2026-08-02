# RL Package Map

This map is the authoritative source for the atomic hierarchy migration. Package
`__init__.py` files remain documentation-only; callers import concrete modules.

## Shared core

| Current path | Target path |
| --- | --- |
| `rl/spec.py` | `rl/core/spec.py` |
| `rl/calibration.py` | `rl/core/calibration.py` |
| `rl/observation.py` | `rl/core/observation.py` |
| New configuration module | `rl/core/config.py` |

## Perception and estimation

| Current path | Target path |
| --- | --- |
| `rl/gatenet.py` | `rl/perception/gatenet.py` |
| `rl/pnp.py` | `rl/perception/pnp.py` |
| `rl/dataset.py` | `rl/perception/dataset.py` |
| `rl/vision_fusion.py` | `rl/perception/vision_fusion.py` |
| `rl/ekf.py` | `rl/estimation/ekf.py` |

## Environment and training

| Current path | Target path |
| --- | --- |
| `rl/sim_interface.py` | `rl/environment/sim_interface.py` |
| `rl/env.py` | `rl/environment/env.py` |
| `rl/train_ppo.py` | `rl/training/train_ppo.py` |
| `rl/train_bc.py` | `rl/training/train_bc.py` |
| `rl/log_demos.py` | `rl/training/log_demos.py` |

## Experts and live flight

| Current path | Target path |
| --- | --- |
| `rl/control.py` | `rl/experts/control.py` |
| `rl/gp_expert.py` | `rl/experts/gp_expert.py` |
| `rl/fly2.py` | `rl/experts/fly2.py` |
| `rl/fly2_course.py` | `rl/experts/fly2_course.py` |
| `rl/fly_geometric.py` | `rl/experts/fly_geometric.py` |
| `rl/fly_odom.py` | `rl/experts/fly_odom.py` |

## Operational scripts

| Current path | Target path |
| --- | --- |
| `rl/diag_mav.py` | `scripts/rl_diag_mav.py` |
| `rl/diag_project.py` | `scripts/rl_diag_project.py` |
| `rl/dynamics_id.py` | `scripts/rl_diag_dynamics.py` |
| `rl/calibrate_cam.py` | `scripts/rl_diag_calibrate_cam.py` |
| `rl/capture_gates.py` | `scripts/capture_gates.py` |

## Stable entry points and data

- `rl/deploy.py` remains at its current path.
- `rl/data/policy.pt`, `policy_gate1_backup.pt`, `gate_map.json`, `baseline.json`,
  `best/`, and `tb/` remain in place.
- `rl/data/policy_velwalk_failed.pt` is deleted during the atomic migration.
- No old-path shims or package re-exports are added.
