# Sim stack — research links

Saved reference for the 5 improvement areas (+ gate detection). Ask the agent to open this file when you want the list again.

---

## 1. Unified state estimation (EKF / factor graph / VIO)

One coherent drone + gate state instead of parallel VIO, gate KF, and pilot odometry.

| Resource | URL |
|----------|-----|
| OpenVINS docs | https://docs.openvins.com/ |
| VINS tutorial (EKF vs factor graphs) | https://udel.edu/~ghuang/icra21-vins-workshop/slides/01-vins_tutorial.pdf |
| UZH RPG visual–inertial fusion course | https://rpg.ifi.uzh.ch/docs/teaching/2024/13_visual_inertial_fusion.pdf |
| GTSAM factor-graph tutorial | https://dongjing3309.github.io/files/gtsam-tutorial.pdf |
| Robotics textbook — VIO + IMU preintegration | https://www.roboticsbook.org/S74_drone_perception.html |
| OpenVINS GitHub | https://github.com/rpng/open_vins |

---

## 2. Robust vision at gate close-range (PnP / corners / segmentation)

Handle large blobs, low quality, and bad PnP when the gate fills the frame.

| Resource | URL |
|----------|-----|
| OpenCV `solvePnP` docs | https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html |
| EPnP paper | https://dl.acm.org/doi/10.1007/s11263-008-0152-6 |
| EPro-PnP (probabilistic / robust PnP) | https://openaccess.thecvf.com/content/CVPR2022/html/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.html |
| GNC-PnP (outlier-robust pose) | https://arxiv.org/html/2512.06565v1 |
| Close-range fisheye PnP | https://link.springer.com/article/10.1007/s00138-025-01768-8 |
| GateNet (this repo) | `rl/gatenet.py` — `make train-gatenet` |

---

## 3. Sim / sensor characterization (IMU, camera, odometry)

Measure latency, noise, gravity sign, thrust response — not guess tuning.

| Resource | URL |
|----------|-----|
| OpenVINS calibration guide | https://docs.openvins.com/gs-calibration.html |
| Kalibr camera–IMU calibration | https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration |
| Visual–inertial calibration tutorial (video) | https://www.youtube.com/watch?v=BtzmsuJemgI |
| Camera–IMU calibration math | https://docs.altnautica.com/drone-agent/vision-nav-calibration-math |
| Allan variance (IMU noise) | https://github.com/ori-drs/allan_variance_ros |
| Dynamics ID (this repo) | `make dynamics` → `rl/dynamics_id` |

---

## 4. Perception-aware control (use estimates, not just nx/ny)

Controller driven by gate pose, quality, and VIO — not raw centroid chasing.

| Resource | URL |
|----------|-----|
| Autonomous drone race (TU Delft) | https://ar5iv.labs.arxiv.org/html/1809.05958 |
| Gate-to-gate visual navigation + IBVS | https://ar5iv.labs.arxiv.org/html/2503.05251 |
| Visual servoing overview | https://eureka.patsnap.com/report-enhance-visual-servoing-in-autonomous-drone-navigation |
| IBVS + obstacle avoidance (RGB) | https://arxiv.org/html/2509.17435 |
| IBVS for UAVs (fuzzy logic) | https://www.researchgate.net/publication/369984531_An_image-based_visual_servoing_control_method_for_UAVs_based_on_fuzzy_logic |
| AI Grand Prix | https://www.theaigrandprix.com/ |

---

## 5. Obstacle vs gate disambiguation

Stop treating the orange gate as an obstacle while gate is visible.

| Resource | URL |
|----------|-----|
| EDFNet (RGB + depth + edge fusion) | https://github.com/negarfathi/EDFNet |
| DDOS dataset | https://huggingface.co/datasets/benediktkol/DDOS |
| Semantic segmentation + depth for UAV | https://arxiv.org/html/2510.16624v1 |
| Thin obstacle detection (event cameras) | https://arxiv.org/html/2508.09397 |
| OpenCV background subtraction | https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html |

---

## Bonus: Gate detection methods (multi-method / fusion)

| Method | Notes |
|--------|--------|
| HSV + contours | Current — `simulator/gate_detector.py` |
| GateNet U-Net | `rl/gatenet.py`, `make train-gatenet` |
| OpenCV contours / Hough | https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html |
| Temporal corner tracking | https://docs.opencv.org/4.x/dc/d6b/group__video__track.html |

---

## Suggested reading order

1. [Autonomous drone race (TU Delft)](https://ar5iv.labs.arxiv.org/html/1809.05958)
2. [OpenVINS calibration](https://docs.openvins.com/gs-calibration.html)
3. [Gate-to-gate IBVS](https://ar5iv.labs.arxiv.org/html/2503.05251)
4. [VINS tutorial PDF](https://udel.edu/~ghuang/icra21-vins-workshop/slides/01-vins_tutorial.pdf)
5. [EDFNet / DDOS](https://github.com/negarfathi/EDFNet)
