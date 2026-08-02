"""Gate detection, pose recovery, datasets, and vision-fusion components.

Perception pipeline handoff:

    image --GateNet (U-Net)--> gate mask --PnP (IPPE_SQUARE)--> gate pose
                                                                |
                                      rl.estimation EKF <-- vision_fusion

* ``gatenet`` — U-Net segmenter; trained weights (``data/gatenet.pt``) are
  absent by default and user-supplied.
* ``pnp`` — 2D gate corners -> camera/body-frame pose via the fixed intrinsics.
* ``dataset`` — auto-labeled (image, mask) pair generation from sim telemetry.
* ``synthetic`` — deterministic offline QA scene (the only perception test path
  while GateNet weights are absent).
* ``vision_fusion`` — feeds GateNet+PnP pose estimates into the EKF; degrades
  gracefully (warn + coast) when weights are missing.
"""
