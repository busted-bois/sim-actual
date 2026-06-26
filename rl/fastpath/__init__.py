"""Fast raceline pilot, ported from the 24s AI-GP round-1 qualifier run.

Pure, sim-agnostic flight logic: a Catmull-Rom raceline through the
(hardcoded) gate map, an accel-limited speed profile, and a velocity ->
accel -> tilt -> attitude-rate control stack. The live-sim boundary lives
in ``rl.fly2``, which feeds these modules state from ``rl.sim_interface``.
"""
