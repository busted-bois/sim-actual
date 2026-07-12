"""RL drone-racing pipeline (Modules 1-8).

Shared foundation lives in ``rl.spec``. Live-sim I/O reuses the working
``simulator`` package (pymavlink + UDP camera). Action space is
attitude-rate + thrust (the only channel this simulator actuates on).
"""
