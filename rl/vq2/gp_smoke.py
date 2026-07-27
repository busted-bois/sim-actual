"""STAGE 1 verification for residual-RL: does the GP base fly through
`GPFlightInterface`?

Creates the interface (which runs the proven auto_gp stack in a background loop),
sends NO residual, so the GP pilot flies its own guidance untouched -- identical
to `make control-flight`, but routed through the class the RL env uses. Reports
whether it reaches FLYING and how many gates it clears.

PASS: it clears gates like control-flight  -> the residual-RL foundation is solid,
and we can layer the policy's correction on top in Stage 2.
FAIL: it doesn't fly / doesn't clear gates -> the interface itself breaks the base
flight (thread/override/lifecycle) and we fix that before anything else.

    make rl2-gp-smoke                 # start/restart the race when it says to
    make rl2-gp-smoke ARGS="--seconds 90"
"""

from __future__ import annotations

import argparse
import time

from rl.vq2.gp_flight import GPFlightInterface


def smoke(seconds: float = 60.0, residual_zero: bool = False):
    iface = GPFlightInterface()
    mode = ("ZERO residual added each tick (Stage 2: residual path must not change "
            "the flight)" if residual_zero else "NO residual (Stage 1: pure GP base)")
    print(f"[gp-smoke] GP base flying through GPFlightInterface -- {mode}. "
          "Start/Restart the race...", flush=True)
    t0 = time.time()
    max_gate = 0
    reached_flying = False
    last_log = 0.0
    try:
        while time.time() - t0 < seconds:
            if residual_zero:
                iface.send_residual_deg(0.0, 0.0, 0.0, 0.0)   # exercise residual path with 0
            gi = iface.data.get("active_gate_index")
            gi = int(gi) if gi is not None else 0
            if gi > max_gate:
                max_gate = gi
                print(f"[gp-smoke] +++ reached gate {max_gate} at {time.time()-t0:.1f}s",
                      flush=True)
            flying = iface.flying()
            if flying and not reached_flying:
                reached_flying = True
                print(f"[gp-smoke] FLYING at {time.time()-t0:.1f}s", flush=True)
            now = time.time()
            if now - last_log >= 3.0:
                last_log = now
                col = iface.data.get("collision")
                print(f"  t={now-t0:5.1f}s  flying={flying}  active_gate={gi}  "
                      f"max_gate={max_gate}  collision={'yes' if col else 'no'}",
                      flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[gp-smoke] stopped.", flush=True)
    finally:
        print(f"\n[gp-smoke] RESULT: reached_flying={reached_flying}  "
              f"MAX GATE CLEARED = {max_gate}", flush=True)
        print("[gp-smoke] PASS if max gate > 0 (base flies through the interface).",
              flush=True)
        iface.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--residual-zero", action="store_true",
                    help="send zero residual each tick (Stage 2: verify the "
                         "residual-add path flies identically to the pure base)")
    args = ap.parse_args()
    smoke(args.seconds, residual_zero=args.residual_zero)
