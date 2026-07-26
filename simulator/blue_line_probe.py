"""Passive A/B probe for the blue-line estimators — no MAVLink, no arming.

Reads ONLY the camera stream (UDP 5600) and scores the inner-edge estimator
against the legacy centroid estimator on identical frames. Safe to run during
a live race; it never commands the drone.

    uv run -m simulator.blue_line_probe        (or: make bl-probe)

With the drone PARKED the true corridor offset is constant, so every
frame-to-frame change in cx is measurement noise — that jitter number is the
headline result. Annotated PNGs land in runs/bl_probe/ so the tracked inner
edges can be checked by eye against the ribbons.
"""

from __future__ import annotations

import os
import statistics as st
import time

import cv2

_SECONDS = float(os.environ.get("BL_PROBE_SECONDS", "20"))
_SAVE_EVERY = int(os.environ.get("BL_PROBE_SAVE_EVERY", "30"))
_OUT_DIR = os.path.join("runs", "bl_probe")
_RAW_DIR = os.path.join(_OUT_DIR, "raw")


def _jitter(vals: list[float]) -> float:
    """Mean |frame-to-frame| change — pure noise when the drone is parked."""
    if len(vals) < 2:
        return float("nan")
    return st.mean(abs(b - a) for a, b in zip(vals[:-1], vals[1:]))


def _summarize(edge: list[float], cent: list[float], hdg_e, hdg_c, rows, n, found):
    print("\n=== blue-line estimator A/B ===", flush=True)
    print(f"frames={n} found={found} ({100.0 * found / max(n, 1):.0f}%)", flush=True)
    if not edge:
        print("no corridor detected — point the drone down the track", flush=True)
        return
    print(f"edge rows tracked: mean={st.mean(rows):.1f} min={min(rows)}", flush=True)
    for name, vals in (
        ("cx  edge", edge),
        ("cx  cent", cent),
        ("hdg edge", hdg_e),
        ("hdg cent", hdg_c),
    ):
        print(
            f"{name}: mean={st.mean(vals):+.4f} sd={st.pstdev(vals):.4f} "
            f"jitter={_jitter(vals):.4f}",
            flush=True,
        )
    disagree = [abs(a - b) for a, b in zip(edge, cent)]
    print(
        f"|cx_edge - cx_cent|: mean={st.mean(disagree):.4f} max={max(disagree):.4f}",
        flush=True,
    )
    je, jc = _jitter(edge), _jitter(cent)
    if len(edge) >= 2 and jc > 1e-9:
        verdict = "QUIETER" if je < jc else "NOISIER"
        print(
            f"\n-> inner-edge cx is {verdict}: {je / jc:.2f}x centroid jitter",
            flush=True,
        )


def main():
    os.environ.setdefault("SKIP_YOLO", "1")  # probe only needs the HSV path
    from simulator.vision_rx import VisionRX

    data: dict = {}
    VisionRX(data)
    os.makedirs(_RAW_DIR, exist_ok=True)
    print(f"[bl-probe] listening {_SECONDS:.0f}s — frames -> {_OUT_DIR}/", flush=True)

    edge: list[float] = []
    cent: list[float] = []
    hdg_e: list[float] = []
    hdg_c: list[float] = []
    rows: list[int] = []
    n = found = saved = 0
    last_fid = None
    t_end = time.monotonic() + _SECONDS
    try:
        while time.monotonic() < t_end:
            # Key off blue_line's OWN frame_id: vision_rx swaps in data["frame"]
            # before it has attached the estimate or the annotation.
            bl = data.get("blue_line")
            if bl is None or bl["frame_id"] == last_fid:
                time.sleep(0.002)
                continue
            last_fid = bl["frame_id"]
            n += 1
            if bl.get("found"):
                found += 1
                edge.append(float(bl["cx_norm"]))
                cent.append(float(bl["cx_centroid"]))
                hdg_e.append(float(bl["heading_err"]))
                hdg_c.append(float(bl["heading_centroid"]))
                rows.append(int(bl["edge_rows"]))
            frame = data.get("frame")
            if (
                frame is not None
                and frame["frame_id"] == last_fid
                and n % _SAVE_EVERY == 0
            ):
                # Raw frame stays lossless PNG — it is the offline replay input,
                # so the estimator can be re-run on the exact same pixels. The
                # annotated copy is only for eyeballing, and PNG-encoding it here
                # blinds the poll loop for ~26 ms, so it goes out as JPEG.
                saved += 1
                cv2.imwrite(
                    os.path.join(_RAW_DIR, f"f{last_fid:06d}.png"), frame["img"]
                )
                ann = frame.get("annotated")
                if ann is not None:
                    cv2.imwrite(os.path.join(_OUT_DIR, f"f{last_fid:06d}.jpg"), ann)
            if n % 60 == 0:
                src = bl.get("source")
                print(
                    f"[bl-probe] n={n} src={src} "
                    f"cx_edge={bl.get('cx_norm', 0):+.3f} "
                    f"cx_cent={bl.get('cx_centroid', 0):+.3f} "
                    f"rows={bl.get('edge_rows')}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    _summarize(edge, cent, hdg_e, hdg_c, rows, n, found)
    print(f"saved {saved} annotated frames to {_OUT_DIR}/", flush=True)
    os._exit(0)  # hard-exit past the non-daemon VisionRX receiver thread


if __name__ == "__main__":
    main()
