from dataclasses import dataclass


@dataclass
class _StateSample:
    sim_time_ns: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    roll: float
    pitch: float
    yaw: float


class StateRingBuffer:
    def __init__(self, capacity=256):
        self._samples: list[_StateSample] = []
        self._capacity = capacity

    def push(self, sample: _StateSample):
        self._samples.append(sample)
        if len(self._samples) > self._capacity:
            self._samples.pop(0)

    def interpolate_at(self, sim_time_ns: int):
        if not self._samples:
            return None
        if sim_time_ns <= self._samples[0].sim_time_ns:
            return self._samples[0]
        if sim_time_ns >= self._samples[-1].sim_time_ns:
            return self._samples[-1]

        for idx in range(1, len(self._samples)):
            prev = self._samples[idx - 1]
            nxt = self._samples[idx]
            if prev.sim_time_ns <= sim_time_ns <= nxt.sim_time_ns:
                span = nxt.sim_time_ns - prev.sim_time_ns
                if span <= 0:
                    return prev
                alpha = (sim_time_ns - prev.sim_time_ns) / span
                return _StateSample(
                    sim_time_ns=sim_time_ns,
                    x=prev.x + alpha * (nxt.x - prev.x),
                    y=prev.y + alpha * (nxt.y - prev.y),
                    z=prev.z + alpha * (nxt.z - prev.z),
                    vx=prev.vx + alpha * (nxt.vx - prev.vx),
                    vy=prev.vy + alpha * (nxt.vy - prev.vy),
                    vz=prev.vz + alpha * (nxt.vz - prev.vz),
                    roll=prev.roll + alpha * (nxt.roll - prev.roll),
                    pitch=prev.pitch + alpha * (nxt.pitch - prev.pitch),
                    yaw=prev.yaw + alpha * (nxt.yaw - prev.yaw),
                )
        return self._samples[-1]
