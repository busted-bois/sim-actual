from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingSnapshot:
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
    status: str

    def as_dict(self):
        return {
            "sim_time_ns": self.sim_time_ns,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "status": self.status,
        }
