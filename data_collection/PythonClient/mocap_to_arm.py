#!/usr/bin/env python3
"""Map Motive rigid-body poses into the xArm base frame using a calibration.

solve_mocap_calibration.py fits, for every calibration sample,

    A_i * Y = X * S(B_i)

with A_i the TCP pose the arm reported, B_i the rigid-body pose from Motive
(position in mm, scaled by the fitted mocap scale S), X the Motive frame in the
arm base frame and Y the rigid body in the TCP frame.  During calibration the
controller's TCP offset T_cal (from <name>_config.json) was in effect, so the
*flange* pose is A_i * T_cal^-1.  Hence, for any rigid-body pose B,

    flange = X * S(B) * Y^-1 * T_cal^-1

is where the arm's flange must be so that the end effector (with its markers)
sits exactly where B says it is.  That is what the recorder logs and what the
rollout script tracks with IK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit("Needs scipy: python -m pip install scipy") from exc

import xarm6_kinematics as kin

DATA_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = DATA_DIR / "calibration"
RECORDINGS_DIR = DATA_DIR / "recordings"


@dataclass
class MocapCalibration:
    path: Path
    x: np.ndarray            # Motive frame in arm base frame (mm)
    y_inv: np.ndarray        # TCP frame in rigid-body frame
    scale: float             # mocap scale factor
    cal_tcp_inv: np.ndarray  # inverse of the TCP offset used while calibrating
    pos_rms_mm: float
    rot_rms_deg: float
    tool_length: Optional[float]  # mm, from the calibration run's config if known

    def flange_poses(self, pos_m: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        """(N,3) Motive positions in m and (N,4) NatNet quaternions -> (N,4,4)
        flange poses in the arm base frame (mm)."""
        pos_m = np.atleast_2d(pos_m)
        quat_xyzw = np.atleast_2d(quat_xyzw)
        b = np.tile(np.eye(4), (len(pos_m), 1, 1))
        b[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        b[:, :3, 3] = pos_m * 1000.0 * self.scale
        return self.x @ b @ self.y_inv @ self.cal_tcp_inv

    def flange_pose(self, pos_m, quat_xyzw) -> np.ndarray:
        return self.flange_poses(np.asarray(pos_m, float), np.asarray(quat_xyzw, float))[0]

    def describe(self) -> str:
        return (f"{self.path} (fit RMS {self.pos_rms_mm:.2f} mm / "
                f"{self.rot_rms_deg:.2f} deg, scale {self.scale:.4f})")


def latest_solution(root: Path = CALIBRATION_DIR) -> Optional[Path]:
    solutions = sorted(root.rglob("*_solution.json"), key=lambda p: p.stat().st_mtime)
    return solutions[-1] if solutions else None


def load_calibration(path: Optional[Path] = None) -> MocapCalibration:
    """Load a solve_mocap_calibration.py result (default: the newest one)."""
    if path is None:
        path = latest_solution()
        if path is None:
            raise SystemExit(f"No *_solution.json under {CALIBRATION_DIR}; run the "
                             "calibration and solve_mocap_calibration.py first")
    if path.is_dir():
        found = latest_solution(path)
        if found is None:
            raise SystemExit(f"No *_solution.json under {path}")
        path = found
    sol = json.loads(path.read_text())
    if sol.get("mode") != "full" or "Y" not in sol:
        raise SystemExit(f"{path} is a positions-only calibration (no marker->TCP "
                         "rotation); re-run the calibration with tilt/roll so the "
                         "orientation can be mapped")

    # The TCP offset in effect during calibration lives in the run's config.
    cfg_path = path.with_name(path.name.replace("_solution.json", "_config.json"))
    tcp_offset = [0.0] * 6
    tool_length = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tcp_offset = cfg.get("tcp_offset", tcp_offset)
        tool_length = cfg.get("tool_length")
    else:
        print(f"WARNING: {cfg_path.name} not found; assuming the calibration used a zero "
              "TCP offset")

    return MocapCalibration(
        path=path,
        x=np.array(sol["X"], float),
        y_inv=np.linalg.inv(np.array(sol["Y"], float)),
        scale=float(sol.get("scale", 1.0)),
        cal_tcp_inv=np.linalg.inv(kin.pose_to_matrix(tcp_offset)),
        pos_rms_mm=float(sol.get("pos_rms_mm", float("nan"))),
        rot_rms_deg=float(sol.get("rot_rms_deg", float("nan"))),
        tool_length=tool_length,
    )
