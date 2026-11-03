#!/usr/bin/env python3
"""Offline kinematics for the UFactory xArm 6 (no robot connection needed).

Forward kinematics come from the published modified-DH parameters; inverse
kinematics are a damped-least-squares solve warm-started from a nearby joint
configuration, so a path solved in small steps follows one continuous IK
branch the way the controller's own linear (Cartesian) moves do.

Units follow the xArm SDK: millimetres and degrees.  Poses are 4x4 numpy
transforms in the arm base frame; xArm roll/pitch/yaw are R = Rz(yaw) Ry(pitch)
Rx(roll).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

_T2 = math.degrees(math.atan2(284.5, 53.5))  # shoulder/elbow angle offset

# Modified DH rows: (alpha_{i-1} deg, a_{i-1} mm, d_i mm, theta offset deg).
XARM6_DH = [
    (0.0, 0.0, 267.0, 0.0),
    (-90.0, 0.0, 0.0, -_T2),
    (0.0, 289.48866, 0.0, _T2),
    (-90.0, 77.5, 342.5, 0.0),
    (90.0, 0.0, 0.0, 0.0),
    (-90.0, 76.0, 97.0, 0.0),
]

# Joint limits in degrees (xArm 6 user manual).
XARM6_JOINT_LIMITS = np.array([
    [-360.0, 360.0], [-118.0, 120.0], [-225.0, 11.0],
    [-360.0, 360.0], [-97.0, 180.0], [-360.0, 360.0],
])


# ------------------------------------------------------------------ rotations
def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector (rad) -> rotation matrix."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (rad), robust near 0 and pi."""
    cos_t = max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > math.pi - 1e-6:
        # Axis from the symmetric part; sign is arbitrary at exactly pi.
        b = (r + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(b), 0.0, None))
        i = int(np.argmax(axis))
        for j in range(3):
            if j != i and b[i, j] < 0:
                axis[j] = -axis[j]
        return theta * axis / np.linalg.norm(axis)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return theta * w / (2.0 * math.sin(theta))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(r: np.ndarray) -> Tuple[float, float, float]:
    """xArm roll/pitch/yaw (deg); at pitch = +-90 yaw is fixed to 0."""
    sp = max(-1.0, min(1.0, -r[2, 0]))
    pitch = math.asin(sp)
    if math.cos(pitch) > 1e-6:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        yaw = 0.0
        roll = math.atan2(math.copysign(1.0, sp) * r[0, 1], r[1, 1])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] (mm, deg) -> 4x4."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*pose[3:6])
    t[:3, 3] = pose[:3]
    return t


def matrix_to_pose(t: np.ndarray) -> List[float]:
    return [*map(float, t[:3, 3]), *matrix_to_rpy(t[:3, :3])]


def rotation_angle_deg(r: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(matrix_to_rotvec(r))))


def interpolate(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Linear position + slerp orientation between two 4x4 poses."""
    t = np.eye(4)
    t[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
    t[:3, :3] = a[:3, :3] @ rotvec_to_matrix(s * matrix_to_rotvec(a[:3, :3].T @ b[:3, :3]))
    return t


# ----------------------------------------------------------------- kinematics
def _mdh(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    al, th = math.radians(alpha), math.radians(theta)
    ca, sa, ct, st = math.cos(al), math.sin(al), math.cos(th), math.sin(th)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0, 0, 0, 1],
    ])


def forward_frames(joints: Sequence[float]) -> List[np.ndarray]:
    """Base frame followed by the frames of joints 1..6 (the last is the flange)."""
    frames = [np.eye(4)]
    for (alpha, a, d, offset), q in zip(XARM6_DH, joints):
        frames.append(frames[-1] @ _mdh(alpha, a, d, q + offset))
    return frames


def forward(joints: Sequence[float]) -> np.ndarray:
    """Flange pose (TCP with zero offset) for joint angles in degrees."""
    return forward_frames(joints)[-1]


def jacobian(joints: Sequence[float]) -> np.ndarray:
    """6x6 geometric Jacobian of the flange: rows [v (mm/rad); w (rad/rad)]."""
    frames = forward_frames(joints)
    p = frames[-1][:3, 3]
    jac = np.zeros((6, 6))
    for i in range(6):
        z, o = frames[i + 1][:3, 2], frames[i + 1][:3, 3]
        jac[:3, i] = np.cross(z, p - o)
        jac[3:, i] = z
    return jac


def inverse(target: np.ndarray, seed: Sequence[float], iterations: int = 100,
            tol_mm: float = 0.01, tol_deg: float = 0.005) -> Optional[np.ndarray]:
    """Flange IK near `seed` (deg); None if it does not converge."""
    q = np.array(seed, dtype=float)
    w = 200.0  # mm per rad, to balance position and orientation error
    for _ in range(iterations):
        cur = forward(q)
        dp = target[:3, 3] - cur[:3, 3]
        dr = matrix_to_rotvec(target[:3, :3] @ cur[:3, :3].T)
        if np.linalg.norm(dp) < tol_mm and math.degrees(np.linalg.norm(dr)) < tol_deg:
            return q
        jac = jacobian(q)
        jac[3:] *= w
        err = np.concatenate([dp, w * dr])
        lam = 1.0
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam ** 2 * np.eye(6), err)
        q = q + np.degrees(dq)
    return None


def within_limits(joints: Sequence[float], margin_deg: float = 0.0) -> bool:
    q = np.asarray(joints)
    return bool(np.all(q >= XARM6_JOINT_LIMITS[:, 0] + margin_deg)
                and np.all(q <= XARM6_JOINT_LIMITS[:, 1] - margin_deg))
