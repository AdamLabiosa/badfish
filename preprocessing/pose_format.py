"""State/action vector format used in the badfish LeRobot datasets.

A pose is the xArm *flange* pose in the arm base frame, stored as 9 floats:

    [x_mm, y_mm, z_mm, r00, r10, r20, r01, r11, r21]

i.e. position in mm followed by the first two columns of the rotation matrix
(the "6D" rotation representation of Zhou et al. 2019).  Unlike roll/pitch/yaw
it is continuous (no +-180 deg wraparound) and subtracting two of them gives a
sensible "relative action", which pi0.5 can use (--policy.use_relative_actions).

Use vec_to_pose() on the policy's output to get back a 4x4 matrix for IK /
xarm6_kinematics.matrix_to_pose().
"""

from __future__ import annotations

import numpy as np

POSE_NAMES = ["x_mm", "y_mm", "z_mm", "r00", "r10", "r20", "r01", "r11", "r21"]
POSE_DIM = len(POSE_NAMES)


def pose_to_vec(t: np.ndarray) -> np.ndarray:
    """(..., 4, 4) homogeneous poses (mm) -> (..., 9) vectors."""
    t = np.asarray(t, dtype=np.float64)
    pos = t[..., :3, 3]
    rot6 = np.concatenate([t[..., :3, 0], t[..., :3, 1]], axis=-1)
    return np.concatenate([pos, rot6], axis=-1)


def vec_to_pose(v: np.ndarray) -> np.ndarray:
    """(..., 9) vectors -> (..., 4, 4) poses.  The two rotation columns are
    re-orthonormalised (Gram-Schmidt), so raw network outputs are fine."""
    v = np.asarray(v, dtype=np.float64)
    a, b = v[..., 3:6], v[..., 6:9]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(c0 * b, axis=-1, keepdims=True) * c0
    c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c2 = np.cross(c0, c1)
    t = np.zeros(v.shape[:-1] + (4, 4))
    t[..., :3, 0], t[..., :3, 1], t[..., :3, 2] = c0, c1, c2
    t[..., :3, 3] = v[..., :3]
    t[..., 3, 3] = 1.0
    return t
