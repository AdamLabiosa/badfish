#!/usr/bin/env python3
"""Solve the arm <-> mocap calibration and report how well they agree.

Input: a samples CSV from xarm_mocap_auto_calibration.py (or the manual
xarm_mocap_calibration.py).  Each sample pairs the arm's reported TCP pose A_i
(base frame) with the averaged rigid-body pose B_i (Motive frame).  With

    X = Motive frame in the arm base frame     (p_base = X * p_mocap)
    Y = rigid body (markers) in the TCP frame

every sample should satisfy  A_i * Y = X * B_i.  X, Y and a mocap scale factor
are found by least squares (closed-form start, then nonlinear refinement), and
the leftover error is reported.  That error is how well mocap agrees with the
arm once calibrated.  The report also says what you'd lose by *not* calibrating:
how far Motive's axes are from the arm's, and the error if you assumed they
were exactly aligned.

If every sample has the same tool orientation, Y can't be solved; the script
then fits only the position transform (rotation, translation, scale).

Collect data that only goes up from the ready pose, with plenty of rotation:

    python xarm_mocap_auto_calibration.py --tool-length 300 --min-z ready \\
        --z-range 0 200 --tilt 30 --roll 90 --per-point 3
    python solve_mocap_calibration.py              # latest samples CSV

Writes <name>_solution.json next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit("Needs scipy: python -m pip install scipy") from exc

import xarm6_kinematics as kin

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"
MIN_ROTATION_SPREAD = 5.0   # deg; below this the marker offset is unobservable
MM_PER_DEG = 1.0            # weight of rotation vs position residuals in the fit


# ------------------------------------------------------------------- loading
def resolve_csv(target: Optional[Path]) -> Path:
    target = target or CALIBRATION_DIR
    if target.is_dir():
        csvs = [p for p in target.rglob("*.csv") if not p.name.endswith("_stream.csv")]
        if not csvs:
            raise SystemExit(f"No samples CSV under {target}")
        return max(csvs, key=lambda p: p.stat().st_mtime)
    if target.name.endswith("_stream.csv"):
        return target.with_name(target.name[: -len("_stream.csv")] + ".csv")
    return target


def load(path: Path, max_std_mm: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    a_list, b_list, ids = [], [], []
    for r in rows:
        std = float(r["mocap_pos_std_m"]) * 1000
        if std > max_std_mm:
            print(f"  dropping sample {r['sample']}: mocap std {std:.2f} mm "
                  f"> {max_std_mm} (arm still moving?)")
            continue
        a = kin.pose_to_matrix([float(r[k]) for k in (
            "arm_x_mm", "arm_y_mm", "arm_z_mm", "arm_roll_deg", "arm_pitch_deg",
            "arm_yaw_deg")])
        b = np.eye(4)
        b[:3, :3] = Rotation.from_quat([float(r[k]) for k in (
            "mocap_qx", "mocap_qy", "mocap_qz", "mocap_qw")]).as_matrix()
        b[:3, 3] = [float(r[k]) * 1000 for k in ("mocap_x_m", "mocap_y_m", "mocap_z_m")]
        a_list.append(a)
        b_list.append(b)
        ids.append(r["sample"])
    return np.array(a_list), np.array(b_list), ids


# -------------------------------------------------------------------- maths
def to_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def angle_deg(r: np.ndarray) -> float:
    return kin.rotation_angle_deg(r)


def make_t(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def initial_guess(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form X, Y for A_i Y = X B_i (Shah 2013): rotations from the null
    space of the Kronecker system, then translations by linear least squares."""
    eye = np.eye(3)
    m = np.vstack([np.hstack([np.kron(eye, ai[:3, :3]), -np.kron(bi[:3, :3].T, eye)])
                   for ai, bi in zip(a, b)])
    v = np.linalg.svd(m)[2][-1]
    ry = v[:9].reshape(3, 3, order="F")
    rx = v[9:].reshape(3, 3, order="F")
    det = np.linalg.det(ry)
    scale = np.sign(det) * abs(det) ** (-1 / 3)
    ry, rx = to_so3(ry * scale), to_so3(rx * scale)
    # R_A t_Y - t_X = R_X t_B - t_A
    lhs = np.vstack([np.hstack([ai[:3, :3], -eye]) for ai in a])
    rhs = np.concatenate([rx @ bi[:3, 3] - ai[:3, 3] for ai, bi in zip(a, b)])
    t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return make_t(rx, t[3:]), make_t(ry, t[:3])


def residuals(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray,
              scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample position error (mm, base frame) and rotation error (deg)."""
    pos, rot = [], []
    for ai, bi in zip(a, b):
        lhs = ai @ y
        bs = bi.copy()
        bs[:3, 3] *= scale
        rhs = x @ bs
        pos.append(lhs[:3, 3] - rhs[:3, 3])
        rot.append(np.degrees(Rotation.from_matrix(
            rhs[:3, :3].T @ lhs[:3, :3]).as_rotvec()))
    return np.array(pos), np.array(rot)


def refine(a, b, x0, y0, fit_scale=True, fixed_rx: Optional[np.ndarray] = None):
    """Nonlinear least squares over X, Y (and scale).  With fixed_rx, X's
    rotation is held fixed and only the translations / Y are fitted."""
    def unpack(p):
        i = 0
        if fixed_rx is None:
            rx = Rotation.from_rotvec(p[0:3]).as_matrix()
            i = 3
        else:
            rx = fixed_rx
        tx = p[i:i + 3]
        ry = Rotation.from_rotvec(p[i + 3:i + 6]).as_matrix()
        ty = p[i + 6:i + 9]
        s = math.exp(p[i + 9]) if fit_scale else 1.0
        return make_t(rx, tx), make_t(ry, ty), s

    def fun(p):
        x, y, s = unpack(p)
        pos, rot = residuals(a, b, x, y, s)
        return np.concatenate([pos.ravel(), rot.ravel() * MM_PER_DEG])

    p0 = []
    if fixed_rx is None:
        p0 += list(Rotation.from_matrix(x0[:3, :3]).as_rotvec())
    p0 += list(x0[:3, 3]) + list(Rotation.from_matrix(y0[:3, :3]).as_rotvec()) + list(y0[:3, 3])
    if fit_scale:
        p0 += [0.0]
    sol = least_squares(fun, np.array(p0), x_scale="jac")
    return unpack(sol.x)


def nearest_axis_mapping(r: np.ndarray) -> np.ndarray:
    """Signed permutation (proper rotation) closest to r."""
    best, best_trace = None, -math.inf
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            p = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                p[row, col] = s
            if np.linalg.det(p) < 0:
                continue
            tr = np.trace(p.T @ r)
            if tr > best_trace:
                best, best_trace = p, tr
    return best


def describe_mapping(p: np.ndarray) -> str:
    parts = []
    for col, name in enumerate("xyz"):
        row = int(np.argmax(np.abs(p[:, col])))
        parts.append(f"Motive +{name} -> arm {'+' if p[row, col] > 0 else '-'}{'XYZ'[row]}")
    return ", ".join(parts)


def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(cd.T @ cs / len(src))
    sgn = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sgn[2, 2] = -1
    r = u @ sgn @ vt
    s = np.trace(np.diag(d) @ sgn) / cs.var(0).sum()
    return r, mu_d - s * r @ mu_s, s


def stats(e: np.ndarray) -> str:
    n = np.linalg.norm(e, axis=1)
    return f"RMS {math.sqrt((n ** 2).mean()):.2f}, median {np.median(n):.2f}, max {n.max():.2f}"


def distance_check(a: np.ndarray, b: np.ndarray) -> Optional[str]:
    """Frame-free: for sample pairs with the same tool orientation, the distance
    between them must be the same in mocap and in the arm."""
    errs, dists = [], []
    for i, j in itertools.combinations(range(len(a)), 2):
        if angle_deg(a[i][:3, :3].T @ a[j][:3, :3]) > 0.2:
            continue
        da = np.linalg.norm(a[i][:3, 3] - a[j][:3, 3])
        if da < 20:
            continue
        db = np.linalg.norm(b[i][:3, 3] - b[j][:3, 3])
        errs.append(db - da)
        dists.append(da)
    if not errs:
        return None
    errs, dists = np.array(errs), np.array(dists)
    ratio = errs.sum() / dists.sum()
    return (f"{len(errs)} same-orientation pairs ({dists.min():.0f}-{dists.max():.0f} mm "
            f"apart): mocap distance - arm distance = {errs.mean():+.2f} mm mean, "
            f"{np.abs(errs).max():.2f} mm worst ({ratio * 100:+.2f}% overall)")


# --------------------------------------------------------------------- main
def solve(path: Path, max_std_mm: float) -> dict:
    print(f"{path}")
    a, b, ids = load(path, max_std_mm)
    n = len(a)
    if n < 3:
        raise SystemExit(f"Need at least 3 samples, have {n}")
    spread = max(angle_deg(a[0][:3, :3].T @ ai[:3, :3]) for ai in a)
    extent = np.ptp(a[:, :3, 3], axis=0)
    print(f"  {n} samples, arm positions span {extent[0]:.0f} x {extent[1]:.0f} x "
          f"{extent[2]:.0f} mm, tool orientations span {spread:.0f} deg\n")
    result: dict = {"source": str(path), "samples": n}

    check = distance_check(a, b)
    if check:
        print("Frame-free distance check (no calibration involved):\n  " + check + "\n")
        result["distance_check"] = check

    if spread < MIN_ROTATION_SPREAD:
        print(f"Tool orientation varies < {MIN_ROTATION_SPREAD:g} deg: fitting positions "
              "only (the marker offset can't be separated from the frame offset).")
        if n < 4 or np.linalg.matrix_rank(a[:, :3, 3] - a[:, :3, 3].mean(0), tol=5) < 3:
            print("  Positions don't span 3D; the frame rotation is under-determined. "
                  "Collect points spread in x, y and z.")
        r, t, s = umeyama(b[:, :3, 3], a[:, :3, 3])
        err = (s * (r @ b[:, :3, 3].T)).T + t - a[:, :3, 3]
        x = make_t(r, t)
        print(f"  position error after fit (mm): {stats(err)}")
        result.update(mode="positions only", X=x.tolist(), scale=s,
                      pos_rms_mm=float(np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean())))
        y = None
    else:
        x0, y0 = initial_guess(a, b)
        x, y, s = refine(a, b, x0, y0)
        pos, rot = residuals(a, b, x, y, s)
        print("Calibrated fit (A_i Y = X B_i, with mocap scale):")
        print(f"  position error (mm):  {stats(pos)}")
        print(f"  rotation error (deg): {stats(rot)}")
        worst = np.argsort(-np.linalg.norm(pos, axis=1))[:3]
        print("  worst samples: " + ", ".join(
            f"#{ids[i]} {np.linalg.norm(pos[i]):.2f} mm / {np.linalg.norm(rot[i]):.2f} deg"
            for i in worst))
        result.update(mode="full", X=x.tolist(), Y=y.tolist(), scale=s,
                      pos_rms_mm=float(np.sqrt((np.linalg.norm(pos, axis=1) ** 2).mean())),
                      rot_rms_deg=float(np.sqrt((np.linalg.norm(rot, axis=1) ** 2).mean())))

    r = x[:3, :3]
    p = nearest_axis_mapping(r)
    misalign = angle_deg(p.T @ r)
    print(f"\nMotive frame vs arm base frame:")
    print(f"  axes: {describe_mapping(p)}")
    print(f"  misalignment from that exact mapping: {misalign:.2f} deg "
          f"(= {math.tan(math.radians(misalign)) * 100:.1f} mm of error per 100 mm moved)")
    print(f"  Motive origin in arm base frame: "
          f"[{x[0, 3]:.1f}, {x[1, 3]:.1f}, {x[2, 3]:.1f}] mm"
          + (" (includes the marker offset)" if y is None else ""))
    print(f"  mocap scale: {s:.5f} ({(s - 1) * 100:+.2f}%, "
          f"{abs(s - 1) * 300:.2f} mm over 300 mm)")
    if y is not None:
        print(f"\nRigid body (markers) relative to the TCP: offset "
              f"[{y[0, 3]:.1f}, {y[1, 3]:.1f}, {y[2, 3]:.1f}] mm "
              f"(|{np.linalg.norm(y[:3, 3]):.1f}| mm), rotated "
              f"{angle_deg(y[:3, :3]):.1f} deg")
        # What if we skipped calibrating the frame rotation and scale?
        xa, ya, _ = refine(a, b, make_t(p, x[:3, 3]), y, fit_scale=False, fixed_rx=p)
        pos_a, rot_a = residuals(a, b, xa, ya, 1.0)
        print(f"\nIf you assume the axes are exactly aligned and the scale is 1 "
              f"(fitting only offsets):")
        print(f"  position error (mm):  {stats(pos_a)}")
        print(f"  rotation error (deg): {stats(rot_a)}")
        result["aligned_axes_pos_rms_mm"] = float(
            np.sqrt((np.linalg.norm(pos_a, axis=1) ** 2).mean()))
    result.update(axis_mapping=p.tolist(), axis_misalignment_deg=misalign)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", type=Path,
                        help="samples CSV (or its _stream.csv, or a directory: latest)")
    parser.add_argument("--max-std", type=float, default=0.5,
                        help="drop samples whose mocap position std exceeds this, mm "
                             "(default: %(default)s)")
    args = parser.parse_args()
    path = resolve_csv(args.target)
    result = solve(path, args.max_std)
    out = path.with_name(path.stem + "_solution.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
