#!/usr/bin/env python3
"""Replay a recorded end-effector demonstration on the xArm 6.

Input: a recording from record_mocap_camera.py (the rigid body of the
hand-held end effector).  With the arm<->mocap calibration
(solve_mocap_calibration.py) every mocap pose becomes the flange pose that puts
the mounted end effector exactly where it was held (see mocap_to_arm.py).

Planning (offline, identical for --dry-run and a real run):
  1. Drop untracked frames, trim to --trim if given, then trim still time at
     the start and end (the demo should start and end at rest).
  2. Resample to --rate Hz and low-pass the pose (--smooth) to remove mocap
     jitter, which servo control would otherwise turn into vibration.
  3. Solve IK for every sample, warm-started from the previous one so the
     arm follows one continuous configuration.  Every sample is checked with
     the same safety model as the auto calibration: joint limits, table and
     tool-to-arm clearance, arm folding onto its base, J4/J6 cable wrap and
     wrist singularity.  Any failure aborts with the time it happens at.
  4. Time-scale: --speed-scale (default 0.5 = half the demo speed), slowed
     further if needed so joint speed/acceleration and flange/tool-tip speed
     stay under --max-joint-speed / --max-joint-acc / --max-speed.
  The plan is written to <recording>_plan.csv.

Execution:
  1. Joint move (position mode, --approach-speed) to the plan's first pose,
     after you press Enter.
  2. Press Enter again: the joint trajectory is streamed in servo mode
     (set_servo_angle_j) at --rate Hz.  With --control cartesian the planned
     end-effector poses are streamed instead (set_servo_cartesian) and the
     arm's built-in IK picks the joints; the offline checks above still apply,
     but near a singularity the controller may move differently or stop.
  3. The arm stops at the demo's final pose and returns to position mode.
  Commanded/reported joints go to <recording>_rollout_<time>.csv.  With
  --mocap, the end effector is tracked during the replay too, and the
  script reports how far it was from the demonstration.

E-STOP: SPACE (or Ctrl+C) at any time once the arm is connected.  This is a
software stop over the network -- keep the hardware e-stop within reach.

    python rollout_trajectory.py --dry-run --tool-length 300        # newest demo
    python rollout_trajectory.py ../recordings/11-05-2026/demo_x.csv --tool-length 300
    python rollout_trajectory.py demo.csv --tool-length 300 --mocap --speed-scale 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from scipy.spatial.transform import Rotation, Slerp
except ImportError as exc:
    raise SystemExit("Needs scipy: python -m pip install scipy") from exc

import xarm6_kinematics as kin
from mocap_to_arm import RECORDINGS_DIR, MocapCalibration, load_calibration
from xarm_mocap_auto_calibration import (
    CYLINDER_TOOL_MODEL, DEFAULT_READY_JOINTS, EStopped, KeyMonitor, Safety,
    arm_to_base, check_joint_move, check_joints, clearances,
)

MAX_JOINT_SPEED = 180.0   # deg/s, xArm 6 spec
MAX_IK_STEP_DEG = 10.0    # per sample at the planning rate; more is an IK branch flip
RAMP_S = 0.5               # s; how gradually the replay speeds up/slows down
BOUNDARY_SPEED_WARN = 5.0  # deg/s; the servo stream starts and ends from rest


class PlanError(Exception):
    pass


# ------------------------------------------------------------------ loading
def resolve_recording(target: Optional[Path]) -> Path:
    target = target or RECORDINGS_DIR
    if target.is_dir():
        csvs = [p for p in target.rglob("*.csv")
                if not p.name.endswith("_plan.csv") and "_rollout_" not in p.name]
        if not csvs:
            raise SystemExit(f"No recording CSV under {target}")
        return max(csvs, key=lambda p: p.stat().st_mtime)
    if target.suffix != ".csv":  # the .mp4 or _meta.json of a recording
        stem = target.stem[:-len("_meta")] if target.stem.endswith("_meta") else target.stem
        target = target.with_name(stem + ".csv")
    if not target.exists():
        raise SystemExit(f"{target} not found")
    return target


def is_true(value: str) -> bool:
    return value.strip().lower() in ("1", "true")


@dataclass
class Demo:
    path: Path
    t: np.ndarray      # s, recording time
    pos: np.ndarray    # (N,3) Motive, m
    quat: np.ndarray   # (N,4) NatNet x, y, z, w
    dropped: int       # untracked frames


def load_demo(path: Path) -> Demo:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty")
    # Recorder CSVs use capture_time_s; calibration _stream.csv files use time_s.
    time_key = "capture_time_s" if "capture_time_s" in rows[0] else "time_s"
    t, pos, quat, dropped = [], [], [], 0
    for r in rows:
        if not is_true(r.get("tracking_valid", "1")) or not r.get("x_m", "").strip():
            dropped += 1
            continue
        t.append(float(r[time_key]))
        pos.append([float(r[k]) for k in ("x_m", "y_m", "z_m")])
        quat.append([float(r[k]) for k in ("qx", "qy", "qz", "qw")])
    if len(t) < 10:
        raise SystemExit(f"{path}: only {len(t)} tracked frames")
    t = np.array(t)
    order = np.argsort(t, kind="stable")
    t, pos, quat = t[order], np.array(pos)[order], np.array(quat)[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])  # Slerp needs increasing times
    return Demo(path, t[keep], pos[keep], quat[keep], dropped)


def calibration_for(recording: Path, override: Optional[Path]) -> MocapCalibration:
    if override is not None:
        return load_calibration(override)
    meta = recording.with_name(recording.stem + "_meta.json")
    if meta.exists():
        cal = json.loads(meta.read_text()).get("calibration")
        if cal and Path(cal).exists():
            return load_calibration(Path(cal))
        if cal:
            print(f"WARNING: calibration {cal} from {meta.name} is missing; using the newest")
    else:
        print("WARNING: recording has no _meta.json naming its calibration; using the "
              "newest.  Motive must not have been re-grounded in between.")
    return load_calibration(None)


# ----------------------------------------------------------------- planning
@dataclass
class Plan:
    rate: float
    slowdown: float          # execution time / demo time, overall
    base_pct: float          # % of demo speed where no limit applies
    slowest_pct: float       # % of demo speed at the most limited moment
    slow_time: float         # s of execution spent slowed below base_pct
    limited_by: str
    src_t: np.ndarray        # demo time of each execution sample
    joints: np.ndarray       # (M,6) deg, one per 1/rate s of execution
    window: Tuple[float, float]
    peak_joint_speed: np.ndarray   # deg/s per joint, at execution speed
    peak_joint_acc: float
    peak_lin_speed: float          # mm/s, flange or tool tip, at execution speed
    min_table: float
    min_self: float
    extent: np.ndarray             # (3,2) flange xyz range, mm
    boundary_speed: Tuple[float, float]  # max joint speed at the first/last sample, deg/s

    @property
    def duration(self) -> float:
        return (len(self.joints) - 1) / self.rate


def motion_window(demo: Demo, still_mm: float, still_deg: float,
                  pad_s: float) -> Tuple[float, float]:
    """First/last time the body is away from its start/end pose, padded."""
    rots = Rotation.from_quat(demo.quat)

    def first_move(idx: np.ndarray) -> Optional[int]:
        ref_p, ref_r = demo.pos[idx[0]], rots[idx[0]]
        d = np.linalg.norm(demo.pos[idx] - ref_p, axis=1) * 1000
        a = np.degrees((ref_r.inv() * rots[idx]).magnitude())
        moved = np.nonzero((d > still_mm) | (a > still_deg))[0]
        return None if len(moved) == 0 else int(idx[moved[0]])

    n = len(demo.t)
    i0 = first_move(np.arange(n))
    i1 = first_move(np.arange(n)[::-1])
    if i0 is None or i1 is None:
        raise PlanError(f"the end effector never moves more than {still_mm:g} mm / "
                        f"{still_deg:g} deg")
    return (max(demo.t[0], demo.t[i0] - pad_s), min(demo.t[-1], demo.t[i1] + pad_s))


def resample(demo: Demo, t0: float, t1: float, rate: float,
             smooth_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tg = np.arange(t0, t1 + 1e-9, 1.0 / rate)
    pos = np.column_stack([np.interp(tg, demo.t, demo.pos[:, i]) for i in range(3)])
    quat = Slerp(demo.t, Rotation.from_quat(demo.quat))(tg).as_quat()
    if smooth_s > 0:
        # Make quaternions continuous (q and -q are the same rotation) before filtering.
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        sigma = smooth_s * rate
        pos = gaussian_filter1d(pos, sigma, axis=0, mode="nearest")
        quat = gaussian_filter1d(quat, sigma, axis=0, mode="nearest")
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return tg, pos, quat


def singular(q: np.ndarray, margin_deg: float) -> bool:
    return abs(math.sin(math.radians(q[4]))) < math.sin(math.radians(margin_deg))


def solve_ik(targets: np.ndarray, times: np.ndarray, safety: Safety,
             singular_margin: float) -> np.ndarray:
    def fail(i: int, why: str) -> PlanError:
        x, y, z, roll, pitch, yaw = kin.matrix_to_pose(targets[i])
        return PlanError(f"t={times[i]:.2f} s: {why} (flange xyz=[{x:.0f} {y:.0f} {z:.0f}] "
                         f"mm rpy=[{roll:.0f} {pitch:.0f} {yaw:.0f}])")

    ready = np.array(DEFAULT_READY_JOINTS, float)
    first = targets[0]
    toward = ready.copy()
    toward[0] = math.degrees(math.atan2(first[1, 3], first[0, 3]))
    best, best_reason = None, "unreachable"
    for seed in (ready, toward):
        q = kin.inverse(first, seed, iterations=400)
        if q is None:
            continue
        reason = check_joints(q, safety) or (
            "near wrist singularity" if singular(q, singular_margin) else None)
        if reason:
            best_reason = reason
            continue
        if best is None or np.max(np.abs(q - ready)) < np.max(np.abs(best - ready)):
            best = q
    if best is None:
        raise fail(0, f"start pose {best_reason}")

    joints = [best]
    for i in range(1, len(targets)):
        q = kin.inverse(targets[i], joints[-1], iterations=100)
        if q is None:
            q = kin.inverse(targets[i], joints[-1], iterations=500)
        if q is None:
            raise fail(i, "unreachable (IK failed)")
        jump = float(np.max(np.abs(q - joints[-1])))
        if jump > MAX_IK_STEP_DEG:
            raise fail(i, f"IK jumped {jump:.0f} deg in one step (singularity or "
                          "configuration flip)")
        if singular(q, singular_margin):
            raise fail(i, f"near wrist singularity (J5={q[4]:.0f} deg)")
        reason = check_joints(q, safety)
        if reason:
            raise fail(i, reason)
        joints.append(q)
    return np.array(joints)


def tool_points(joints: np.ndarray, tool_length: float) -> Tuple[np.ndarray, np.ndarray]:
    flanges = np.array([kin.forward(q) for q in joints])
    origin = flanges[:, :3, 3]
    return origin, origin + tool_length * flanges[:, :3, 2]


def build_plan(demo: Demo, calibration: MocapCalibration, args: argparse.Namespace,
               safety: Safety) -> Plan:
    t0, t1 = demo.t[0], demo.t[-1]
    if args.trim:
        t0, t1 = max(t0, args.trim[0]), min(t1, args.trim[1])
    inside = (demo.t >= t0) & (demo.t <= t1)
    if inside.sum() < 10:
        raise PlanError(f"--trim {args.trim} leaves no data (recording is "
                        f"{demo.t[0]:.1f}-{demo.t[-1]:.1f} s)")
    demo = replace(demo, t=demo.t[inside], pos=demo.pos[inside], quat=demo.quat[inside])
    if not args.no_auto_trim:
        t0, t1 = motion_window(demo, args.still_mm, args.still_deg, args.still_pad)

    gaps = np.diff(demo.t)
    gap_idx = np.nonzero((gaps > args.max_gap) & (demo.t[1:] > t0) & (demo.t[:-1] < t1))[0]
    if len(gap_idx):
        worst = gap_idx[np.argmax(gaps[gap_idx])]
        raise PlanError(f"tracking gap of {gaps[worst]:.2f} s at t={demo.t[worst]:.2f} s "
                        f"(--max-gap {args.max_gap:g}); --trim around it or raise --max-gap")

    dt = 1.0 / args.rate
    tg, pos, quat = resample(demo, t0, t1, args.rate, args.smooth)
    targets = calibration.flange_poses(pos, quat)
    targets[:, :3, 3] += np.asarray(args.offset)
    joints = solve_ik(targets, tg, safety, args.singular_margin)

    src_t, exec_joints, warp = time_warp(tg, joints, safety.tool_length, args)
    flange = tool_points(joints, safety.tool_length)[0]

    min_table = min_self = math.inf
    for q in joints[:: max(1, int(args.rate // 20))]:
        table, self_clear = clearances(q, safety)
        min_table, min_self = min(min_table, table), min(min_self, self_clear)

    return Plan(rate=args.rate, src_t=src_t, joints=exec_joints, window=(t0, t1),
                min_table=min_table, min_self=min_self,
                extent=np.column_stack([flange.min(0), flange.max(0)]), **warp)


def time_warp(tg: np.ndarray, joints: np.ndarray, tool_length: float,
              args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Choose the execution timing.  Each moment of the demo is replayed at
    --speed-scale, or slower where a speed/acceleration limit needs it; the
    speed changes gradually (over ~RAMP_S) around those slow stretches.

    Returns the demo time of each execution sample (1/rate apart), the joint
    angles there, and summary numbers for the Plan."""
    dt = tg[1] - tg[0]
    spline = CubicSpline(tg, joints, axis=0)
    flange, tip = tool_points(joints, tool_length)
    lin = np.maximum(np.linalg.norm(np.gradient(flange, dt, axis=0), axis=1),
                     np.linalg.norm(np.gradient(tip, dt, axis=0), axis=1))
    qd = np.gradient(joints, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    # Slowdown each limit needs at each moment (slowing by k divides speeds by
    # k and accelerations by k^2).
    need = {
        "joint speed": np.max(np.abs(qd), axis=1) / args.max_joint_speed,
        "joint acceleration": np.sqrt(np.max(np.abs(qdd), axis=1) / args.max_joint_acc),
        "flange/tool-tip speed": lin / args.max_speed,
    }
    base = 1.0 / args.speed_scale
    k_need = np.maximum.reduce([np.full(len(tg), base), *need.values()])
    win = max(1, int(RAMP_S / dt))
    k = maximum_filter1d(k_need, size=2 * win + 1, mode="nearest")
    k = np.maximum(gaussian_filter1d(k, win / 2, mode="nearest"), k_need)

    # Changing speed adds acceleration of its own; check the actual execution
    # trajectory and slow down further until every limit holds.
    exec_dt = 1.0 / args.rate
    for _ in range(20):
        tau = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2 * dt)])
        tau_grid = np.arange(0.0, tau[-1], exec_dt)
        tau_grid = np.append(tau_grid, tau[-1]) if tau[-1] - tau_grid[-1] > 1e-9 else tau_grid
        src_t = np.interp(tau_grid, tau, tg)
        q = spline(src_t)
        qv = np.gradient(q, exec_dt, axis=0)
        v, a = np.abs(qv), np.abs(np.gradient(qv, exec_dt, axis=0))
        lin_exec = np.interp(src_t, tg, lin) / np.interp(src_t, tg, k)
        ratio = max(v.max() / args.max_joint_speed, math.sqrt(a.max() / args.max_joint_acc),
                    lin_exec.max() / args.max_speed)
        if ratio <= 1.001:
            break
        k *= min(ratio, 1.5)

    worst = int(np.argmax(k_need))
    limited_by = ("--speed-scale" if k_need[worst] <= base + 1e-9
                  else max(need, key=lambda name: need[name][worst]))
    k_exec = np.interp(src_t, tg, k)
    return src_t, q, dict(
        slowdown=float(tau[-1] / (tg[-1] - tg[0])),
        base_pct=100.0 / float(np.min(k)),
        slowest_pct=100.0 / float(np.max(k)),
        slow_time=float(np.sum(k_exec > 1.05 * np.min(k)) * exec_dt),
        limited_by=limited_by,
        peak_joint_speed=v.max(axis=0),
        peak_joint_acc=float(a.max()),
        peak_lin_speed=float(lin_exec.max()),
        boundary_speed=(float(v[0].max()), float(v[-1].max())),
    )


def print_plan(demo: Demo, plan: Plan, safety: Safety) -> None:
    t0, t1 = plan.window
    print(f"\nDemo: {demo.t[-1] - demo.t[0]:.1f} s recorded, {demo.dropped} untracked frames "
          f"dropped; replaying {t0:.2f}-{t1:.2f} s ({t1 - t0:.1f} s of motion)")
    ex = plan.extent
    print(f"Flange workspace: x {ex[0, 0]:.0f}..{ex[0, 1]:.0f}, y {ex[1, 0]:.0f}..{ex[1, 1]:.0f}"
          f", z {ex[2, 0]:.0f}..{ex[2, 1]:.0f} mm (arm base frame)")
    print("Start joints: [" + ", ".join(f"{v:.1f}" for v in plan.joints[0]) + "] deg")
    print("End joints:   [" + ", ".join(f"{v:.1f}" for v in plan.joints[-1]) + "] deg")
    print(f"Execution: {plan.duration:.1f} s ({len(plan.joints)} servo commands at "
          f"{plan.rate:g} Hz), {plan.base_pct:.0f}% of demo speed", end="")
    if plan.slow_time > 0:
        print(f"; slows to {plan.slowest_pct:.0f}% for {plan.slow_time:.1f} s where "
              f"limited by {plan.limited_by}")
    else:
        print()
    print("Peak joint speed (deg/s): [" + ", ".join(
        f"{v:.0f}" for v in plan.peak_joint_speed) + f"], peak joint accel "
        f"{plan.peak_joint_acc:.0f} deg/s^2")
    print(f"Peak flange/tool-tip speed {plan.peak_lin_speed:.0f} mm/s")
    self_text = (f"{plan.min_self:.0f} mm" if math.isfinite(plan.min_self) else "n/a")
    table_req = ("check off" if not math.isfinite(safety.table_margin)
                 else f"required {safety.table_margin:.0f}")
    print(f"Closest approach: table {plan.min_table:.0f} mm ({table_req}), tool-to-arm {self_text} (required "
          f"{safety.self_margin:.0f})")
    for name, v in zip(("starts", "ends"), plan.boundary_speed):
        if v > BOUNDARY_SPEED_WARN:
            print(f"WARNING: the replay {name} with a joint moving at {v:.0f} deg/s, so the "
                  "arm will jerk.  Hold the end effector still at the start and end of "
                  "the demo (or --trim to a still moment).")


def save_plan(path: Path, plan: Plan) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["exec_time_s", "demo_time_s",
                         "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
                         "flange_x_mm", "flange_y_mm", "flange_z_mm",
                         "flange_roll_deg", "flange_pitch_deg", "flange_yaw_deg"])
        for i, (src, q) in enumerate(zip(plan.src_t, plan.joints)):
            pose = kin.matrix_to_pose(kin.forward(q))
            writer.writerow([round(i / plan.rate, 4), round(float(src), 4),
                             *np.round(q, 4), *np.round(pose, 3)])


# ---------------------------------------------------------------- execution
class MocapLog:
    """Records the end effector's rigid body during the replay."""

    def __init__(self, args: argparse.Namespace) -> None:
        from NatNetClient import NatNetClient
        from xarm_mocap_calibration import resolve_client_ip
        self.rigid_id = args.rigid_id
        self.frames: List[tuple] = []
        self.lock = threading.Lock()
        self.recording = False
        client_ip = args.client_ip or resolve_client_ip(args.server_ip)
        print(f"Mocap: Motive at {args.server_ip}, listening on {client_ip}, rigid body "
              f"{args.rigid_id}")
        self.client = NatNetClient()
        self.client.set_client_address(client_ip)
        self.client.set_server_address(args.server_ip)
        self.client.set_use_multicast(not args.unicast)
        self.client.set_print_level(0)
        self.client.new_frame_with_data_listener = self.on_frame
        self.last_seen = 0.0

    def on_frame(self, data: dict) -> None:
        bodies = data["mocap_data"].rigid_body_data.rigid_body_list
        body = next((rb for rb in bodies if rb.id_num == self.rigid_id), None)
        if body is None or not body.tracking_valid:
            return
        now = time.perf_counter()
        self.last_seen = now
        if self.recording:
            with self.lock:
                self.frames.append((now, tuple(body.pos), tuple(body.rot)))

    def start(self) -> bool:
        if not self.client.run("d"):
            print("ERROR: could not start the NatNet client", file=sys.stderr)
            return False
        time.sleep(1.0)
        if not self.client.connected() or time.perf_counter() - self.last_seen > 0.5:
            print(f"ERROR: no tracked frames for rigid body {self.rigid_id}", file=sys.stderr)
            self.client.shutdown()
            return False
        return True

    def shutdown(self) -> None:
        self.client.shutdown()


def setup_collision(arm, safety: Safety, skip: bool) -> bool:
    if skip:
        print("WARNING: controller self-collision setup skipped (--no-controller-collision)")
        return True
    sensitivity = getattr(arm, "collision_sensitivity", None)
    if sensitivity == 0:
        print("ERROR: the arm's collision detection is OFF (sensitivity 0).  Set it to 1-5 "
              "in UFactory Studio, or pass --no-controller-collision.", file=sys.stderr)
        return False
    try:
        if safety.tool_length > 0:
            code = arm.set_collision_tool_model(CYLINDER_TOOL_MODEL, radius=safety.tool_radius,
                                                height=safety.tool_length)
        else:
            code = arm.set_collision_tool_model(0)
        code2 = arm.set_self_collision_detection(True)
    except Exception as exc:
        print(f"ERROR: could not set the controller self-collision model ({exc}); pass "
              "--no-controller-collision to run without it.", file=sys.stderr)
        return False
    if code != 0 or code2 != 0:
        print(f"ERROR: controller self-collision setup returned codes {code}, {code2}",
              file=sys.stderr)
        return False
    print(f"Controller self-collision detection on (cylinder tool r={safety.tool_radius:.0f} "
          f"mm, h={safety.tool_length:.0f} mm, collision sensitivity {sensitivity})")
    return True


class Rollout:
    def __init__(self, args: argparse.Namespace, plan: Plan, safety: Safety,
                 calibration: MocapCalibration, recording: Path) -> None:
        self.args = args
        self.plan = plan
        self.safety = safety
        self.calibration = calibration
        stamp = time.strftime("%H%M%S")
        self.log_path = recording.with_name(f"{recording.stem}_rollout_{stamp}.csv")
        self.mocap_path = recording.with_name(f"{recording.stem}_rollout_{stamp}_mocap.csv")
        self.jogger = None
        self.log: List[list] = []
        self.servo_start = 0.0

    def run(self) -> int:
        from xarm_keyboard_control import ArmJogger, Keyboard
        mocap = None
        if self.args.mocap:
            mocap = MocapLog(self.args)
            if not mocap.start():
                return 1
        ok = False
        keys: Optional[KeyMonitor] = None
        try:
            with Keyboard() as keyboard:
                keys = KeyMonitor(keyboard)
                keys.start()
                self.jogger = ArmJogger(self.args.arm_ip, 0.0, 0.0, 0)
                keys.arm = self.jogger.arm
                print("SPACE = e-stop (also Ctrl+C)")
                keys.check()
                ok = self.session(self.jogger.arm, keys, mocap)
        except EStopped:
            pass
        except KeyboardInterrupt:
            print("\nInterrupted; stopping arm.")
            if self.jogger is not None:
                self.jogger.arm.emergency_stop()
        except BaseException:
            if self.jogger is not None:
                print("\nUnexpected error; stopping arm.", file=sys.stderr)
                self.jogger.arm.emergency_stop()
            raise
        finally:
            if keys is not None:
                keys.stop()
            if self.jogger is not None:
                arm = self.jogger.arm
                if arm.mode != 0:
                    arm.set_mode(0)
                    arm.set_state(0)
                self.jogger.disconnect()
            if mocap is not None:
                mocap.recording = False
                mocap.shutdown()
            self.save_log()
            if mocap is not None:
                self.report_mocap(mocap)
        return 0 if ok else 1

    def session(self, arm, keys: KeyMonitor, mocap: Optional[MocapLog]) -> bool:
        err, warn = self.jogger.startup_error
        if err or warn:
            print(f"\n!!! The arm was reporting error {err} / warning {warn} when the script "
                  "connected (it has now been cleared).  Find out why before moving it.")
            if not keys.wait_for_enter("    Press Enter to continue anyway, q to quit."):
                return False
        result = self.jogger.pose()
        if result is None:
            return False
        pose, joints = result
        joints = np.array(joints, float)

        # The kinematic model must match the controller before trusting the plan.
        tcp = kin.pose_to_matrix(list(arm.tcp_offset[:6]))
        self.tcp = tcp
        model = kin.forward(joints) @ tcp
        actual = kin.pose_to_matrix(pose)
        dp = float(np.linalg.norm(model[:3, 3] - actual[:3, 3]))
        da = kin.rotation_angle_deg(model[:3, :3].T @ actual[:3, :3])
        if dp > 2.0 or da > 0.5:
            print(f"ERROR: kinematic model disagrees with the arm by {dp:.1f} mm / "
                  f"{da:.2f} deg; refusing to run", file=sys.stderr)
            return False

        start = self.plan.joints[0]
        if np.max(np.abs(joints - start)) > 0.5:
            # Like the auto calibration's move to ready: don't fold the arm any
            # closer to its base than it already is, and don't limit unwinding.
            free = replace(self.safety, max_wrist_rotation=None, min_z=None,
                           base_margin=min(self.safety.base_margin,
                                           arm_to_base(joints) - 5.0))
            reason = check_joint_move(joints, start, free)
            if reason:
                print(f"ERROR: the joint move from {np.round(joints, 1).tolist()} to the "
                      f"demo start is unsafe ({reason}).  Move the arm closer to the start "
                      "pose (e.g. xarm_keyboard_control.py) and retry.", file=sys.stderr)
                return False
            if not keys.wait_for_enter(
                    f"\nArm is at {np.round(joints, 1).tolist()}.  Clear the area, then press "
                    f"Enter to move to the demo start {np.round(start, 1).tolist()} at "
                    f"{self.args.approach_speed:g} deg/s (q quits)."):
                return False
            keys.check()
            code = arm.set_servo_angle(angle=[float(v) for v in start],
                                       speed=self.args.approach_speed,
                                       mvacc=self.args.approach_speed * 10,
                                       is_radian=False, wait=True)
            keys.check()
            if code != 0 or arm.error_code != 0:
                print(f"ERROR: move to start failed (code={code} error={arm.error_code})",
                      file=sys.stderr)
                return False

        if not setup_collision(arm, self.safety, self.args.no_controller_collision):
            return False
        if not keys.wait_for_enter(
                f"\nAt the demo start.  Press Enter to REPLAY ({self.plan.duration:.1f} s at "
                f"{100 / self.plan.slowdown:.0f}% of demo speed on average), q to quit.  "
                "SPACE = e-stop."):
            print("Quit without replaying.")
            return False

        # Servo mode: the controller tracks each set_servo_angle_j target with
        # no planning, so the plan's small, smooth steps are what keep it safe.
        _, now_joints = arm.get_servo_angle(is_radian=False)
        if np.max(np.abs(np.array(now_joints[:6]) - start)) > 1.0:
            print("ERROR: arm is not at the demo start; refusing to stream", file=sys.stderr)
            return False
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.3)
        if arm.mode != 1 or arm.error_code != 0:
            print(f"ERROR: could not enter servo mode (mode={arm.mode} "
                  f"error={arm.error_code})", file=sys.stderr)
            return False

        print(f"Replaying ({self.args.control} servo control) ...")
        if mocap is not None:
            mocap.recording = True
        ok = self.stream(arm, keys)
        time.sleep(0.5)  # let the arm settle on the final target
        if mocap is not None:
            mocap.recording = False
        arm.set_mode(0)
        arm.set_state(0)
        if ok:
            late = [row[1] - row[0] for row in self.log]
            print(f"Done.  Sent {len(self.log)} commands; worst send lateness "
                  f"{max(late) * 1000:.1f} ms.  The arm is at the demo's final pose.")
        return ok

    def stream(self, arm, keys: KeyMonitor) -> bool:
        dt = 1.0 / self.plan.rate
        self.servo_start = t_start = time.perf_counter()
        for i, q in enumerate(self.plan.joints):
            target = t_start + i * dt
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.002))
            keys.check()
            sent = time.perf_counter() - t_start
            if self.args.control == "cartesian":
                # Same target, as a TCP pose; the controller solves the IK.
                pose = kin.matrix_to_pose(kin.forward(q) @ self.tcp)
                code = arm.set_servo_cartesian(pose, is_radian=False)
            else:
                code = arm.set_servo_angle_j([float(v) for v in q], is_radian=False)
            reported = list(arm.angles[:6])
            self.log.append([i * dt, sent, *q, *reported, code, arm.error_code])
            if code != 0 or arm.error_code != 0:
                print(f"  ! servo command {i} failed: code={code} error={arm.error_code} "
                      f"warn={arm.warn_code}; stopping", file=sys.stderr)
                arm.emergency_stop()
                return False
        return True

    def save_log(self) -> None:
        if not self.log:
            return
        with self.log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["plan_time_s", "sent_time_s",
                             *(f"cmd_j{i}_deg" for i in range(1, 7)),
                             # The SDK's cached report: lags and repeats between updates.
                             *(f"reported_j{i}_deg" for i in range(1, 7)),
                             "code", "error_code"])
            for row in self.log:
                writer.writerow([round(v, 5) if isinstance(v, float) else v for v in row])
        print(f"Saved {self.log_path}")

    def report_mocap(self, mocap: MocapLog) -> None:
        """How closely did the replay follow the demo, as seen by Motive?"""
        frames = [f for f in mocap.frames if f[0] >= self.servo_start]
        if len(frames) < 10 or not self.log:
            return
        t = np.array([f[0] - self.servo_start for f in frames])
        measured = self.calibration.flange_poses(np.array([f[1] for f in frames]),
                                                 np.array([f[2] for f in frames]))
        plan_t = np.arange(len(self.plan.joints)) / self.plan.rate

        def errors(lag: float) -> Tuple[np.ndarray, np.ndarray]:
            tt = np.clip(t - lag, 0, plan_t[-1])
            q = np.column_stack([np.interp(tt, plan_t, self.plan.joints[:, j])
                                 for j in range(6)])
            cmd = np.array([kin.forward(v) for v in q])
            dp = np.linalg.norm(cmd[:, :3, 3] - measured[:, :3, 3], axis=1)
            da = np.array([kin.rotation_angle_deg(c[:3, :3].T @ m[:3, :3])
                           for c, m in zip(cmd, measured)])
            return dp, da

        dp0, da0 = errors(0.0)
        lags = np.arange(0.0, 0.301, 0.01)
        best = min(lags, key=lambda lag: float(np.sqrt(np.mean(errors(lag)[0] ** 2))))
        dp, da = errors(best)
        with self.mocap_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
                             "flange_x_mm", "flange_y_mm", "flange_z_mm",
                             "flange_roll_deg", "flange_pitch_deg", "flange_yaw_deg",
                             "pos_err_mm", "rot_err_deg"])
            for f, ti, m, e, a in zip(frames, t, measured, dp0, da0):
                writer.writerow([round(float(ti), 4), *f[1], *f[2],
                                 *np.round(kin.matrix_to_pose(m), 3),
                                 round(float(e), 3), round(float(a), 3)])
        rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
        print(f"Saved {self.mocap_path}")
        print(f"Replay vs demo (Motive, through the calibration): position RMS "
              f"{rms(dp0):.1f} mm / max {dp0.max():.1f} mm, rotation RMS {rms(da0):.1f} deg")
        print(f"  allowing for {best * 1000:.0f} ms of servo lag: position RMS {rms(dp):.1f} mm"
              f" / max {dp.max():.1f} mm, rotation RMS {rms(da):.1f} deg  "
              f"(calibration fit itself: {self.calibration.pos_rms_mm:.1f} mm)")


# --------------------------------------------------------------------- main
def make_safety(args: argparse.Namespace) -> Safety:
    # No --table-margin: the demo itself showed the hand fits above the table.
    table_margin = -math.inf if args.table_margin is None else args.table_margin
    return Safety(args.tool_length, args.tool_radius, args.table_z, table_margin,
                  args.link_radius, args.self_margin,
                  wrist_ref=(DEFAULT_READY_JOINTS[3], DEFAULT_READY_JOINTS[5]),
                  max_wrist_rotation=args.max_wrist_rotation,
                  min_z=args.min_z, base_margin=args.base_margin)


def parse_args() -> argparse.Namespace:
    from xarm_mocap_calibration import DEFAULT_RIGID_ID, MOCAP_SERVER_IP
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", type=Path,
                        help="record_mocap_camera.py CSV (or its .mp4/_meta.json, or a "
                             "folder: newest; default: newest under recordings/)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="*_solution.json (default: the one named in the recording's "
                             "_meta.json, else the newest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and check only; don't connect to the arm")

    traj = parser.add_argument_group("trajectory")
    traj.add_argument("--trim", type=float, nargs=2, metavar=("START", "END"),
                      help="only use this part of the recording, s")
    traj.add_argument("--no-auto-trim", action="store_true",
                      help="keep still time at the start/end of the recording")
    traj.add_argument("--still-mm", type=float, default=5.0,
                      help="auto-trim: motion threshold, mm (default: %(default)s)")
    traj.add_argument("--still-deg", type=float, default=3.0,
                      help="auto-trim: rotation threshold, deg (default: %(default)s)")
    traj.add_argument("--still-pad", type=float, default=0.5,
                      help="auto-trim: still time kept before/after the motion, s "
                           "(default: %(default)s)")
    traj.add_argument("--max-gap", type=float, default=0.1,
                      help="longest tracking dropout to interpolate over, s "
                           "(default: %(default)s)")
    traj.add_argument("--smooth", type=float, default=0.04,
                      help="Gaussian smoothing sigma, s; 0 disables (default: %(default)s)")
    traj.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                      metavar=("DX", "DY", "DZ"),
                      help="shift the whole trajectory in the arm base frame, mm")
    traj.add_argument("--rate", type=float, default=100.0,
                      help="servo command rate, Hz (default: %(default)s)")

    speed = parser.add_argument_group("speed limits (the replay is slowed to satisfy them)")
    speed.add_argument("--speed-scale", type=float, default=0.5,
                       help="fraction of the demo's speed, at most (default: %(default)s)")
    speed.add_argument("--max-joint-speed", type=float, default=60.0,
                       help="deg/s (default: %(default)s)")
    speed.add_argument("--max-joint-acc", type=float, default=300.0,
                       help="deg/s^2 (default: %(default)s)")
    speed.add_argument("--max-speed", type=float, default=250.0,
                       help="flange and tool-tip speed, mm/s (default: %(default)s)")
    speed.add_argument("--control", choices=("joint", "cartesian"), default="joint",
                       help="joint: stream the planned joint angles (our IK); cartesian: "
                            "stream end-effector poses and let the arm's built-in IK "
                            "solve them (default: %(default)s)")
    speed.add_argument("--approach-speed", type=float, default=20.0,
                       help="joint speed for the move to the demo start, deg/s "
                            "(default: %(default)s)")

    tool = parser.add_argument_group("tool and collision safety")
    tool.add_argument("--tool-length", type=float, required=True,
                      help="end-effector length along the flange Z axis, mm")
    tool.add_argument("--tool-radius", type=float, default=25.0,
                      help="tool radius incl. markers, mm (default: %(default)s)")
    tool.add_argument("--table-z", type=float, default=0.0,
                      help="table height in the arm base frame, mm (default: %(default)s)")
    tool.add_argument("--table-margin", type=float, default=None,
                      help="min clearance of the tool/arm above --table-z, mm (default: no "
                           "table check -- the demo already put the hand there)")
    tool.add_argument("--link-radius", type=float, default=60.0,
                      help="arm link radius for clearance checks, mm (default: %(default)s)")
    tool.add_argument("--self-margin", type=float, default=30.0,
                      help="min tool-to-arm clearance, mm (default: %(default)s)")
    tool.add_argument("--max-wrist-rotation", type=float, default=120.0,
                      help="max J4/J6 turn from 0, deg (cable wrap) (default: %(default)s)")
    tool.add_argument("--base-margin", type=float, default=130.0,
                      help="min forearm/wrist distance to the base column axis, mm "
                           "(default: %(default)s)")
    tool.add_argument("--min-z", type=float, default=None,
                      help="z floor for the flange centre and tool tip, mm (default: none)")
    tool.add_argument("--singular-margin", type=float, default=10.0,
                      help="min |J5| away from 0/180 deg (wrist singularity) "
                           "(default: %(default)s)")
    tool.add_argument("--no-controller-collision", action="store_true",
                      help="don't configure the controller's self-collision detection")

    conn = parser.add_argument_group("connections")
    conn.add_argument("--arm-ip", default="192.168.1.103",
                      help="xArm controller IP (default: %(default)s)")
    conn.add_argument("--mocap", action="store_true",
                      help="track the end effector in Motive during the replay and report "
                           "the error vs. the demo")
    conn.add_argument("--rigid-id", type=int, default=DEFAULT_RIGID_ID,
                      help="rigid body for --mocap (default: %(default)s)")
    conn.add_argument("--server-ip", default=MOCAP_SERVER_IP,
                      help="Motive/NatNet server IP (default: %(default)s)")
    conn.add_argument("--client-ip", default=None,
                      help="local IP for NatNet (default: automatic)")
    conn.add_argument("--unicast", action="store_true", help="NatNet unicast")

    args = parser.parse_args()
    if not 0 < args.speed_scale <= 2:
        parser.error("--speed-scale must be in (0, 2]")
    if not 0 < args.max_joint_speed <= MAX_JOINT_SPEED:
        parser.error(f"--max-joint-speed must be in (0, {MAX_JOINT_SPEED:g}]")
    if not 20 <= args.rate <= 250:
        parser.error("--rate must be 20-250 Hz")
    if not 0 < args.approach_speed <= 60:
        parser.error("--approach-speed must be in (0, 60] deg/s")
    if args.trim and args.trim[0] >= args.trim[1]:
        parser.error("--trim START must be < END")
    return args


def main() -> int:
    args = parse_args()
    recording = resolve_recording(args.recording)
    print(f"Recording: {recording}")
    calibration = calibration_for(recording, args.calibration)
    print(f"Calibration: {calibration.describe()}")
    if calibration.tool_length is not None and abs(
            calibration.tool_length - args.tool_length) > 1:
        print(f"Note: calibration was run with --tool-length {calibration.tool_length:g}")
    demo = load_demo(recording)
    safety = make_safety(args)
    try:
        plan = build_plan(demo, calibration, args, safety)
    except PlanError as exc:
        print(f"\nCannot replay this recording: {exc}", file=sys.stderr)
        return 1
    print_plan(demo, plan, safety)
    plan_path = recording.with_name(recording.stem + "_plan.csv")
    save_plan(plan_path, plan)
    print(f"Plan written to {plan_path}")
    if args.dry_run:
        return 0
    if not sys.stdin.isatty():
        print("ERROR: needs an interactive terminal (SPACE is the e-stop)", file=sys.stderr)
        return 1
    return Rollout(args, plan, safety, calibration, recording).run()


if __name__ == "__main__":
    raise SystemExit(main())
