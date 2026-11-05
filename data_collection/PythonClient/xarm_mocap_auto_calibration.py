#!/usr/bin/env python3
"""Automatically drive a UFactory xArm 6 through a calibration pattern.

Same output as xarm_mocap_calibration.py (arm pose + averaged mocap pose per
sample), but the arm moves itself instead of being jogged by hand.

  1. After you press Enter, the arm moves (joint move, slow) to a "ready" pose:
     tool pointing forward (+X) at the same height as the xArm home pose
     with the wrist bent, but 100 mm farther out.  (The home-based pose
     [0, 0, 0, 0, -90, 0] keeps the arm folded so tightly that moving back
     from it brought the wrist next to the base and tripped the controller's
     self-collision check.)  It then plans the pattern, prints it, and waits: mount
     the tool if it isn't on yet, then press Enter to run (q quits).
     The default is joints [0, 10.6, -18.9, 0, -81.7, 0]; override with
     --ready-joints.  (Tool points forward whenever J4 = J6 = 180 and
     J5 = J2 + J3 + 90, or J4 = J6 = 0 and J2 + J3 + J5 = -90.)
  2. From there it visits a grid of positions inside a box defined *relative to
     the ready TCP position* (--x-range/--y-range/--z-range, in mm), tilting
     and rolling the tool at each point so the data has rotation about every
     axis.  These are linear (Cartesian) moves at --speed (default 150 mm/s);
     fallback joint moves are slowed so the tool tip stays under that too.
  3. At each pose it waits --settle seconds and records a sample.
  4. It returns to the ready pose.

Besides the samples CSV, every mocap frame of the rigid body (plus the arm's
reported pose) is logged to <name>_stream.csv from the move to ready onward;
plot it with plot_mocap_trajectory.py.

E-STOP: press SPACE (or Ctrl+C) at any time once the arm is connected; the arm
stops immediately and the samples recorded so far are saved.  This is a
software stop over the network -- keep the hardware e-stop within reach.

Every move is checked *before the arm moves at all*: the whole route is
simulated in small steps with an xArm 6 kinematic model and a model of the
tool (a cylinder of --tool-length x --tool-radius along the flange Z axis).
Poses that would put the tool or arm below the table, bring the tool close to
the arm, hit joint limits, pass near a singularity or turn J4/J6 more than
--max-wrist-rotation from the ready pose (cable wrap) are skipped.  The
controller's own self-collision detection is also enabled with the same
cylinder tool model; the run refuses to start if that can't be set up.

NOT modelled: anything mounted off the flange axis (e.g. the arm camera and its
cable -- grow --tool-radius to cover it), and everything around the arm
(Motive cameras/tripods, laptops, people).  Run --dry-run and check the plan.

Network: Motive streams from 192.168.1.1 (--server-ip); the local address is
picked automatically from the interface that reaches it.  The xArm is at
--arm-ip on the same ethernet.

Typical use:
  # Preview the plan offline (no arm/mocap needed).
  python xarm_mocap_auto_calibration.py --dry-run --tool-length 300
  # Run it (rigid body 1): arm goes to ready, mount the tool, press Enter.
  python xarm_mocap_auto_calibration.py 1 --tool-length 300

If the move to the ready pose would only be safe without the tool (e.g. the
arm starts at home), the script says so and asks you to confirm the tool is
not mounted yet before moving.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

import xarm6_kinematics as kin
from xarm_keyboard_control import POSE_FORMAT, Keyboard
from xarm_mocap_calibration import Calibrator, add_calibration_args, local_ip_for


# Tool pointing forward (+X) at TCP (328, 0, 285) mm: the home height with the
# wrist bent, 100 mm farther out than [0, 0, 0, 0, -90, 0].  That closer pose
# sits within ~10 deg of the J3 fold limit, so pulling back from it folds the
# wrist down beside the base (controller self-collision, error 22).
DEFAULT_READY_JOINTS = [0.0, 10.6, -18.9, 0.0, -81.7, 0.0]
FALLBACK_MAX_SPEED = 1000.0    # mm/s, xArm 6 spec
DEFAULT_SPEED = 150.0          # mm/s; raise with --speed once the plan is trusted
DEFAULT_ACC = 10000.0          # mm/s^2 (~1 g); the arm allows 50000, which whips a long tool
MAX_JOINT_SPEED = 180.0        # deg/s, xArm 6 spec
JOINT_SPEED_USE = 0.8          # fraction of MAX_JOINT_SPEED a linear move may demand
MIN_LINEAR_SPEED = 100.0       # mm/s; needing slower than this means near a singularity
WRIST_SINGULAR_MARGIN = 15.0   # deg; J5 near 0/+-180 aligns J4 and J6 (they spin wildly)
MM_PER_DEG = 5.0               # equivalence of rotation to travel when timing moves
CYLINDER_TOOL_MODEL = 21       # set_collision_tool_model type for a cylinder

FLANGE_RADIUS = 45.0           # mm, wrist/flange housing
BASE_COLUMN_TOP = 267.0        # mm, height of joint 2
SELF_CHECK_START = 100.0       # mm along the tool; the part nearer the flange can't reach other links

# Orientation perturbations cycled through the grid, as multiples of
# (--tilt down, --tilt left, --roll about the tool axis).
ORIENTATIONS = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
    (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1), (1, -1, 1),
]
SHRINK_FACTORS = (1.0, 0.5)    # retry an unreachable orientation at half the angle


# --------------------------------------------------------------------- safety
@dataclass
class Safety:
    tool_length: float
    tool_radius: float
    table_z: float
    table_margin: float
    link_radius: float
    self_margin: float
    joint_margin: float = 2.0
    # J4/J6 may not turn further than this from wrist_ref (the ready pose's J4,
    # J6): cables running down the arm (camera, tool) wrap around the wrist.
    wrist_ref: Optional[Tuple[float, float]] = None
    max_wrist_rotation: Optional[float] = None
    # Neither the flange centre nor the tool tip may go below this base-frame z (mm).
    min_z: Optional[float] = None
    # Min distance from the forearm/wrist (link centrelines) to the base column axis.
    base_margin: float = 0.0


MIN_Z_TOLERANCE = 1.0  # mm; "--min-z ready" allows this much below ready (IK noise)


def tool_low_z(joints: Sequence[float], tool_length: float) -> float:
    """Lowest z of the flange centre and the tool tip, mm."""
    flange = kin.forward(joints)
    return float(min(flange[2, 3], flange[2, 3] + tool_length * flange[2, 2]))


def make_safety(args: argparse.Namespace) -> Safety:
    min_z = None
    if args.min_z is not None:
        min_z = (tool_low_z(args.ready_joints, args.tool_length) - MIN_Z_TOLERANCE
                 if args.min_z == "ready" else float(args.min_z))
    return Safety(args.tool_length, args.tool_radius, args.table_z, args.table_margin,
                  args.link_radius, args.self_margin,
                  wrist_ref=(args.ready_joints[3], args.ready_joints[5]),
                  max_wrist_rotation=args.max_wrist_rotation, min_z=min_z,
                  base_margin=args.base_margin)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def clearances(joints: Sequence[float], safety: Safety) -> Tuple[float, float]:
    """(table clearance, tool-to-arm clearance) in mm; negative means contact."""
    frames = kin.forward_frames(joints)
    flange = frames[-1]
    origin, axis = flange[:3, 3], flange[:3, 2]
    tip = origin + safety.tool_length * axis

    lowest = min(
        origin[2] - FLANGE_RADIUS,
        tip[2] - safety.tool_radius,
        origin[2] - safety.tool_radius,
        *(f[2, 3] - safety.link_radius for f in frames[3:6]),  # elbow, wrist
    )
    table = lowest - safety.table_z

    self_clear = math.inf
    if safety.tool_length > SELF_CHECK_START:
        links = [
            (np.zeros(3), np.array([0.0, 0.0, BASE_COLUMN_TOP])),  # base column
            (frames[2][:3, 3], frames[3][:3, 3]),                    # upper arm
            (frames[3][:3, 3], frames[4][:3, 3]),                    # forearm
        ]
        for s in np.arange(SELF_CHECK_START, safety.tool_length + 1e-9, 10.0):
            p = origin + s * axis
            for a, b in links:
                d = _point_segment_distance(p, a, b) - safety.tool_radius - safety.link_radius
                self_clear = min(self_clear, d)
    return table, self_clear


def arm_to_base(joints: Sequence[float]) -> float:
    """Closest distance (mm) from the forearm and wrist link centrelines to the
    base column's axis.  The planner's tool checks don't cover the arm folding
    onto itself; the controller does, and stops the run (error 22).  Measured:
    88 mm tripped it, 103 mm and up did not."""
    frames = kin.forward_frames(joints)
    col_a, col_b = np.zeros(3), np.array([0.0, 0.0, BASE_COLUMN_TOP])
    best = math.inf
    for i in (3, 4, 5):
        a, b = frames[i][:3, 3], frames[i + 1][:3, 3]
        n = max(2, int(np.linalg.norm(b - a) / 10.0) + 1)
        for s in np.linspace(0.0, 1.0, n):
            best = min(best, _point_segment_distance(a + (b - a) * s, col_a, col_b))
    return best


def check_joints(joints: Sequence[float], safety: Safety) -> Optional[str]:
    """Reason the configuration is unsafe, or None."""
    if not kin.within_limits(joints, safety.joint_margin):
        return "joint limit"
    if safety.wrist_ref is not None and safety.max_wrist_rotation is not None:
        turn = max(abs(joints[3] - safety.wrist_ref[0]), abs(joints[5] - safety.wrist_ref[1]))
        if turn > safety.max_wrist_rotation:
            return f"wrist turns {turn:.0f} deg from ready (cable wrap)"
    if safety.min_z is not None:
        low = tool_low_z(joints, safety.tool_length)
        if low < safety.min_z:
            return f"below the z floor ({low:.0f} < {safety.min_z:.0f} mm)"
    if safety.base_margin > 0:
        d = arm_to_base(joints)
        if d < safety.base_margin:
            return f"arm folds too close to its base ({d:.0f} mm)"
    table, self_clear = clearances(joints, safety)
    if table < safety.table_margin:
        return f"too close to table ({table:.0f} mm)"
    if self_clear < safety.self_margin:
        return f"tool too close to arm ({self_clear:.0f} mm)"
    return None


# -------------------------------------------------------------------- planner
@dataclass
class Segment:
    """One simulated move: a straight-line TCP move ("linear") at `speed` mm/s,
    or a joint-space move ("joint") at `speed` deg/s, ending at joints[-1]."""
    target: np.ndarray
    joints: List[np.ndarray]
    kind: str
    speed: float


@dataclass
class Stop:
    label: str
    segments: List[Segment]
    record: bool = True


@dataclass
class Plan:
    stops: List[Stop] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    min_table: float = math.inf
    min_self: float = math.inf
    path_mm: float = 0.0


class Planner:
    def __init__(self, ready_joints: Sequence[float], tcp_offset: np.ndarray,
                 safety: Safety, speed: float = FALLBACK_MAX_SPEED,
                 joint_speed: float = 60.0) -> None:
        self.safety = safety
        self.speed = speed
        self.joint_speed = joint_speed
        self.tcp_offset = tcp_offset
        self.tcp_offset_inv = np.linalg.inv(tcp_offset)
        self.ready_joints = np.array(ready_joints, dtype=float)
        self.ready_tcp = kin.forward(self.ready_joints) @ tcp_offset

    def tcp(self, joints: Sequence[float]) -> np.ndarray:
        return kin.forward(joints) @ self.tcp_offset

    def simulate_linear(self, start_q: np.ndarray, target: np.ndarray,
                        plan: Plan) -> Tuple[Optional[Segment], str]:
        """Follow the controller's straight-line TCP move in small IK steps."""
        start = self.tcp(start_q)
        dist = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        angle = kin.rotation_angle_deg(start[:3, :3].T @ target[:3, :3])
        steps = max(1, math.ceil(dist / 5.0), math.ceil(angle / 1.0))
        # Equivalent travel per step, to estimate joint speed at a given TCP speed.
        step_travel = max(dist / steps, MM_PER_DEG * angle / steps, 1e-6)
        q = start_q
        path = []
        min_table = min_self = math.inf
        max_ratio = 0.0  # deg of joint motion per mm of travel
        for i in range(1, steps + 1):
            pose = kin.interpolate(start, target, i / steps)
            q_next = kin.inverse(pose @ self.tcp_offset_inv, q)
            if q_next is None:
                return None, "unreachable"
            max_ratio = max(max_ratio, float(np.max(np.abs(q_next - q))) / step_travel)
            if abs(math.sin(math.radians(q_next[4]))) < math.sin(
                    math.radians(WRIST_SINGULAR_MARGIN)):
                return None, "near wrist singularity"
            reason = check_joints(q_next, self.safety)
            if reason:
                return None, reason
            table, self_clear = clearances(q_next, self.safety)
            min_table, min_self = min(min_table, table), min(min_self, self_clear)
            q = q_next
            path.append(q)
        # Slow down where the joints would otherwise exceed their speed limit.
        # The singularity test uses the joint-limited speed alone: comparing the
        # final speed would reject every move whenever --speed < MIN_LINEAR_SPEED,
        # silently turning a slow run into one made of (faster) joint moves.
        joint_limited = JOINT_SPEED_USE * MAX_JOINT_SPEED / max(max_ratio, 1e-9)
        if joint_limited < MIN_LINEAR_SPEED:
            return None, "near singularity"
        speed = min(self.speed, joint_limited)
        plan.min_table = min(plan.min_table, min_table)
        plan.min_self = min(plan.min_self, min_self)
        plan.path_mm += dist
        return Segment(target, path, "linear", speed), ""

    def simulate_joint(self, start_q: np.ndarray, target: np.ndarray,
                       plan: Plan) -> Tuple[Optional[Segment], str]:
        """Joint-space move to an IK solution of target; immune to singularities."""
        flange = target @ self.tcp_offset_inv
        end_q = None
        for seed in (start_q, self.ready_joints):
            end_q = kin.inverse(flange, seed, iterations=300)
            if end_q is not None and kin.within_limits(end_q, self.safety.joint_margin):
                break
            end_q = None
        if end_q is None:
            return None, "unreachable"
        span = float(np.max(np.abs(end_q - start_q)))
        steps = max(1, math.ceil(span / 1.0))
        path = [start_q + (end_q - start_q) * i / steps for i in range(1, steps + 1)]
        for q in path:
            reason = check_joints(q, self.safety)
            if reason:
                return None, reason
        # The controller moves the fastest joint at `speed` deg/s; cap that so
        # neither the flange nor the tool tip moves faster than --speed mm/s.
        step_deg = max(span / steps, 1e-9)
        max_mm_per_deg = 0.0
        prev = self.tool_points(start_q)
        for q in path:
            cur = self.tool_points(q)
            max_mm_per_deg = max(max_mm_per_deg, *(
                float(np.linalg.norm(c - p)) / step_deg for c, p in zip(cur, prev)))
            prev = cur
        speed = min(self.joint_speed, self.speed / max(max_mm_per_deg, 1e-9))
        for q in path:
            table, self_clear = clearances(q, self.safety)
            plan.min_table = min(plan.min_table, table)
            plan.min_self = min(plan.min_self, self_clear)
        plan.path_mm += float(np.linalg.norm(target[:3, 3] - self.tcp(start_q)[:3, 3]))
        return Segment(target, path, "joint", speed), ""

    def tool_points(self, joints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Flange centre and tool tip, mm, in the base frame."""
        flange = kin.forward(joints)
        return flange[:3, 3], flange[:3, 3] + self.safety.tool_length * flange[:3, 2]

    def route(self, start_q: np.ndarray, target: np.ndarray,
              plan: Plan) -> Tuple[Optional[List[Segment]], str]:
        """Direct linear move, else linear via the neutral orientation at both
        ends, else a joint-space move."""
        neutral = self.ready_tcp[:3, :3]
        start = self.tcp(start_q)
        via_a, via_b = start.copy(), target.copy()
        via_a[:3, :3] = neutral
        via_b[:3, :3] = neutral
        reasons = []
        options = [[(self.simulate_linear, target)],
                   [(self.simulate_linear, wp) for wp in (via_a, via_b, target)],
                   [(self.simulate_joint, target)]]
        for option in options:
            trial = Plan()
            segments: List[Segment] = []
            q = start_q
            for simulate, wp in option:
                seg, reason = simulate(q, wp, trial)
                if seg is None:
                    reasons.append(reason)
                    break
                segments.append(seg)
                q = seg.joints[-1]
            else:
                plan.min_table = min(plan.min_table, trial.min_table)
                plan.min_self = min(plan.min_self, trial.min_self)
                plan.path_mm += trial.path_mm
                return segments, ""
        return None, reasons[0]

    def target(self, offset: Sequence[float], down: float, left: float,
               roll: float) -> np.ndarray:
        t = np.eye(4)
        t[:3, 3] = self.ready_tcp[:3, 3] + np.asarray(offset)
        t[:3, :3] = (kin.rot_z(left) @ kin.rot_y(down) @ self.ready_tcp[:3, :3]
                     @ kin.rot_z(roll))
        return t

    def build(self, ranges: Sequence[Tuple[float, float]], grid: int, per_point: int,
              tilt: float, roll: float) -> Plan:
        plan = Plan()
        axes = [np.linspace(lo, hi, grid) if grid > 1 else np.array([(lo + hi) / 2])
                for lo, hi in ranges]
        # Serpentine order so consecutive points are neighbours.
        offsets = []
        for ix, x in enumerate(axes[0]):
            ys = axes[1] if ix % 2 == 0 else axes[1][::-1]
            for iy, y in enumerate(ys):
                zs = axes[2] if (ix * len(ys) + iy) % 2 == 0 else axes[2][::-1]
                offsets.extend((x, y, z) for z in zs)

        orientations = itertools.cycle(ORIENTATIONS)
        q = self.ready_joints
        for offset in offsets:
            for _ in range(per_point):
                mult = next(orientations)
                label = (f"offset=({offset[0]:+.0f},{offset[1]:+.0f},{offset[2]:+.0f}) mm "
                         f"down/left/roll=({mult[0] * tilt:+.0f},{mult[1] * tilt:+.0f},"
                         f"{mult[2] * roll:+.0f}) deg")
                reason = ""
                for f in SHRINK_FACTORS:
                    tgt = self.target(offset, f * mult[0] * tilt, f * mult[1] * tilt,
                                      f * mult[2] * roll)
                    segments, reason = self.route(q, tgt, plan)
                    if segments is not None:
                        if f != 1.0:
                            label += f" (angles x{f:g})"
                        plan.stops.append(Stop(label, segments))
                        q = segments[-1].joints[-1]
                        break
                else:
                    plan.skipped.append((label, reason))

        segments, reason = self.route(q, self.ready_tcp, plan)
        if segments is None:
            raise SystemExit(f"ERROR: no safe path back to the ready pose ({reason}); "
                             "shrink the ranges or angles")
        plan.stops.append(Stop("return to ready", segments, record=False))
        return plan


def check_joint_move(start: Sequence[float], end: Sequence[float],
                     safety: Safety) -> Optional[str]:
    """Check a joint-space move (the controller interpolates joints linearly)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    steps = max(1, math.ceil(np.max(np.abs(end - start)) / 1.0))
    for i in range(steps + 1):
        q = start + (end - start) * i / steps
        reason = check_joints(q, safety)
        if reason:
            return f"{reason} at joints {np.round(q, 1).tolist()}"
    return None


def describe_route(segments: List[Segment], speed: float) -> str:
    notes = []
    if len(segments) > 1:
        notes.append(f"via {len(segments) - 1} waypoints")
    joint = [s.speed for s in segments if s.kind == "joint"]
    if joint:
        notes.append(f"joint move at {min(joint):.0f} deg/s")
    slow = [s.speed for s in segments if s.kind == "linear" and s.speed < speed - 0.5]
    if slow:
        notes.append(f"slowed to {min(slow):.0f} mm/s")
    return f" ({', '.join(notes)})" if notes else ""


def print_plan(plan: Plan, safety: Safety, speed: float) -> None:
    n = sum(s.record for s in plan.stops)
    print(f"\nPlan: {n} sample poses, {len(plan.skipped)} skipped, "
          f"~{plan.path_mm / 1000:.1f} m of travel")
    for i, stop in enumerate(plan.stops):
        print(f"  {i:3d}  {stop.label}{describe_route(stop.segments, speed)}")
    for label, reason in plan.skipped:
        print(f"  skip {label}: {reason}")
    if safety.tool_length > SELF_CHECK_START:
        self_text = f"{plan.min_self:.0f} mm"
    else:
        self_text = "n/a (tool shorter than "f"{SELF_CHECK_START:.0f} mm)"
    print(f"Closest approach: table {plan.min_table:.0f} mm "
          f"(required {safety.table_margin:.0f}), tool-to-arm {self_text} "
          f"(required {safety.self_margin:.0f})")
    print(f"Linear speed {speed:.0f} mm/s\n")


# --------------------------------------------------------------------- runner
class EStopped(Exception):
    pass


class KeyMonitor:
    """Reads the keyboard on a background thread for the whole run.

    SPACE stops the arm at once (a blocking move in the main thread then
    returns); Enter and q are handed to the main thread's prompts.
    """

    def __init__(self, keyboard: Keyboard) -> None:
        self.keyboard = keyboard
        self.arm = None
        self.estop = threading.Event()
        self.enter = threading.Event()
        self.quit = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="estop", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.done.set()
        self.thread.join(timeout=1.0)

    def trigger_estop(self) -> None:
        if self.arm is not None:
            self.arm.emergency_stop()
        if not self.estop.is_set():
            self.estop.set()
            print("\n*** E-STOP: arm stopped ***")

    def _loop(self) -> None:
        while not self.done.is_set():
            try:
                key = self.keyboard.read_key(0.05)
            except KeyboardInterrupt:  # Windows delivers Ctrl+C here
                key = " "
            if key == " ":
                self.trigger_estop()
            elif key in ("\n", "\r"):
                self.enter.set()
            elif key == "q":
                self.quit.set()

    def check(self) -> None:
        if self.estop.is_set():
            raise EStopped

    def wait_for_enter(self, prompt: str) -> bool:
        """True on Enter, False on q; raises EStopped on SPACE."""
        print(prompt, flush=True)
        self.keyboard.flush()
        self.enter.clear()
        self.quit.clear()
        while not self.enter.wait(0.1):
            self.check()
            if self.quit.is_set():
                return False
        self.check()
        return True


STREAM_HEADER = [
    "time_s", "mocap_frame", "phase", "tracking_valid",
    "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "error_m",
    # The SDK's cached report (a few Hz), not a fresh read: repeats between updates.
    "arm_x_mm", "arm_y_mm", "arm_z_mm", "arm_roll_deg", "arm_pitch_deg", "arm_yaw_deg",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
]


class AutoCalibrator(Calibrator):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.safety = make_safety(args)
        # Every mocap frame of the rigid body, for plot_mocap_trajectory.py.
        self.stream_path = self.csv_path.with_name(self.csv_path.stem + "_stream.csv")
        self.stream_lock = threading.Lock()
        self.stream_file = None
        self.stream_writer = None
        self.phase = "connect"

    def start_stream(self) -> None:
        import csv
        with self.stream_lock:
            self.stream_file = self.stream_path.open("w", newline="", encoding="utf-8",
                                                     buffering=1)
            self.stream_writer = csv.writer(self.stream_file)
            self.stream_writer.writerow(STREAM_HEADER)
        print(f"Logging every mocap frame to {self.stream_path}")

    def stop_stream(self) -> None:
        with self.stream_lock:
            if self.stream_file is not None:
                self.stream_file.close()
            self.stream_file = self.stream_writer = None

    def on_mocap_frame(self, data: dict) -> None:
        super().on_mocap_frame(data)
        with self.stream_lock:
            if self.stream_writer is None:
                return
            bodies = data["mocap_data"].rigid_body_data.rigid_body_list
            body = next((rb for rb in bodies if rb.id_num == self.args.rigid_id), None)
            if body is None:
                mocap = [0] + [""] * 8
            else:
                mocap = [int(body.tracking_valid), *body.pos, *body.rot, body.error]
            arm = self.jogger.arm if self.jogger is not None else None
            try:
                pose, joints = list(arm.position[:6]), list(arm.angles[:6])
            except Exception:  # not connected yet
                pose, joints = [""] * 6, [""] * 6
            self.stream_writer.writerow([
                round(time.monotonic() - self.clock_zero, 4), data.get("frame_number", ""),
                self.phase, *mocap, *pose, *joints])

    def connect_arm(self):
        from xarm_keyboard_control import ArmJogger  # imports the xArm SDK
        print(f"xArm: {self.args.arm_ip} via local "
              f"{local_ip_for(self.args.arm_ip) or '(no route!)'}")
        self.jogger = ArmJogger(self.args.arm_ip, 0.0, 0.0, 0)
        arm = self.jogger.arm
        speed = min(self.args.speed, _limit(arm, "tcp_speed_limit", FALLBACK_MAX_SPEED))
        mvacc = min(self.args.mvacc, _limit(arm, "tcp_acc_limit", self.args.mvacc))
        return arm, speed, mvacc

    def setup_tool_model(self, arm) -> bool:
        """Turn on the controller's collision checks; False if they can't be trusted."""
        if self.args.no_controller_collision:
            print("WARNING: controller self-collision setup skipped "
                  "(--no-controller-collision)")
            return True
        sensitivity = getattr(arm, "collision_sensitivity", None)
        if sensitivity == 0:
            print("ERROR: the arm's collision detection is OFF (sensitivity 0).  Set it "
                  "to 1-5 in UFactory Studio, or pass --no-controller-collision to run "
                  "without it.", file=sys.stderr)
            return False
        try:
            if self.safety.tool_length > 0:
                code = arm.set_collision_tool_model(
                    CYLINDER_TOOL_MODEL, radius=self.safety.tool_radius,
                    height=self.safety.tool_length)
            else:
                code = arm.set_collision_tool_model(0)
            code2 = arm.set_self_collision_detection(True)
        except Exception as exc:  # older SDK/firmware
            print(f"ERROR: could not set the controller self-collision model ({exc}); "
                  "pass --no-controller-collision to run without it.", file=sys.stderr)
            return False
        if code != 0 or code2 != 0:
            print(f"ERROR: controller self-collision setup returned codes {code}, {code2}; "
                  "pass --no-controller-collision to run without it.", file=sys.stderr)
            return False
        print(f"Controller self-collision detection on (cylinder tool "
              f"r={self.safety.tool_radius:.0f} mm, h={self.safety.tool_length:.0f} mm, "
              f"collision sensitivity {sensitivity})")
        return True

    def goto_ready(self, arm, ready_joints: Sequence[float],
                   keys: Optional[KeyMonitor]) -> bool:
        code, joints = arm.get_servo_angle(is_radian=False)
        if code != 0:
            print(f"ERROR: could not read joint angles (code {code})", file=sys.stderr)
            return False
        joints = np.array(joints[:6], float)
        if np.max(np.abs(joints - ready_joints)) < 0.5:
            print("Arm already at the ready pose.")
            return True
        # Going *to* ready only unwinds the wrist, so no cable-wrap limit here.
        # The floor is relative to ready, and the arm may start below it (e.g. at home).
        # Likewise it may start folded closer to its base than the pattern allows
        # (home is 115 mm); just don't fold any further on the way.
        free = replace(self.safety, max_wrist_rotation=None, min_z=None,
                       base_margin=min(self.safety.base_margin, arm_to_base(joints) - 5.0))
        reason = check_joint_move(joints, ready_joints, free)
        if reason:
            bare = replace(free, tool_length=0.0, tool_radius=0.0)
            if check_joint_move(joints, ready_joints, bare) or keys is None:
                print(f"ERROR: moving to the ready pose from {np.round(joints, 1).tolist()} "
                      f"is unsafe ({reason}).\nMove the arm closer to the ready pose "
                      "by hand (e.g. with xarm_keyboard_control.py) and retry.",
                      file=sys.stderr)
                return False
            print(f"\n!!! Moving from {np.round(joints, 1).tolist()} to the ready pose is "
                  f"only safe WITHOUT the {self.safety.tool_length:.0f} mm tool "
                  f"({reason}).")
            if not keys.wait_for_enter("    Press Enter ONLY if the tool is NOT mounted "
                                       "(q to quit)."):
                return False
        elif keys is not None and not self.args.yes and not keys.wait_for_enter(
                f"\nArm is at {np.round(joints, 1).tolist()}.  Clear the area, then press "
                f"Enter to move to the ready pose {list(ready_joints)} (q quits)."):
            return False
        print(f"Moving to ready joints {list(ready_joints)} at {self.args.ready_speed} deg/s "
              "(SPACE = e-stop) ...")
        if keys is not None:
            keys.check()
        code = arm.set_servo_angle(angle=list(ready_joints), speed=self.args.ready_speed,
                                   mvacc=self.args.ready_speed * 10, is_radian=False,
                                   wait=True)
        if keys is not None:
            keys.check()
        if code != 0 or arm.error_code != 0:
            print(f"ERROR: move to ready failed (code={code} error={arm.error_code})",
                  file=sys.stderr)
            return False
        return True

    def verify_model(self, arm, planner: Planner) -> bool:
        """The kinematic model must match the controller before trusting the plan."""
        result = self.jogger.pose()
        if result is None:
            return False
        pose, joints = result
        model = planner.tcp(joints)
        actual = kin.pose_to_matrix(pose)
        dp = float(np.linalg.norm(model[:3, 3] - actual[:3, 3]))
        da = kin.rotation_angle_deg(model[:3, :3].T @ actual[:3, :3])
        if dp > 2.0 or da > 0.5:
            print(f"ERROR: kinematic model disagrees with the arm by {dp:.1f} mm / "
                  f"{da:.2f} deg; refusing to run (is this an xArm 6?)", file=sys.stderr)
            return False
        return True

    def execute(self, arm, plan: Plan, mvacc: float, keys: KeyMonitor) -> bool:
        stops = [s for s in plan.stops if s.record]
        for stop in plan.stops:
            keys.check()
            if stop.record:
                n = stops.index(stop) + 1
                self.phase = f"move {n}"
                print(f"[{n}/{len(stops)}] {stop.label}")
            else:
                self.phase = "return"
                print(stop.label)
            for seg in stop.segments:
                keys.check()
                if seg.kind == "joint":
                    code = arm.set_servo_angle(angle=[float(v) for v in seg.joints[-1]],
                                               speed=seg.speed, mvacc=seg.speed * 10,
                                               is_radian=False, wait=True)
                else:
                    x, y, z, roll, pitch, yaw = kin.matrix_to_pose(seg.target)
                    code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                            speed=seg.speed, mvacc=mvacc, is_radian=False,
                                            relative=False, wait=True)
                keys.check()
                if code != 0 or arm.error_code != 0:
                    print(f"  ! move failed: code={code} error={arm.error_code} "
                          f"warn={arm.warn_code}; stopping", file=sys.stderr)
                    return False
            if stop.record and self.args.no_mocap:
                time.sleep(self.args.settle)
                result = self.jogger.pose()
                if result is not None:
                    print("  reached " + POSE_FORMAT.format(*result[0]) +
                          "  (--no-mocap: nothing recorded)")
            elif stop.record:
                self.phase = f"sample {n}"
                self.record_sample()
        self.phase = "done"
        keys.check()
        return True

    def save_config(self, plan: Plan, tcp_offset: Sequence[float], speed: float,
                    mvacc: float) -> None:
        cfg = {k: (str(v) if not isinstance(v, (int, float, list, type(None), bool)) else v)
               for k, v in vars(self.args).items()}
        cfg["speed"], cfg["mvacc"] = speed, mvacc
        cfg["tcp_offset"] = list(tcp_offset)
        cfg["planned_samples"] = sum(s.record for s in plan.stops)
        path = self.csv_path.with_name(self.csv_path.stem + "_config.json")
        path.write_text(json.dumps(cfg, indent=2))

    def run(self) -> int:
        if self.args.no_mocap:
            print("--no-mocap: arm-only test run, Motive is not used and nothing is saved")
        elif not self.connect_mocap(require_body=True):
            return 1
        ok = False
        keys: Optional[KeyMonitor] = None
        try:
            with Keyboard() as keyboard:
                keys = KeyMonitor(keyboard)
                keys.start()
                arm, speed, mvacc = self.connect_arm()
                keys.arm = arm
                print("SPACE = e-stop (also Ctrl+C)")
                keys.check()  # SPACE pressed while connecting
                ok = self.session(arm, speed, mvacc, keys)
        except EStopped:
            pass
        except KeyboardInterrupt:
            print("\nInterrupted; stopping arm.")
            if self.jogger is not None:
                self.jogger.arm.emergency_stop()
        except BaseException:
            # Any crash (SDK, network, bug): stop the arm before reporting it.
            if self.jogger is not None:
                print("\nUnexpected error; stopping arm.", file=sys.stderr)
                self.jogger.arm.emergency_stop()
            raise
        finally:
            if keys is not None:
                keys.stop()
            self.client.shutdown()
            self.stop_stream()
            if self.jogger is not None:
                self.jogger.disconnect()

        if self.samples:
            print(f"Saved {len(self.samples)} samples to {self.csv_path}")
        else:
            print("No samples recorded.")
        return 0 if ok else 1

    def session(self, arm, speed: float, mvacc: float, keys: KeyMonitor) -> bool:
        args = self.args
        err, warn = self.jogger.startup_error
        if err or warn:
            # ArmJogger cleared it; make the user acknowledge it even with --yes.
            print(f"\n!!! The arm was reporting error {err} / warning {warn} when the script "
                  "connected (it has now been cleared).  Find out why (collision? e-stop? "
                  "joint limit?) before moving it again.")
            if not keys.wait_for_enter("    Press Enter to continue anyway, q to quit."):
                return False
        tcp_offset = list(arm.tcp_offset[:6])
        if any(abs(v) > 1e-6 for v in tcp_offset):
            print(f"Using controller TCP offset {tcp_offset}")
        planner = Planner(args.ready_joints, kin.pose_to_matrix(tcp_offset), self.safety,
                          speed, args.joint_speed)
        if not args.no_mocap:
            self.start_stream()
        self.phase = "to ready"
        if not self.goto_ready(arm, args.ready_joints, keys):
            return False
        self.phase = "idle"
        if not self.verify_model(arm, planner):
            return False

        print("At the ready pose; planning ...")
        plan = planner.build(args.ranges, args.grid, args.per_point, args.tilt, args.roll)
        print_plan(plan, self.safety, speed)
        if not any(s.record for s in plan.stops):
            print("ERROR: no safe sample poses; widen the ranges or check the tool size",
                  file=sys.stderr)
            return False
        if not args.yes and not keys.wait_for_enter(
                f"Mount the {self.safety.tool_length:.0f} mm tool now if it isn't on.\n"
                f"Press Enter to RUN at up to {speed:.0f} mm/s ({mvacc:.0f} mm/s^2), "
                "q to quit.  SPACE = e-stop."):
            print("Quit without running.")
            return False
        if not self.setup_tool_model(arm):
            return False
        if not args.no_mocap:
            self.save_config(plan, tcp_offset, speed, mvacc)
        return self.execute(arm, plan, mvacc, keys)


def _limit(arm, name: str, fallback: float) -> float:
    try:
        value = float(getattr(arm, name)[1])
        return value if value > 0 else fallback
    except Exception:
        return fallback


def dry_run(args: argparse.Namespace) -> int:
    safety = make_safety(args)
    reason = check_joints(args.ready_joints, safety)
    if reason:
        print(f"WARNING: the ready pose itself is unsafe with this tool: {reason}")
    speed = min(args.speed, FALLBACK_MAX_SPEED)
    planner = Planner(args.ready_joints, kin.pose_to_matrix(args.tcp_offset), safety,
                      speed, args.joint_speed)
    ready = kin.matrix_to_pose(planner.ready_tcp)
    print("Ready TCP pose: xyz=[{:.1f} {:.1f} {:.1f}] mm rpy=[{:.1f} {:.1f} {:.1f}] deg"
          .format(*ready))
    plan = planner.build(args.ranges, args.grid, args.per_point, args.tilt, args.roll)
    print_plan(plan, safety, speed)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_calibration_args(parser, rigid_optional=True)

    arm = parser.add_argument_group("arm")
    arm.add_argument("--arm-ip", default="192.168.1.103",
                     help="xArm controller IP (default: %(default)s)")
    arm.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                     help="max linear speed of the TCP and tool tip, mm/s; capped at the "
                          "arm's limit (default: %(default)s)")
    arm.add_argument("--mvacc", type=float, default=DEFAULT_ACC,
                     help="linear acceleration in mm/s^2; the arm's max is 50000 but "
                          "that jerks a long tool hard (default: %(default)s)")
    arm.add_argument("--ready-joints", type=float, nargs=6, default=DEFAULT_READY_JOINTS,
                     metavar="DEG", help="ready pose joint angles (default: %(default)s)")
    arm.add_argument("--joint-speed", type=float, default=60.0,
                     help="joint speed for fallback joint-space moves in the pattern, "
                          "deg/s; the tool tip moves much faster than the TCP in these "
                          "(default: %(default)s)")
    arm.add_argument("--ready-speed", type=float, default=30.0,
                     help="joint speed for the move to the ready pose, deg/s "
                          "(default: %(default)s)")

    pattern = parser.add_argument_group("pattern (offsets relative to the ready TCP position)")
    pattern.add_argument("--x-range", type=float, nargs=2, default=[-50.0, 150.0],
                         metavar=("MIN", "MAX"), help="mm (default: %(default)s)")
    pattern.add_argument("--y-range", type=float, nargs=2, default=[-150.0, 150.0],
                         metavar=("MIN", "MAX"), help="mm (default: %(default)s)")
    pattern.add_argument("--z-range", type=float, nargs=2, default=[-100.0, 150.0],
                         metavar=("MIN", "MAX"), help="mm (default: %(default)s)")
    pattern.add_argument("--grid", type=int, default=3,
                         help="points per axis (default: %(default)s)")
    pattern.add_argument("--per-point", type=int, default=2,
                         help="orientations recorded at each grid point (default: %(default)s)")
    pattern.add_argument("--tilt", type=float, default=20.0,
                         help="tool tilt up/down/left/right, deg (default: %(default)s)")
    pattern.add_argument("--roll", type=float, default=45.0,
                         help="roll about the tool axis, deg (default: %(default)s)")

    tool = parser.add_argument_group("tool and collision safety")
    tool.add_argument("--tool-length", type=float, required=True,
                      help="how far the tool sticks out from the flange face along "
                           "the flange Z axis, mm (0 if no tool is mounted)")
    tool.add_argument("--tool-radius", type=float, default=25.0,
                      help="tool radius incl. markers, mm (default: %(default)s)")
    tool.add_argument("--table-z", type=float, default=0.0,
                      help="table height in the arm base frame, mm (default: %(default)s)")
    tool.add_argument("--table-margin", type=float, default=30.0,
                      help="min clearance above the table, mm (default: %(default)s)")
    tool.add_argument("--link-radius", type=float, default=60.0,
                      help="arm link radius used for clearance checks, mm "
                           "(default: %(default)s)")
    tool.add_argument("--self-margin", type=float, default=30.0,
                      help="min tool-to-arm clearance, mm (default: %(default)s)")
    tool.add_argument("--max-wrist-rotation", type=float, default=120.0,
                      help="max J4/J6 turn away from the ready pose, deg; keeps the arm "
                           "camera/tool cables from wrapping (default: %(default)s)")
    tool.add_argument("--base-margin", type=float, default=130.0,
                      help="min distance from the forearm/wrist to the base column "
                           "axis, mm; the controller's self-collision stop tripped at "
                           "88 mm (default: %(default)s)")
    tool.add_argument("--min-z", default=None, metavar="MM|ready",
                      help="z floor (arm base frame, mm) for the flange centre and tool "
                           "tip during the pattern; 'ready' = as low as they are at the "
                           "ready pose (default: no floor)")
    tool.add_argument("--no-controller-collision", action="store_true",
                      help="don't configure the controller's self-collision detection")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--dry-run", action="store_true",
                      help="print the plan without connecting to anything")
    mode.add_argument("--no-mocap", action="store_true",
                      help="arm-only test: run the moves without Motive and record "
                           "nothing (no rigid_id needed)")
    mode.add_argument("--tcp-offset", type=float, nargs=6, default=[0.0] * 6,
                      metavar="V", help="TCP offset assumed by --dry-run (the real run "
                                        "reads it from the arm)")
    mode.add_argument("-y", "--yes", action="store_true",
                      help="skip the Enter prompts before the move to ready and before "
                           "the pattern (a pre-existing arm error still asks)")

    args = parser.parse_args()
    args.ranges = [args.x_range, args.y_range, args.z_range]
    for name, (lo, hi) in zip("xyz", args.ranges):
        if lo > hi:
            parser.error(f"--{name}-range MIN must be <= MAX")
    if args.grid < 1 or args.per_point < 1:
        parser.error("--grid and --per-point must be >= 1")
    if not 0 <= args.tilt <= 45:
        parser.error("--tilt must be between 0 and 45 deg")
    if not 0 < args.joint_speed <= MAX_JOINT_SPEED:
        parser.error(f"--joint-speed must be in (0, {MAX_JOINT_SPEED:g}]")
    if args.tool_length < 0 or args.tool_radius < 0:
        parser.error("tool dimensions must be >= 0")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    if not 0 < args.max_wrist_rotation <= 360:
        parser.error("--max-wrist-rotation must be in (0, 360]")
    if args.min_z not in (None, "ready"):
        try:
            float(args.min_z)
        except ValueError:
            parser.error("--min-z must be a number (mm) or 'ready'")
    if args.rigid_id is None and not (args.dry_run or args.no_mocap):
        parser.error("rigid_id is required unless --dry-run or --no-mocap")
    return args


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)
    if not sys.stdin.isatty():
        print("ERROR: needs an interactive terminal (SPACE is the e-stop)", file=sys.stderr)
        return 1
    return AutoCalibrator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
