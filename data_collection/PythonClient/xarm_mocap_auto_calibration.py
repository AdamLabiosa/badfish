#!/usr/bin/env python3
"""Automatically drive a UFactory xArm 6 through a calibration pattern.

Same output as xarm_mocap_calibration.py (arm pose + averaged mocap pose per
sample), but the arm moves itself instead of being jogged by hand.

  1. On launch the arm moves (joint move, slow) to a "ready" pose: the xArm
     home pose with the wrist bent so the tool points forward (+X) instead of
     down.  It then plans the pattern, prints it, and waits: mount the tool if
     it isn't on yet, then press Enter to run (q quits).
     The default is joints [0, 0, -20, 180, 70, 180]; override with
     --ready-joints.  (Tool points forward whenever J4 = J6 = 180 and
     J5 = J2 + J3 + 90, or J4 = J6 = 0 and J2 + J3 + J5 = -90.)
  2. From there it visits a grid of positions inside a box defined *relative to
     the ready TCP position* (--x-range/--y-range/--z-range, in mm), tilting
     and rolling the tool at each point so the data has rotation about every
     axis.  These moves are linear (Cartesian) moves at the arm's max speed.
  3. At each pose it waits --settle seconds and records a sample.
  4. It returns to the ready pose.

E-STOP: press SPACE (or Ctrl+C) at any time once the arm is connected; the arm
stops immediately and the samples recorded so far are saved.  This is a
software stop over the network -- keep the hardware e-stop within reach.

Every move is checked *before the arm moves at all*: the whole route is
simulated in small steps with an xArm 6 kinematic model and a model of the
tool (a cylinder of --tool-length x --tool-radius along the flange Z axis).
Poses that would put the tool or arm below the table, bring the tool close to
the arm, hit joint limits or pass near a singularity are skipped.  The
controller's own self-collision detection is also enabled with the same
cylinder tool model.

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
from xarm_keyboard_control import Keyboard
from xarm_mocap_calibration import Calibrator, add_calibration_args


# Home pose with the elbow raised 20 deg and the wrist flipped (J4 = J6 = 180)
# and bent so the tool points forward (+X).  Same tool orientation as the literal
# "home but tool forward" [0, 0, 0, 0, -90, 0], but that one sits within a few
# degrees of the J3 and J5 limits, so most of the pattern is unreachable from
# it.  Here J5 = 70 keeps the wrist far from both its limits and its
# singularity (J5 = 0), so every move in the default pattern is a straight line.
DEFAULT_READY_JOINTS = [0.0, 0.0, -20.0, 180.0, 70.0, 180.0]
FALLBACK_MAX_SPEED = 1000.0    # mm/s, xArm 6 spec
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


def check_joints(joints: Sequence[float], safety: Safety) -> Optional[str]:
    """Reason the configuration is unsafe, or None."""
    if not kin.within_limits(joints, safety.joint_margin):
        return "joint limit"
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
        speed = min(self.speed, JOINT_SPEED_USE * MAX_JOINT_SPEED / max(max_ratio, 1e-9))
        if speed < MIN_LINEAR_SPEED:
            return None, "near singularity"
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
        steps = max(1, math.ceil(float(np.max(np.abs(end_q - start_q))) / 1.0))
        path = [start_q + (end_q - start_q) * i / steps for i in range(1, steps + 1)]
        for q in path:
            reason = check_joints(q, self.safety)
            if reason:
                return None, reason
        for q in path:
            table, self_clear = clearances(q, self.safety)
            plan.min_table = min(plan.min_table, table)
            plan.min_self = min(plan.min_self, self_clear)
        plan.path_mm += float(np.linalg.norm(target[:3, 3] - self.tcp(start_q)[:3, 3]))
        return Segment(target, path, "joint", self.joint_speed), ""

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
    if any(s.kind == "joint" for s in segments):
        notes.append("joint move")
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


class AutoCalibrator(Calibrator):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.safety = Safety(args.tool_length, args.tool_radius, args.table_z,
                             args.table_margin, args.link_radius, args.self_margin)

    def connect_arm(self):
        from xarm_keyboard_control import ArmJogger  # imports the xArm SDK
        self.jogger = ArmJogger(self.args.arm_ip, 0.0, 0.0, 0)
        arm = self.jogger.arm
        speed = self.args.speed or _limit(arm, "tcp_speed_limit", FALLBACK_MAX_SPEED)
        mvacc = min(self.args.mvacc, _limit(arm, "tcp_acc_limit", self.args.mvacc))
        return arm, speed, mvacc

    def setup_tool_model(self, arm) -> None:
        if self.args.no_controller_collision:
            return
        try:
            if self.safety.tool_length > 0:
                code = arm.set_collision_tool_model(
                    CYLINDER_TOOL_MODEL, radius=self.safety.tool_radius,
                    height=self.safety.tool_length)
            else:
                code = arm.set_collision_tool_model(0)
            code2 = arm.set_self_collision_detection(True)
        except Exception as exc:  # older SDK/firmware
            print(f"  ! could not set controller self-collision model: {exc}")
            return
        if code != 0 or code2 != 0:
            print(f"  ! controller self-collision setup returned codes {code}, {code2}")
        else:
            print(f"Controller self-collision detection on (cylinder tool "
                  f"r={self.safety.tool_radius:.0f} mm, h={self.safety.tool_length:.0f} mm)")

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
        reason = check_joint_move(joints, ready_joints, self.safety)
        if reason:
            bare = replace(self.safety, tool_length=0.0, tool_radius=0.0)
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
                print(f"[{stops.index(stop) + 1}/{len(stops)}] {stop.label}")
            else:
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
            if stop.record:
                self.record_sample()
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
        if not self.connect_mocap(require_body=True):
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
        finally:
            if keys is not None:
                keys.stop()
            self.client.shutdown()
            if self.jogger is not None:
                self.jogger.disconnect()

        if self.samples:
            print(f"Saved {len(self.samples)} samples to {self.csv_path}")
        else:
            print("No samples recorded.")
        return 0 if ok else 1

    def session(self, arm, speed: float, mvacc: float, keys: KeyMonitor) -> bool:
        args = self.args
        tcp_offset = list(arm.tcp_offset[:6])
        if any(abs(v) > 1e-6 for v in tcp_offset):
            print(f"Using controller TCP offset {tcp_offset}")
        planner = Planner(args.ready_joints, kin.pose_to_matrix(tcp_offset), self.safety,
                          speed, args.joint_speed)
        if not self.goto_ready(arm, args.ready_joints, keys):
            return False
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
        self.setup_tool_model(arm)
        self.save_config(plan, tcp_offset, speed, mvacc)
        return self.execute(arm, plan, mvacc, keys)


def _limit(arm, name: str, fallback: float) -> float:
    try:
        value = float(getattr(arm, name)[1])
        return value if value > 0 else fallback
    except Exception:
        return fallback


def dry_run(args: argparse.Namespace) -> int:
    safety = Safety(args.tool_length, args.tool_radius, args.table_z, args.table_margin,
                    args.link_radius, args.self_margin)
    reason = check_joints(args.ready_joints, safety)
    if reason:
        print(f"WARNING: the ready pose itself is unsafe with this tool: {reason}")
    speed = args.speed or FALLBACK_MAX_SPEED
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
    arm.add_argument("--speed", type=float,
                     help="linear speed in mm/s (default: the arm's max)")
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
    tool.add_argument("--no-controller-collision", action="store_true",
                      help="don't configure the controller's self-collision detection")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--dry-run", action="store_true",
                      help="print the plan without connecting to anything")
    mode.add_argument("--tcp-offset", type=float, nargs=6, default=[0.0] * 6,
                      metavar="V", help="TCP offset assumed by --dry-run (the real run "
                                        "reads it from the arm)")
    mode.add_argument("-y", "--yes", action="store_true",
                      help="start the pattern without waiting for Enter at the ready pose")

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
    if args.rigid_id is None and not args.dry_run:
        parser.error("rigid_id is required unless --dry-run")
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
