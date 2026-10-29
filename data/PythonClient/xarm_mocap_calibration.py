#!/usr/bin/env python3
"""Jog a UFactory xArm 6 from the keyboard and record arm/mocap pose pairs.

Attach a Motive rigid body to the arm's end effector, jog the arm to a
variety of positions *and orientations*, and press space at each one.  Every
sample stores the arm's reported TCP pose and joint angles alongside the
rigid-body pose averaged over several mocap frames.  The resulting CSV is the
input for solving the mocap <-> arm-base transform (plus the flange <-> marker
offset, since the rigid-body pivot is generally not at the TCP).

Moves are discrete, blocking steps.  Holding a key repeats the step; queued
key presses are discarded after each move so the arm never runs ahead of you.
Ctrl+C stops the arm and exits.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque, Iterator, List, Optional, Tuple

try:
    from xarm.wrapper import XArmAPI
except ImportError as exc:
    raise SystemExit("xArm SDK is required: python -m pip install xarm-python-sdk") from exc

from NatNetClient import NatNetClient


# Calibration data lives in data/calibration/<MM-DD-YYYY>/, independent of the cwd.
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"

LINEAR_STEPS_MM = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
ANGULAR_STEPS_DEG = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

# key -> (x, y, z, roll, pitch, yaw) unit direction in the arm base frame
JOG_KEYS = {
    "w": (1, 0, 0, 0, 0, 0), "s": (-1, 0, 0, 0, 0, 0),
    "a": (0, 1, 0, 0, 0, 0), "d": (0, -1, 0, 0, 0, 0),
    "r": (0, 0, 1, 0, 0, 0), "f": (0, 0, -1, 0, 0, 0),
    "u": (0, 0, 0, 1, 0, 0), "o": (0, 0, 0, -1, 0, 0),
    "i": (0, 0, 0, 0, 1, 0), "k": (0, 0, 0, 0, -1, 0),
    "j": (0, 0, 0, 0, 0, 1), "l": (0, 0, 0, 0, 0, -1),
}

HELP = """
  Translate (base frame)      Rotate (base frame)
    w / s   +X / -X             u / o   +roll  / -roll
    a / d   +Y / -Y             i / k   +pitch / -pitch
    r / f   +Z / -Z             j / l   +yaw   / -yaw

    - / =   smaller / larger step     space   record sample
    p       print current pose        x       delete last sample
    c       clear arm error/warning   h       show this help
    q       save and quit             Ctrl+C  stop arm and quit
"""

CSV_HEADER = [
    "sample", "time_s",
    "arm_x_mm", "arm_y_mm", "arm_z_mm", "arm_roll_deg", "arm_pitch_deg", "arm_yaw_deg",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "rigid_id", "mocap_x_m", "mocap_y_m", "mocap_z_m",
    "mocap_qx", "mocap_qy", "mocap_qz", "mocap_qw",
    "mocap_pos_std_m", "mocap_frames", "mocap_mean_error_m",
]


@contextmanager
def cbreak_terminal() -> Iterator[int]:
    """Deliver key presses immediately (no Enter) while keeping Ctrl+C."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(fd: int, timeout: float) -> Optional[str]:
    readable, _, _ = select.select([fd], [], [], timeout)
    if not readable:
        return None
    data = os.read(fd, 32)
    # Ignore escape sequences (arrow keys etc.) rather than misreading them.
    if not data or data.startswith(b"\x1b"):
        return None
    return data[:1].decode(errors="ignore")


def average_quaternion(quats: List[Tuple[float, float, float, float]]) -> Tuple[float, ...]:
    """Sign-aligned mean; accurate for the tightly clustered samples of a still body."""
    ref = quats[0]
    total = [0.0, 0.0, 0.0, 0.0]
    for q in quats:
        sign = 1.0 if sum(a * b for a, b in zip(q, ref)) >= 0 else -1.0
        for i in range(4):
            total[i] += sign * q[i]
    norm = math.sqrt(sum(v * v for v in total))
    return tuple(v / norm for v in total)


class Calibrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.step_index = args.step_index
        self.samples: List[list] = []
        self.clock_zero = time.monotonic()

        self.mocap_lock = threading.Lock()
        self.mocap_frames: Deque[tuple] = deque(maxlen=1000)
        self.last_mocap_time = 0.0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = args.output_dir / f"{args.name}.csv"

        self.client = NatNetClient()
        self.client.set_client_address(args.client_ip)
        self.client.set_server_address(args.server_ip)
        self.client.set_use_multicast(not args.unicast)
        self.client.set_print_level(0)
        self.client.new_frame_with_data_listener = self.on_mocap_frame

        self.arm: Optional[XArmAPI] = None

    # ----------------------------------------------------------------- mocap
    def on_mocap_frame(self, data: dict) -> None:
        """NatNet callback; invoked after a complete mocap frame is decoded."""
        rigid_bodies = data["mocap_data"].rigid_body_data.rigid_body_list
        body = next((rb for rb in rigid_bodies if rb.id_num == self.args.rigid_id), None)
        if body is None:
            return
        now = time.monotonic()
        with self.mocap_lock:
            self.last_mocap_time = now
            if body.tracking_valid:
                self.mocap_frames.append((now, tuple(body.pos), tuple(body.rot), body.error))

    def collect_mocap(self) -> Optional[tuple]:
        """Average the next N valid rigid-body frames; None if tracking is lost."""
        start = time.monotonic()
        deadline = start + self.args.mocap_timeout
        frames: list = []
        while time.monotonic() < deadline:
            with self.mocap_lock:
                frames = [f for f in self.mocap_frames if f[0] >= start]
            if len(frames) >= self.args.mocap_frames:
                break
            time.sleep(0.01)
        frames = frames[: self.args.mocap_frames]
        if len(frames) < max(1, self.args.mocap_frames // 2):
            return None

        n = len(frames)
        mean_pos = [sum(f[1][i] for f in frames) / n for i in range(3)]
        pos_std = math.sqrt(
            sum(sum((f[1][i] - mean_pos[i]) ** 2 for i in range(3)) for f in frames) / n
        )
        quat = average_quaternion([f[2] for f in frames])
        mean_error = sum(f[3] for f in frames) / n
        return mean_pos, quat, pos_std, n, mean_error

    # ------------------------------------------------------------------- arm
    def connect_arm(self) -> None:
        print(f"Connecting to xArm at {self.args.arm_ip} ...")
        self.arm = XArmAPI(self.args.arm_ip, is_radian=False)
        if not self.arm.connected:
            raise SystemExit(f"Could not connect to xArm at {self.args.arm_ip}")
        self.enable_arm()

    def enable_arm(self) -> None:
        arm = self.arm
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)  # position control
        arm.set_state(state=0)
        time.sleep(0.5)
        print(f"Arm state={arm.state} error={arm.error_code} warn={arm.warn_code}")

    def arm_pose(self) -> Optional[Tuple[List[float], List[float]]]:
        code, pose = self.arm.get_position(is_radian=False)
        code2, joints = self.arm.get_servo_angle(is_radian=False)
        if code != 0 or code2 != 0:
            print(f"  ! could not read arm pose (codes {code}, {code2})")
            return None
        return list(pose[:6]), list(joints[:6])

    def jog(self, direction: Tuple[int, ...]) -> None:
        lin = LINEAR_STEPS_MM[self.step_index]
        ang = ANGULAR_STEPS_DEG[self.step_index]
        dx, dy, dz, droll, dpitch, dyaw = direction
        code = self.arm.set_position(
            x=dx * lin, y=dy * lin, z=dz * lin,
            roll=droll * ang, pitch=dpitch * ang, yaw=dyaw * ang,
            relative=True, speed=self.args.speed, mvacc=self.args.mvacc, wait=True,
        )
        if code != 0 or self.arm.error_code != 0:
            print(f"  ! move failed: code={code} error={self.arm.error_code} "
                  f"warn={self.arm.warn_code} (press c to clear)")
        else:
            self.print_pose()

    def print_pose(self) -> None:
        result = self.arm_pose()
        if result is None:
            return
        pose, _ = result
        with self.mocap_lock:
            mocap_age = time.monotonic() - self.last_mocap_time
        mocap = "ok" if mocap_age < 0.5 else "LOST"
        print("  xyz=[{:8.1f} {:8.1f} {:8.1f}] mm  rpy=[{:7.1f} {:7.1f} {:7.1f}] deg"
              "  mocap={}".format(*pose, mocap))

    # --------------------------------------------------------------- samples
    def record_sample(self) -> None:
        time.sleep(self.args.settle)  # let residual vibration die out
        mocap = self.collect_mocap()
        result = self.arm_pose()
        if mocap is None:
            print("  ! no valid mocap data for rigid body "
                  f"{self.args.rigid_id}; sample not recorded")
            return
        if result is None:
            return
        pose, joints = result
        mean_pos, quat, pos_std, n, mean_error = mocap
        row = [
            len(self.samples), round(time.monotonic() - self.clock_zero, 3),
            *pose, *joints, self.args.rigid_id, *mean_pos, *quat, pos_std, n, mean_error,
        ]
        self.samples.append(row)
        self.save()
        warn = "  (noisy! arm may still be moving)" if pos_std > self.args.max_std else ""
        print(f"  + sample {row[0]} recorded ({n} frames, std {pos_std * 1000:.2f} mm)"
              f"{warn}")

    def delete_sample(self) -> None:
        if not self.samples:
            print("  no samples to delete")
            return
        row = self.samples.pop()
        self.save()
        print(f"  - deleted sample {row[0]}; {len(self.samples)} remain")

    def save(self) -> None:
        # Rewrite the whole file so deletions persist and a crash loses nothing.
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for i, row in enumerate(self.samples):
                writer.writerow([i, *row[1:]])
        tmp.replace(self.csv_path)

    def print_step(self) -> None:
        print(f"  step: {LINEAR_STEPS_MM[self.step_index]} mm / "
              f"{ANGULAR_STEPS_DEG[self.step_index]} deg")

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        if not self.client.run("d"):
            print("ERROR: Could not start the NatNet client", file=sys.stderr)
            return 1
        time.sleep(1.0)
        if not self.client.connected():
            self.client.shutdown()
            print("ERROR: Could not connect to Motive; check streaming and IP settings",
                  file=sys.stderr)
            return 1

        try:
            self.connect_arm()
            print(HELP)
            self.print_step()
            self.print_pose()
            with cbreak_terminal() as fd:
                while True:
                    key = read_key(fd, 0.1)
                    if key is None:
                        continue
                    key = key.lower()
                    if key in JOG_KEYS:
                        self.jog(JOG_KEYS[key])
                        # Drop keys queued (e.g. by key repeat) during the move.
                        termios.tcflush(fd, termios.TCIFLUSH)
                    elif key == " ":
                        self.record_sample()
                        termios.tcflush(fd, termios.TCIFLUSH)
                    elif key == "x":
                        self.delete_sample()
                    elif key == "-":
                        self.step_index = max(0, self.step_index - 1)
                        self.print_step()
                    elif key in "=+":
                        self.step_index = min(len(LINEAR_STEPS_MM) - 1, self.step_index + 1)
                        self.print_step()
                    elif key == "p":
                        self.print_pose()
                    elif key == "c":
                        self.enable_arm()
                    elif key == "h":
                        print(HELP)
                    elif key == "q":
                        break
        except KeyboardInterrupt:
            print("\nInterrupted; stopping arm.")
            if self.arm is not None:
                self.arm.set_state(state=4)
        finally:
            self.client.shutdown()
            if self.arm is not None:
                self.arm.disconnect()

        if self.samples:
            print(f"Saved {len(self.samples)} samples to {self.csv_path}")
        else:
            print("No samples recorded.")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rigid_id", type=int,
                        help="Motive streaming ID of the end-effector rigid body")
    parser.add_argument("--arm-ip", required=True, help="xArm controller IP")
    parser.add_argument("--server-ip", default="192.168.31.51", help="Motive/NatNet server IP")
    parser.add_argument("--client-ip", default="192.168.31.50", help="local network-interface IP")
    parser.add_argument("--unicast", action="store_true", help="use unicast instead of multicast")
    parser.add_argument("--speed", type=float, default=50.0,
                        help="TCP jog speed in mm/s (default: %(default)s)")
    parser.add_argument("--mvacc", type=float, default=500.0,
                        help="TCP jog acceleration in mm/s^2 (default: %(default)s)")
    parser.add_argument("--step-index", type=int, default=3,
                        choices=range(len(LINEAR_STEPS_MM)),
                        help="initial step size index into "
                             f"{LINEAR_STEPS_MM} mm / {ANGULAR_STEPS_DEG} deg "
                             "(default: %(default)s)")
    parser.add_argument("--settle", type=float, default=0.5,
                        help="seconds to wait before sampling (default: %(default)s)")
    parser.add_argument("--mocap-frames", type=int, default=60,
                        help="mocap frames averaged per sample (default: %(default)s)")
    parser.add_argument("--mocap-timeout", type=float, default=3.0,
                        help="max seconds to wait for mocap frames (default: %(default)s)")
    parser.add_argument("--max-std", type=float, default=0.0005,
                        help="warn if sample position std exceeds this, in m "
                             "(default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=CALIBRATION_DIR / time.strftime("%m-%d-%Y"),
                        help="output folder (default: %(default)s)")
    parser.add_argument("--name", default=time.strftime("xarm_mocap_%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    if not sys.stdin.isatty():
        print("ERROR: keyboard control requires an interactive terminal", file=sys.stderr)
        return 1
    return Calibrator(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
