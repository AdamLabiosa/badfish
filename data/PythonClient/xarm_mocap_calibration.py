#!/usr/bin/env python3
"""Jog a UFactory xArm 6 from the keyboard and record arm/mocap pose pairs.

Attach a Motive rigid body to the arm's end effector, jog the arm to a
variety of positions *and orientations*, and press space at each one.  Every
sample stores the arm's reported TCP pose and joint angles alongside the
rigid-body pose averaged over several mocap frames.  The resulting CSV is the
input for solving the mocap <-> arm-base transform (plus the flange <-> marker
offset, since the rigid-body pivot is generally not at the TCP).

Jogging works as in xarm_keyboard_control.py.  Ctrl+C stops the arm and exits.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import termios
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from NatNetClient import NatNetClient
from xarm_keyboard_control import (
    ArmJogger, JOG_HELP, JOG_KEYS, POSE_FORMAT, add_arm_args, cbreak_terminal, read_key,
)


# Calibration data lives in data/calibration/<MM-DD-YYYY>/, independent of the cwd.
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"

HELP = JOG_HELP + """\
    space   record sample             x       delete last sample
    (q saves and quits)
"""

CSV_HEADER = [
    "sample", "time_s",
    "arm_x_mm", "arm_y_mm", "arm_z_mm", "arm_roll_deg", "arm_pitch_deg", "arm_yaw_deg",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "rigid_id", "mocap_x_m", "mocap_y_m", "mocap_z_m",
    "mocap_qx", "mocap_qy", "mocap_qz", "mocap_qw",
    "mocap_pos_std_m", "mocap_frames", "mocap_mean_error_m",
]


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

        self.jogger: Optional[ArmJogger] = None

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

    def print_pose(self) -> None:
        result = self.jogger.pose()
        if result is None:
            return
        with self.mocap_lock:
            mocap_age = time.monotonic() - self.last_mocap_time
        mocap = "ok" if mocap_age < 0.5 else "LOST"
        print("  " + POSE_FORMAT.format(*result[0]) + f"  mocap={mocap}")

    # --------------------------------------------------------------- samples
    def record_sample(self) -> None:
        time.sleep(self.args.settle)  # let residual vibration die out
        mocap = self.collect_mocap()
        result = self.jogger.pose()
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

        args = self.args
        try:
            self.jogger = ArmJogger(args.arm_ip, args.speed, args.mvacc, args.step_index)
            print(HELP)
            self.jogger.print_step()
            self.print_pose()
            with cbreak_terminal() as fd:
                while True:
                    key = read_key(fd, 0.1)
                    if key is None:
                        continue
                    if self.jogger.handle_key(key, fd):
                        if key in JOG_KEYS:
                            self.print_pose()
                    elif key == " ":
                        self.record_sample()
                        termios.tcflush(fd, termios.TCIFLUSH)
                    elif key == "x":
                        self.delete_sample()
                    elif key == "p":
                        self.print_pose()
                    elif key == "h":
                        print(HELP)
                    elif key == "q":
                        break
        except KeyboardInterrupt:
            print("\nInterrupted; stopping arm.")
            if self.jogger is not None:
                self.jogger.stop()
        finally:
            self.client.shutdown()
            if self.jogger is not None:
                self.jogger.disconnect()

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
    add_arm_args(parser)
    parser.add_argument("--server-ip", default="192.168.31.51", help="Motive/NatNet server IP")
    parser.add_argument("--client-ip", default="192.168.31.50", help="local network-interface IP")
    parser.add_argument("--unicast", action="store_true", help="use unicast instead of multicast")
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
