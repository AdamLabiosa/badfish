#!/usr/bin/env python3
"""Jog a UFactory xArm 6 from the keyboard and print the end-effector pose.

Moves are discrete, blocking steps.  Holding a key repeats the step; queued
key presses are discarded after each move so the arm never runs ahead of you.
The TCP pose is printed every --interval seconds.  Ctrl+C stops the arm and
exits.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty

try:
    from xarm.wrapper import XArmAPI
except ImportError as exc:
    raise SystemExit("xArm SDK is required: python -m pip install xarm-python-sdk") from exc


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

JOG_HELP = """
  Translate (base frame)      Rotate (base frame)
    w / s   +X / -X             u / o   +roll  / -roll
    a / d   +Y / -Y             i / k   +pitch / -pitch
    r / f   +Z / -Z             j / l   +yaw   / -yaw

    - / =   smaller / larger step     p       print current pose
    c       clear arm error/warning   h       show this help
    q       quit                      Ctrl+C  stop arm and quit
"""

POSE_FORMAT = "xyz=[{:8.1f} {:8.1f} {:8.1f}] mm  rpy=[{:7.1f} {:7.1f} {:7.1f}] deg"


class Keyboard:
    """Single key presses without Enter, on Windows and macOS/Linux.

    Use as a context manager; Ctrl+C still raises KeyboardInterrupt.
    """

    def __enter__(self) -> "Keyboard":
        if os.name != "nt":
            self.fd = sys.stdin.fileno()
            self.old_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if os.name != "nt":
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)

    def read_key(self, timeout: float) -> Optional[str]:
        """Next key as a lower-case character, or None on timeout/special key."""
        if os.name == "nt":
            deadline = time.monotonic() + timeout
            while not msvcrt.kbhit():
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.01)
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow/function key prefix
                msvcrt.getwch()
                return None
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch.lower()

        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return None
        data = os.read(self.fd, 32)
        # Ignore escape sequences (arrow keys etc.) rather than misreading them.
        if not data or data.startswith(b"\x1b"):
            return None
        return data[:1].decode(errors="ignore").lower()

    def flush(self) -> None:
        """Discard pending key presses (e.g. key repeat during a move)."""
        if os.name == "nt":
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            termios.tcflush(self.fd, termios.TCIFLUSH)


class ArmJogger:
    """Step-wise Cartesian jogging of an xArm in position-control mode."""

    def __init__(self, ip: str, speed: float, mvacc: float, step_index: int) -> None:
        self.speed = speed
        self.mvacc = mvacc
        self.step_index = step_index
        print(f"Connecting to xArm at {ip} ...")
        self.arm = XArmAPI(ip, is_radian=False)
        if not self.arm.connected:
            raise SystemExit(f"Could not connect to xArm at {ip}")
        self.enable()

    def enable(self) -> None:
        arm = self.arm
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)  # position control
        arm.set_state(state=0)
        time.sleep(0.5)
        print(f"Arm state={arm.state} error={arm.error_code} warn={arm.warn_code}")

    def pose(self) -> Optional[Tuple[List[float], List[float]]]:
        """TCP pose [x, y, z, roll, pitch, yaw] (mm, deg) and joint angles (deg)."""
        code, pose = self.arm.get_position(is_radian=False)
        code2, joints = self.arm.get_servo_angle(is_radian=False)
        if code != 0 or code2 != 0:
            print(f"  ! could not read arm pose (codes {code}, {code2})")
            return None
        return list(pose[:6]), list(joints[:6])

    def jog(self, direction: Tuple[int, ...]) -> bool:
        lin = LINEAR_STEPS_MM[self.step_index]
        ang = ANGULAR_STEPS_DEG[self.step_index]
        dx, dy, dz, droll, dpitch, dyaw = direction
        code = self.arm.set_position(
            x=dx * lin, y=dy * lin, z=dz * lin,
            roll=droll * ang, pitch=dpitch * ang, yaw=dyaw * ang,
            relative=True, speed=self.speed, mvacc=self.mvacc, wait=True,
        )
        if code != 0 or self.arm.error_code != 0:
            print(f"  ! move failed: code={code} error={self.arm.error_code} "
                  f"warn={self.arm.warn_code} (press c to clear)")
            return False
        return True

    def change_step(self, delta: int) -> None:
        self.step_index = min(max(self.step_index + delta, 0), len(LINEAR_STEPS_MM) - 1)
        self.print_step()

    def print_step(self) -> None:
        print(f"  step: {LINEAR_STEPS_MM[self.step_index]} mm / "
              f"{ANGULAR_STEPS_DEG[self.step_index]} deg")

    def handle_key(self, key: str, keyboard: Keyboard) -> bool:
        """Handle a shared jogging key; returns False if the key is not one."""
        if key in JOG_KEYS:
            self.jog(JOG_KEYS[key])
            # Drop keys queued (e.g. by key repeat) during the move.
            keyboard.flush()
        elif key == "-":
            self.change_step(-1)
        elif key in "=+":
            self.change_step(1)
        elif key == "c":
            self.enable()
        else:
            return False
        return True

    def stop(self) -> None:
        self.arm.set_state(state=4)

    def disconnect(self) -> None:
        self.arm.disconnect()


def add_arm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm-ip", default="192.168.1.103",
                        help="xArm controller IP (default: %(default)s)")
    parser.add_argument("--speed", type=float, default=50.0,
                        help="TCP jog speed in mm/s (default: %(default)s)")
    parser.add_argument("--mvacc", type=float, default=500.0,
                        help="TCP jog acceleration in mm/s^2 (default: %(default)s)")
    parser.add_argument("--step-index", type=int, default=3,
                        choices=range(len(LINEAR_STEPS_MM)),
                        help="initial step size index into "
                             f"{LINEAR_STEPS_MM} mm / {ANGULAR_STEPS_DEG} deg "
                             "(default: %(default)s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arm_args(parser)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between pose printouts (default: %(default)s)")
    return parser.parse_args()


def main() -> int:
    if not sys.stdin.isatty():
        print("ERROR: keyboard control requires an interactive terminal", file=sys.stderr)
        return 1
    args = parse_args()
    jogger = ArmJogger(args.arm_ip, args.speed, args.mvacc, args.step_index)
    stop_event = threading.Event()

    def print_pose() -> None:
        result = jogger.pose()
        if result is not None:
            print("  " + POSE_FORMAT.format(*result[0]))

    def report_loop() -> None:
        # Runs on its own thread so the pose keeps printing during blocking moves.
        while not stop_event.wait(args.interval):
            print_pose()

    reporter = threading.Thread(target=report_loop, name="pose-report", daemon=True)
    try:
        print(JOG_HELP)
        jogger.print_step()
        print_pose()
        reporter.start()
        with Keyboard() as keyboard:
            while True:
                key = keyboard.read_key(0.1)
                if key is None or jogger.handle_key(key, keyboard):
                    continue
                if key == "p":
                    print_pose()
                elif key == "h":
                    print(JOG_HELP)
                elif key == "q":
                    break
    except KeyboardInterrupt:
        print("\nInterrupted; stopping arm.")
        jogger.stop()
    finally:
        stop_event.set()
        jogger.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
