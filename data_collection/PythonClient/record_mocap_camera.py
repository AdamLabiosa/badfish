#!/usr/bin/env python3
"""Record a hand-held end-effector demonstration: mocap rigid body + camera.

Hold the end effector (with its Motive markers, rigid body 7 by default) and
move it through the motion you want the arm to replay with
rollout_trajectory.py.  The CSV and camera use the same monotonic clock; each
mocap sample includes the index and time of the nearest video frame, so the
data can be aligned with the MP4 without relying on wall-clock time.

Flow:
  1. Connects to Motive and opens the camera.  A status line shows whether the
     rigid body is tracked and -- using the newest arm<->mocap calibration --
     where the flange would be in the arm base frame and whether the arm could
     reach it (IK + joint limits + table/tool clearance).
  2. Press Enter, then a short countdown (--countdown) gives you time to pick
     the end effector up.  HOLD IT STILL for a moment at the start and end:
     the rollout starts and ends from rest, and trims still time automatically.
  3. Press Enter (or Ctrl+C) to stop.

Outputs in recordings/<MM-DD-YYYY>/:
  <name>.csv        every mocap frame of the rigid body (raw Motive pose) plus
                    the calibrated flange pose in the arm base frame
  <name>.mp4        camera video, constant FPS, same start/end as the CSV
  <name>_meta.json  settings, calibration file used, frame counts

    python record_mocap_camera.py --list-cameras
    python record_mocap_camera.py                    # rigid body 7, camera 0
    python record_mocap_camera.py 7 --camera 0 --name pick_cup_01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV is required: python -m pip install opencv-python") from exc

import xarm6_kinematics as kin
from mocap_to_arm import RECORDINGS_DIR, MocapCalibration, load_calibration
from NatNetClient import NatNetClient
from xarm_mocap_auto_calibration import DEFAULT_READY_JOINTS, Safety, check_joints
from xarm_mocap_calibration import DEFAULT_RIGID_ID, MOCAP_SERVER_IP, resolve_client_ip


CameraSource = Union[int, str]

BACKENDS = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

# How long the camera may go without delivering a frame before we give up.
CAMERA_STALL_S = 2.0
STATUS_PERIOD_S = 0.25

CSV_HEADER = [
    "mocap_frame", "natnet_timestamp_s", "capture_time_s",
    "rigid_id", "x_m", "y_m", "z_m",
    "qx", "qy", "qz", "qw", "tracking_valid", "mean_error_m",
    "video_frame", "video_frame_time_s", "sync_offset_s",
    # Calibrated flange pose in the arm base frame (empty without a calibration).
    "flange_x_mm", "flange_y_mm", "flange_z_mm",
    "flange_roll_deg", "flange_pitch_deg", "flange_yaw_deg",
]


class ReachChecker:
    """Cheap live check: could the arm put its flange at this pose?"""

    def __init__(self, tool_length: float) -> None:
        self.safety = Safety(tool_length=tool_length, tool_radius=25.0, table_z=0.0,
                             table_margin=30.0, link_radius=60.0, self_margin=30.0)
        self.seed = np.array(DEFAULT_READY_JOINTS, float)

    def check(self, flange: np.ndarray) -> str:
        q = kin.inverse(flange, self.seed, iterations=60)
        if q is None:
            q = kin.inverse(flange, DEFAULT_READY_JOINTS, iterations=200)
        if q is None:
            self.seed = np.array(DEFAULT_READY_JOINTS, float)
            return "OUT OF REACH"
        self.seed = q
        reason = check_joints(q, self.safety)
        return "reachable" if reason is None else f"UNSAFE: {reason}"


class Recorder:
    def __init__(self, args: argparse.Namespace,
                 calibration: Optional[MocapCalibration]) -> None:
        self.args = args
        self.calibration = calibration
        self.reach = ReachChecker(args.tool_length) if calibration is not None else None
        self.stop_event = threading.Event()
        self.camera_ready = threading.Event()
        self.armed = threading.Event()             # user pressed Enter + countdown done
        self.recording_started = threading.Event()  # first sample written
        self.clock_zero_ns = time.monotonic_ns()
        self.camera_frames: Deque[Tuple[int, int]] = deque(maxlen=240)
        self.camera_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.camera_error: Optional[str] = None
        self.camera_info = ""
        self.video_fps = 0.0
        self.video_frames_written = 0
        self.latest_mocap_ns = 0
        self.camera_dropouts = 0
        self.rows_written = 0
        self.rows_invalid = 0
        self.show_status = False

        # Latest pose for the status line (written by the NatNet thread).
        self.latest_lock = threading.Lock()
        self.latest: Optional[tuple] = None   # (monotonic, valid, pos, quat)
        self.seen_rigid_ids: set = set()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = args.output_dir / f"{args.name}.csv"
        self.video_path = args.output_dir / f"{args.name}.mp4"
        self.meta_path = args.output_dir / f"{args.name}_meta.json"
        for path in (self.csv_path, self.video_path):
            if path.exists():
                raise SystemExit(f"{path} already exists; pick another --name")
        # Line-buffered so a crash loses at most one row.
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8", buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(CSV_HEADER)

        if args.client_ip is None:
            args.client_ip = resolve_client_ip(args.server_ip)
        print(f"Mocap: Motive at {args.server_ip}, listening on {args.client_ip} "
              f"({'unicast' if args.unicast else 'multicast'}), rigid body {args.rigid_id}")
        self.client = NatNetClient()
        self.client.set_client_address(args.client_ip)
        self.client.set_server_address(args.server_ip)
        self.client.set_use_multicast(not args.unicast)
        self.client.set_print_level(0)
        self.client.new_frame_with_data_listener = self.on_mocap_frame

    def elapsed_ns(self) -> int:
        return time.monotonic_ns() - self.clock_zero_ns

    def nearest_camera_frame(self, sample_ns: int) -> Tuple[Optional[int], Optional[int]]:
        with self.camera_lock:
            if not self.camera_frames:
                return None, None
            return min(self.camera_frames, key=lambda item: abs(item[1] - sample_ns))

    # ----------------------------------------------------------------- mocap
    def on_mocap_frame(self, data: dict) -> None:
        """NatNet callback; invoked after a complete mocap frame is decoded."""
        rigid_bodies = data["mocap_data"].rigid_body_data.rigid_body_list
        self.seen_rigid_ids.update(rb.id_num for rb in rigid_bodies)
        body = next((rb for rb in rigid_bodies if rb.id_num == self.args.rigid_id), None)
        if body is None:
            return
        valid = bool(body.tracking_valid)
        with self.latest_lock:
            self.latest = (time.monotonic(), valid, tuple(body.pos), tuple(body.rot))
        if not self.armed.is_set() or self.stop_event.is_set():
            return

        # The first valid sample after the countdown is the common zero time
        # for both CSV and video.
        if not self.recording_started.is_set():
            if not valid:
                return
            self.clock_zero_ns = time.monotonic_ns()
            self.recording_started.set()
        capture_ns = self.elapsed_ns()
        self.latest_mocap_ns = capture_ns

        flange = [""] * 6
        if self.calibration is not None and valid:
            flange = [round(v, 3) for v in kin.matrix_to_pose(
                self.calibration.flange_pose(body.pos, body.rot))]

        video_index, video_ns = self.nearest_camera_frame(capture_ns)
        video_time = "" if video_ns is None else video_ns / 1e9
        sync_offset = "" if video_ns is None else (capture_ns - video_ns) / 1e9
        row = [
            data["frame_number"], data["timestamp"], capture_ns / 1e9,
            body.id_num, *body.pos, *body.rot, int(valid), body.error,
            "" if video_index is None else video_index, video_time, sync_offset,
            *flange,
        ]
        with self.csv_lock:
            if self.csv_file.closed:
                return
            self.csv_writer.writerow(row)
            self.rows_written += 1
            self.rows_invalid += not valid

    def status_line(self) -> str:
        with self.latest_lock:
            latest = self.latest
        if latest is None or time.monotonic() - latest[0] > 0.5:
            seen = sorted(self.seen_rigid_ids)
            return (f"rigid body {self.args.rigid_id}: NO DATA "
                    f"(streaming IDs: {seen or 'none'})")
        _, valid, pos, quat = latest
        if not valid:
            return f"rigid body {self.args.rigid_id}: TRACKING LOST"
        if self.calibration is None:
            return ("tracked  mocap xyz=[{:7.3f} {:7.3f} {:7.3f}] m".format(*pos))
        flange = self.calibration.flange_pose(pos, quat)
        x, y, z, roll, pitch, yaw = kin.matrix_to_pose(flange)
        return (f"tracked  flange xyz=[{x:6.0f} {y:6.0f} {z:6.0f}] mm "
                f"rpy=[{roll:5.0f} {pitch:5.0f} {yaw:5.0f}]  {self.reach.check(flange)}")

    def status_loop(self) -> None:
        width = 0
        while not self.stop_event.wait(STATUS_PERIOD_S):
            if not self.show_status:
                continue
            prefix = ""
            if self.recording_started.is_set():
                prefix = f"REC {self.elapsed_ns() / 1e9:6.1f}s | "
            line = prefix + self.status_line()
            # Pad to wipe the previous (possibly longer) line.
            sys.stdout.write("\r" + line.ljust(width))
            sys.stdout.flush()
            width = len(line)

    # ---------------------------------------------------------------- camera
    def camera_loop(self) -> None:
        cap = open_camera(self.args.camera, self.args.backend)
        try:
            if not cap.isOpened():
                self.camera_error = (f"Could not open camera {self.args.camera!r} "
                                     f"(backend {self.args.backend}); try --list-cameras")
                self.stop_event.set()
                return

            if self.args.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            if self.args.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            if self.args.fps:
                cap.set(cv2.CAP_PROP_FPS, self.args.fps)

            # Size the writer from a real frame: some backends (notably MSMF
            # on Windows) report 0x0 or a stale size before the first read,
            # and VideoWriter silently drops frames whose size doesn't match.
            first = read_first_frame(cap, timeout_s=10.0)
            if first is None:
                self.camera_error = f"Camera {self.args.camera!r} opened but returned no frames"
                self.stop_event.set()
                return
            height, width = first.shape[:2]
            fps = self.args.fps or cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 1:
                fps = 30.0
            self.video_fps = fps

            writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (width, height),
            )
            if not writer.isOpened():
                self.camera_error = f"Could not create video {self.video_path}"
                self.stop_event.set()
                return

            self.camera_info = f"{width}x{height} @ {fps:g} fps"
            print(f"Camera {self.args.camera!r}: {self.camera_info}")
            try:
                self.camera_ready.set()
                last_written_frame = None
                last_ok = time.monotonic()
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        # Backends drop the occasional frame; only a sustained
                        # stall ends the recording.
                        if time.monotonic() - last_ok > CAMERA_STALL_S:
                            self.camera_error = (
                                f"Camera returned no frames for {CAMERA_STALL_S:g}s")
                            self.stop_event.set()
                            break
                        self.camera_dropouts += 1
                        time.sleep(0.005)
                        continue
                    last_ok = time.monotonic()
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    if not self.recording_started.is_set():
                        continue

                    frame_ns = self.elapsed_ns()
                    # VideoWriter has no per-frame timestamps.  Map elapsed
                    # real time onto its constant-FPS timeline: drop frames
                    # from over-producing cameras and duplicate when needed.
                    # Never let video run beyond the most recently received
                    # mocap sample. This gives both files the same endpoint.
                    target_ns = min(frame_ns, self.latest_mocap_ns)
                    target_index = int(target_ns / 1e9 * fps)
                    if target_index < self.video_frames_written:
                        continue
                    output_frame = frame if last_written_frame is None else last_written_frame
                    while self.video_frames_written <= target_index:
                        if self.video_frames_written == target_index:
                            output_frame = frame
                        frame_index = self.video_frames_written
                        writer.write(output_frame)
                        with self.camera_lock:
                            self.camera_frames.append((frame_index, frame_ns))
                        self.video_frames_written += 1
                    last_written_frame = frame
            finally:
                # A stop signal can arrive in the mocap callback just before
                # the camera loop sees the final time. Pad to that final mocap
                # sample so both outputs cover the identical time interval.
                if last_written_frame is not None and self.recording_started.is_set():
                    final_index = int(self.latest_mocap_ns / 1e9 * fps)
                    while self.video_frames_written <= final_index:
                        writer.write(last_written_frame)
                        self.video_frames_written += 1
                writer.release()
        finally:
            cap.release()

    # ------------------------------------------------------------------ main
    def connect(self) -> bool:
        if not self.client.run("d"):
            self.camera_error = "Could not start the NatNet client"
            return False
        time.sleep(1.0)
        if not self.client.connected():
            self.camera_error = ("Could not connect to Motive; check streaming and IP "
                                 "settings (find_mocap.py)")
            return False
        return True

    def wait_for_camera(self) -> bool:
        # The camera thread has its own 10 s first-frame timeout; this is a backstop.
        deadline = time.monotonic() + 15.0
        while not self.stop_event.is_set() and not self.camera_ready.is_set():
            if time.monotonic() >= deadline:
                self.camera_error = "Timed out waiting for the first camera frame"
                self.stop_event.set()
                return False
            time.sleep(0.01)
        return not self.stop_event.is_set()

    def countdown(self) -> bool:
        for i in range(int(self.args.countdown), 0, -1):
            print(f"  {i} ...", flush=True)
            if self.stop_event.wait(1.0):
                return False
        frac = self.args.countdown - int(self.args.countdown)
        return not (frac > 0 and self.stop_event.wait(frac))

    def run(self) -> int:
        camera_thread = threading.Thread(target=self.camera_loop, name="camera", daemon=True)
        status_thread = threading.Thread(target=self.status_loop, name="status", daemon=True)
        camera_thread.start()
        connected = False
        try:
            if self.wait_for_camera():
                connected = self.connect()
                if not connected:
                    self.stop_event.set()
            if not self.stop_event.is_set():
                status_thread.start()
                print("\nPick up the end effector and check the status line.  Press Enter "
                      f"to start (recording begins after a {self.args.countdown:g} s "
                      "countdown); Ctrl+C quits.")
                self.show_status = True
                input()
                self.show_status = False
                if self.countdown():
                    self.armed.set()
                    print("RECORDING -- hold still for a moment, do the motion, hold "
                          "still, then press Enter (or Ctrl+C) to stop.")
                    self.show_status = True
                    threading.Thread(target=self.wait_for_enter, name="stdin",
                                     daemon=True).start()
                    self.monitor()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.show_status = False
            self.stop_event.set()
            print()
            if connected:
                self.client.shutdown()
            camera_thread.join(timeout=3.0)
            with self.csv_lock:
                self.csv_file.close()

        if self.camera_dropouts:
            print(f"Note: camera dropped {self.camera_dropouts} reads (frames were duplicated "
                  "to keep timing).", file=sys.stderr)
        if not self.recording_started.is_set():
            for path in (self.csv_path, self.video_path):
                path.unlink(missing_ok=True)
            if self.camera_error:
                print(f"ERROR: {self.camera_error}", file=sys.stderr)
                return 1
            print("Nothing recorded.")
            return 0
        self.write_meta()
        duration = self.latest_mocap_ns / 1e9
        print(f"Saved {self.csv_path}\n      {self.video_path}\n      {self.meta_path}")
        print(f"{duration:.1f} s, {self.rows_written} mocap frames "
              f"({self.rows_written / max(duration, 1e-9):.0f} Hz, {self.rows_invalid} "
              f"untracked), {self.video_frames_written} video frames")
        if self.camera_error:
            print(f"ERROR: {self.camera_error}", file=sys.stderr)
            return 1
        return 0

    def monitor(self) -> None:
        warned_stale = False
        while not self.stop_event.wait(0.1):
            # Video is capped at the latest mocap sample, so if mocap stops
            # arriving the video silently stalls too -- say so.
            if self.recording_started.is_set():
                stale_s = (self.elapsed_ns() - self.latest_mocap_ns) / 1e9
                if stale_s > 1.0 and not warned_stale:
                    print(f"\nWARNING: no samples for rigid body {self.args.rigid_id} "
                          f"in {stale_s:.1f}s (lost tracking or stream dropped?)",
                          file=sys.stderr)
                    warned_stale = True
                elif stale_s <= 1.0 and warned_stale:
                    print("\nMocap samples resumed.", file=sys.stderr)
                    warned_stale = False

    def wait_for_enter(self) -> None:
        try:
            if sys.stdin.readline():
                self.stop_event.set()
        except (OSError, ValueError):
            pass

    def write_meta(self) -> None:
        args = self.args
        meta = {
            "csv": self.csv_path.name,
            "video": self.video_path.name,
            "rigid_id": args.rigid_id,
            "server_ip": args.server_ip,
            "client_ip": args.client_ip,
            "duration_s": self.latest_mocap_ns / 1e9,
            "mocap_frames": self.rows_written,
            "mocap_frames_untracked": self.rows_invalid,
            "camera": str(args.camera),
            "camera_info": self.camera_info,
            "video_fps": self.video_fps,
            "video_frames": self.video_frames_written,
            "calibration": None if self.calibration is None else str(self.calibration.path),
            "tool_length": args.tool_length,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2))


def default_backend() -> str:
    # DirectShow opens much faster and more reliably than MSMF (OpenCV's
    # Windows default) for most USB webcams.
    return "dshow" if sys.platform == "win32" else "auto"


def open_camera(source: CameraSource, backend: str) -> "cv2.VideoCapture":
    api = BACKENDS[backend]
    if api is None or isinstance(source, str):  # URLs/files go through FFmpeg
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, api)


def read_first_frame(cap: "cv2.VideoCapture", timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def measure_fps(cap: "cv2.VideoCapture", seconds: float = 1.5) -> float:
    """Frames actually delivered per second (the reported FPS is often wrong)."""
    start = time.monotonic()
    n = 0
    while time.monotonic() - start < seconds:
        ok, _ = cap.read()
        n += bool(ok)
    return n / (time.monotonic() - start)


def list_cameras(backend: str, max_index: int = 10) -> None:
    snap_dir = Path(tempfile.gettempdir()) / "camera_probe"
    snap_dir.mkdir(exist_ok=True)
    print(f"Probing cameras (backend {backend}); snapshots go to {snap_dir}")
    found = False
    for index in range(max_index):
        cap = open_camera(index, backend)
        try:
            if not cap.isOpened():
                break  # indices are contiguous; stop instead of spamming errors
            frame = read_first_frame(cap, timeout_s=3.0)
            if frame is None:
                print(f"  {index}: opens but returns no frames")
            else:
                h, w = frame.shape[:2]
                actual = measure_fps(cap)
                snap = snap_dir / f"camera_{index}.jpg"
                cv2.imwrite(str(snap), frame)
                note = "  <- too slow for recording" if actual < 10 else ""
                print(f"  {index}: {w}x{h}, reports {cap.get(cv2.CAP_PROP_FPS):g} fps, "
                      f"delivers {actual:.0f} fps  ({snap.name}){note}")
            found = True
        finally:
            cap.release()
    if not found:
        print("  no cameras found; try a different --backend")
    else:
        print(f"Open the snapshots to see which is which:  open {snap_dir}")


def camera_source(value: str) -> CameraSource:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rigid_id", type=int, nargs="?", default=DEFAULT_RIGID_ID,
                        help="Motive rigid-body streaming ID (default: %(default)s)")
    parser.add_argument("--server-ip", default=MOCAP_SERVER_IP,
                        help="Motive/NatNet server IP (default: %(default)s)")
    parser.add_argument("--client-ip", default=None,
                        help="local network-interface IP (default: this machine's address "
                             "on the interface that reaches --server-ip)")
    parser.add_argument("--unicast", action="store_true", help="use unicast instead of multicast")
    parser.add_argument("--camera", type=camera_source, default=0,
                        help="camera index or stream URL (default: %(default)s)")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=default_backend(),
                        help="OpenCV capture backend (default: %(default)s)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list working camera indices and exit")
    parser.add_argument("--width", type=int, help="requested camera width")
    parser.add_argument("--height", type=int, help="requested camera height")
    parser.add_argument("--fps", type=float, help="requested/output frame rate")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="solve_mocap_calibration.py *_solution.json (or a folder) used "
                             "for the live reach check and the flange_* columns "
                             "(default: newest under calibration/)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="record raw mocap only (no flange columns or reach check)")
    parser.add_argument("--tool-length", type=float, default=None,
                        help="end-effector length along the flange Z axis for the reach "
                             "check, mm (default: from the calibration run's config)")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="seconds between pressing Enter and recording (default: "
                             "%(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=RECORDINGS_DIR / time.strftime("%m-%d-%Y"),
                        help="output folder (default: %(default)s)")
    parser.add_argument("--name", default=time.strftime("demo_%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.backend)
        return 0
    calibration = None
    if not args.no_calibration:
        calibration = load_calibration(args.calibration)
        print(f"Calibration: {calibration.describe()}")
    if args.tool_length is None:
        args.tool_length = (calibration.tool_length if calibration is not None
                            and calibration.tool_length is not None else 0.0)
    recorder = Recorder(args, calibration)

    def request_stop(_signum: int, _frame: object) -> None:
        recorder.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    return recorder.run()


if __name__ == "__main__":
    raise SystemExit(main())
