#!/usr/bin/env python3
"""Record one NatNet rigid body and a time-aligned camera video.

The CSV and camera use the same monotonic clock.  Each mocap sample includes
the index and timestamp of the nearest camera frame captured so far, allowing
the data to be aligned with frames in the MP4 without relying on wall-clock
time.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV is required: python -m pip install opencv-python") from exc

from NatNetClient import NatNetClient


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

# Recordings live in data/recordings/<MM-DD-YYYY>/, independent of the cwd.
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"


class Recorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_event = threading.Event()
        self.camera_ready = threading.Event()
        self.recording_started = threading.Event()
        self.clock_zero_ns = time.monotonic_ns()
        self.camera_frames: Deque[Tuple[int, int]] = deque(maxlen=240)
        self.camera_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.camera_error: Optional[str] = None
        self.video_frames_written = 0
        self.latest_mocap_ns = 0
        self.camera_dropouts = 0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = args.output_dir / f"{args.name}.csv"
        self.video_path = args.output_dir / f"{args.name}.mp4"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "mocap_frame", "natnet_timestamp_s", "capture_time_s",
                "rigid_id", "x_m", "y_m", "z_m",
                "qx", "qy", "qz", "qw", "tracking_valid", "mean_error_m",
                "video_frame", "video_frame_time_s", "sync_offset_s",
            ]
        )

        self.client = NatNetClient()
        self.client.set_client_address(args.client_ip)
        self.client.set_server_address(args.server_ip)
        self.client.set_use_multicast(not args.unicast)
        self.client.new_frame_with_data_listener = self.on_mocap_frame

    def elapsed_ns(self) -> int:
        return time.monotonic_ns() - self.clock_zero_ns

    def nearest_camera_frame(self, sample_ns: int) -> Tuple[Optional[int], Optional[int]]:
        with self.camera_lock:
            if not self.camera_frames:
                return None, None
            return min(self.camera_frames, key=lambda item: abs(item[1] - sample_ns))

    def on_mocap_frame(self, data: dict) -> None:
        """NatNet callback; invoked after a complete mocap frame is decoded."""
        rigid_bodies = data["mocap_data"].rigid_body_data.rigid_body_list
        body = next((rb for rb in rigid_bodies if rb.id_num == self.args.rigid_id), None)
        if body is None:
            return

        # The first valid target sample is the common zero-time gate for both
        # CSV and video.  Camera initialization and NatNet connection time are
        # therefore excluded from both outputs.
        if not self.recording_started.is_set():
            self.clock_zero_ns = time.monotonic_ns()
            self.recording_started.set()
        capture_ns = self.elapsed_ns()
        self.latest_mocap_ns = capture_ns

        video_index, video_ns = self.nearest_camera_frame(capture_ns)
        video_time = "" if video_ns is None else video_ns / 1e9
        sync_offset = "" if video_ns is None else (capture_ns - video_ns) / 1e9
        row = [
            data["frame_number"], data["timestamp"], capture_ns / 1e9,
            body.id_num, *body.pos, *body.rot, body.tracking_valid, body.error,
            "" if video_index is None else video_index, video_time, sync_offset,
        ]
        with self.csv_lock:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

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

            writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (width, height),
            )
            if not writer.isOpened():
                self.camera_error = f"Could not create video {self.video_path}"
                self.stop_event.set()
                return

            print(f"Camera {self.args.camera!r}: {width}x{height} @ {fps:g} fps")
            try:
                self.camera_ready.set()
                last_written_frame = None
                last_ok = time.monotonic()
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        # Windows backends drop the occasional frame; only a
                        # sustained stall ends the recording.
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

    def run(self) -> int:
        camera_thread = threading.Thread(target=self.camera_loop, name="camera", daemon=True)
        camera_thread.start()

        # Initialize the camera before connecting, but do not record until the
        # first valid target rigid-body sample arrives.  (The camera thread
        # has its own 10 s first-frame timeout; this is a backstop.)
        deadline = time.monotonic() + 15.0
        while not self.stop_event.is_set() and not self.camera_ready.is_set():
            if time.monotonic() >= deadline:
                self.camera_error = "Timed out waiting for the first camera frame"
                self.stop_event.set()
                break
            time.sleep(0.01)

        started = False
        if not self.stop_event.is_set():
            started = self.client.run("d")
            if not started:
                self.camera_error = "Could not start the NatNet client"
                self.stop_event.set()

        if started:
            time.sleep(1.0)
            if not self.client.connected():
                self.camera_error = "Could not connect to Motive; check streaming and IP settings"
                self.stop_event.set()

        if not self.stop_event.is_set():
            print("Recording... press Enter or Ctrl+C to stop.")
            if sys.stdin.isatty():
                # select() on stdin is POSIX-only (WinError 10038 on Windows),
                # so wait for Enter on a background thread instead.
                threading.Thread(target=self.wait_for_enter, name="stdin",
                                 daemon=True).start()
        try:
            warned_stale = False
            while not self.stop_event.wait(0.1):
                # Video is capped at the latest mocap sample, so if mocap
                # stops arriving the video silently stalls too -- say so.
                if self.recording_started.is_set():
                    stale_s = (self.elapsed_ns() - self.latest_mocap_ns) / 1e9
                    if stale_s > 1.0 and not warned_stale:
                        print(f"WARNING: no samples for rigid body {self.args.rigid_id} "
                              f"in {stale_s:.1f}s (lost tracking or stream dropped?)",
                              file=sys.stderr)
                        warned_stale = True
                    elif stale_s <= 1.0 and warned_stale:
                        print("Mocap samples resumed.", file=sys.stderr)
                        warned_stale = False
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            if started:
                self.client.shutdown()
            camera_thread.join(timeout=3.0)
            with self.csv_lock:
                self.csv_file.close()

        if self.camera_dropouts:
            print(f"Note: camera dropped {self.camera_dropouts} reads (frames were duplicated "
                  "to keep timing).", file=sys.stderr)
        if self.camera_error:
            print(f"ERROR: {self.camera_error}", file=sys.stderr)
            return 1
        print(f"Saved {self.csv_path} and {self.video_path} ({self.video_frames_written} frames)")
        return 0


    def wait_for_enter(self) -> None:
        try:
            if sys.stdin.readline():
                self.stop_event.set()
        except (OSError, ValueError):
            pass


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


def list_cameras(backend: str, max_index: int = 10) -> None:
    print(f"Probing camera indices 0-{max_index - 1} (backend {backend})...")
    found = False
    for index in range(max_index):
        cap = open_camera(index, backend)
        try:
            if not cap.isOpened():
                continue
            frame = read_first_frame(cap, timeout_s=3.0)
            if frame is None:
                print(f"  {index}: opens but returns no frames")
            else:
                h, w = frame.shape[:2]
                print(f"  {index}: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):g} fps")
            found = True
        finally:
            cap.release()
    if not found:
        print("  no cameras found; try a different --backend")


def camera_source(value: str) -> CameraSource:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rigid_id", type=int, nargs="?", help="Motive rigid-body streaming ID")
    parser.add_argument("--server-ip", default="192.168.31.51", help="Motive/NatNet server IP")
    parser.add_argument("--client-ip", default="192.168.31.50", help="local network-interface IP")
    parser.add_argument("--unicast", action="store_true", help="use unicast instead of multicast")
    parser.add_argument("--camera", type=camera_source, default=2,
                        help="camera index or stream URL (default: %(default)s)")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=default_backend(),
                        help="OpenCV capture backend (default: %(default)s)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="list working camera indices and exit")
    parser.add_argument("--width", type=int, help="requested camera width")
    parser.add_argument("--height", type=int, help="requested camera height")
    parser.add_argument("--fps", type=float, help="requested/output frame rate")
    parser.add_argument("--output-dir", type=Path,
                        default=RECORDINGS_DIR / time.strftime("%m-%d-%Y"),
                        help="output folder (default: %(default)s)")
    parser.add_argument("--name", default=time.strftime("capture_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if args.rigid_id is None and not args.list_cameras:
        parser.error("rigid_id is required (unless using --list-cameras)")
    return args


def main() -> int:
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.backend)
        return 0
    recorder = Recorder(args)

    def request_stop(_signum: int, _frame: object) -> None:
        recorder.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return recorder.run()


if __name__ == "__main__":
    raise SystemExit(main())
