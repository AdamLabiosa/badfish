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
import select
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
        cap = cv2.VideoCapture(self.args.camera)
        try:
            if not cap.isOpened():
                self.camera_error = f"Could not open camera {self.args.camera!r}"
                self.stop_event.set()
                return

            if self.args.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            if self.args.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            if self.args.fps:
                cap.set(cv2.CAP_PROP_FPS, self.args.fps)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

            try:
                self.camera_ready.set()
                last_written_frame = None
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self.camera_error = "Camera stopped returning frames"
                        self.stop_event.set()
                        break
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
        # first valid target rigid-body sample arrives.
        deadline = time.monotonic() + 5.0
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
        try:
            while not self.stop_event.wait(0.1):
                if sys.stdin.isatty():
                    readable, _, _ = select.select([sys.stdin], [], [], 0)
                    if readable:
                        sys.stdin.readline()
                        self.stop_event.set()
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            if started:
                self.client.shutdown()
            camera_thread.join(timeout=3.0)
            with self.csv_lock:
                self.csv_file.close()

        if self.camera_error:
            print(f"ERROR: {self.camera_error}", file=sys.stderr)
            return 1
        print(f"Saved {self.csv_path} and {self.video_path} ({self.video_frames_written} frames)")
        return 0


def camera_source(value: str) -> CameraSource:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rigid_id", type=int, help="Motive rigid-body streaming ID")
    parser.add_argument("--server-ip", default="192.168.31.51", help="Motive/NatNet server IP")
    parser.add_argument("--client-ip", default="192.168.31.50", help="local network-interface IP")
    parser.add_argument("--unicast", action="store_true", help="use unicast instead of multicast")
    parser.add_argument("--camera", type=camera_source, default=2,
                        help="camera index or stream URL (default: %(default)s)")
    parser.add_argument("--width", type=int, help="requested camera width")
    parser.add_argument("--height", type=int, help="requested camera height")
    parser.add_argument("--fps", type=float, help="requested/output frame rate")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--name", default=time.strftime("capture_%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recorder = Recorder(args)

    def request_stop(_signum: int, _frame: object) -> None:
        recorder.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return recorder.run()


if __name__ == "__main__":
    raise SystemExit(main())
