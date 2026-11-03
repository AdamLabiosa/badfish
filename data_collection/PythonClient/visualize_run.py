#!/usr/bin/env python3
"""Render a recorded camera stream beside its synchronized 3D mocap path."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Visualization requires: python -m pip install opencv-python numpy"
    ) from exc

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"


def load_samples(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("video_frame", "").strip()]
    if not rows:
        raise ValueError(f"No mocap samples in {path}")

    def values(name: str, dtype=float) -> np.ndarray:
        return np.asarray([dtype(row[name]) for row in rows])

    return {
        "frame": values("video_frame", int),
        "xyz": np.column_stack([values("x_m"), values("y_m"), values("z_m")]),
        "quat": np.column_stack([values("qx"), values("qy"), values("qz"), values("qw")]),
    }


def rotated_x_axis(quat: np.ndarray) -> np.ndarray:
    """Rotate the +X unit vector by NatNet's (x, y, z, w) quaternion."""
    x, y, z, w = quat
    return np.asarray([1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                       2 * (x * z - w * y)])


def padded_limits(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    center = (low + high) / 2
    half = max(float((high - low).max()) / 2, 0.05) * 1.15
    return center, half


def view_rotation() -> np.ndarray:
    """Fixed isometric-like view rotation for an orthographic 3D plot."""
    azimuth, elevation = np.deg2rad(-45.0), np.deg2rad(25.0)
    rz = np.asarray([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth), np.cos(azimuth), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rx = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(elevation), -np.sin(elevation)],
        [0.0, np.sin(elevation), np.cos(elevation)],
    ])
    return rx @ rz


def project(points: np.ndarray, center: np.ndarray, half: float,
            size: int, rotation: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    viewed = (points - center) @ rotation.T
    scale = size * 0.33 / half
    pixels = np.column_stack((
        size / 2 + viewed[:, 0] * scale,
        size / 2 - viewed[:, 1] * scale,
    ))
    return np.rint(pixels).astype(np.int32)


def draw_plot(
    samples: dict[str, np.ndarray],
    sample_index: int,
    center: np.ndarray,
    half: float,
    size: int,
) -> np.ndarray:
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    xyz = samples["xyz"]
    current = xyz[sample_index]
    rotation = view_rotation()

    # Reference cube makes depth and scale legible without Matplotlib.
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], dtype=float)
    corners = center + signs * half
    corner_pixels = project(corners, center, half, size, rotation)
    for first in range(8):
        for axis in range(3):
            second = first ^ (1 << (2 - axis))
            if first < second:
                cv2.line(image, tuple(corner_pixels[first]), tuple(corner_pixels[second]),
                         (205, 205, 205), 1, cv2.LINE_AA)

    path_pixels = project(xyz[: sample_index + 1], center, half, size, rotation)
    if len(path_pixels) > 1:
        cv2.polylines(image, [path_pixels], False, (210, 110, 25), 3, cv2.LINE_AA)
    current_pixel = tuple(path_pixels[-1])
    cv2.circle(image, current_pixel, 7, (40, 40, 210), -1, cv2.LINE_AA)

    direction = rotated_x_axis(samples["quat"][sample_index])
    arrow_end = project(current + direction * half * 0.3, center, half, size, rotation)[0]
    cv2.arrowedLine(image, current_pixel, tuple(arrow_end), (40, 40, 210),
                    3, cv2.LINE_AA, tipLength=0.25)

    # Draw world axes from the lower cube corner.
    origin = center - half
    axis_colors = ((40, 40, 210), (40, 155, 40), (210, 110, 25))
    for axis, (label, color) in enumerate(zip("XYZ", axis_colors)):
        endpoint = origin.copy()
        endpoint[axis] += half * 0.55
        axis_pixels = project(np.vstack((origin, endpoint)), center, half, size, rotation)
        cv2.arrowedLine(image, tuple(axis_pixels[0]), tuple(axis_pixels[1]), color,
                        2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(image, label, tuple(axis_pixels[1] + (5, -5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.putText(image, "3D rigid-body trajectory", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(image, f"x={current[0]:.3f}  y={current[1]:.3f}  z={current[2]:.3f} m",
                (18, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (35, 35, 35), 1, cv2.LINE_AA)
    return image


def render(csv_path: Path, video_path: Path, output_path: Path) -> None:
    samples = load_samples(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Render the plot at the camera height and append it to the right.
    plot_width = height
    center, half = padded_limits(samples["xyz"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width + plot_width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create {output_path}")

    frame_number = 0
    sample_index = 0
    mapped_frames = samples["frame"]
    try:
        while True:
            ok, camera_frame = cap.read()
            if not ok:
                break
            # Select the mocap observation mapped nearest to this video frame.
            sample_index = int(np.argmin(np.abs(mapped_frames - frame_number)))
            plot_frame = draw_plot(samples, sample_index, center, half, plot_width)
            if plot_frame.shape[:2] != (height, plot_width):
                plot_frame = cv2.resize(plot_frame, (plot_width, height))
            writer.write(np.hstack((camera_frame, plot_frame)))
            frame_number += 1
    finally:
        cap.release()
        writer.release()
    print(f"Saved {output_path} ({frame_number} frames)")


def resolve_csvs(target: Path) -> list[Path]:
    """Accept a CSV, a recording name, or a folder of recordings."""
    for candidate in (target, RECORDINGS_DIR / target):
        if candidate.is_dir():
            return sorted(candidate.glob("*.csv"))
        if candidate.suffix == ".csv" and candidate.is_file():
            return [candidate]
    matches = sorted(RECORDINGS_DIR.rglob(f"{target.stem}.csv"))
    if not matches:
        raise SystemExit(f"No recording named {target} under {RECORDINGS_DIR}")
    if len(matches) > 1:
        listed = "\n  ".join(str(match) for match in matches)
        raise SystemExit(f"Recording name {target} is ambiguous:\n  {listed}")
    return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="CSV from record_mocap_camera.py, a recording name "
                             f"(searched under {RECORDINGS_DIR}), or a folder")
    parser.add_argument("video", type=Path, nargs="?",
                        help="camera MP4 (default: same stem as CSV)")
    parser.add_argument("--output", type=Path,
                        help="output MP4 (default: <stem>_visualization.mp4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-render outputs that already exist")
    args = parser.parse_args()
    args.csvs = resolve_csvs(args.target)
    if len(args.csvs) > 1 and (args.video or args.output):
        parser.error("video/--output can only be given for a single recording")
    return args


def main() -> int:
    args = parse_args()
    for csv_path in args.csvs:
        video = args.video or csv_path.with_suffix(".mp4")
        output = args.output or csv_path.with_name(csv_path.stem + "_visualization.mp4")
        if output.exists() and not args.overwrite:
            print(f"Skipping {csv_path.name}: {output} exists (use --overwrite)")
            continue
        if not video.exists():
            print(f"Skipping {csv_path.name}: missing video {video}")
            continue
        render(csv_path, video, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
