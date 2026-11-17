#!/usr/bin/env python3
"""Convert recorded demonstrations into a LeRobot (v3.0) dataset for pi0.5.

Input: recordings made by data_collection/PythonClient/record_mocap_camera.py,
i.e. for every demo <name>:

    <name>.csv            mocap frames (raw Motive pose + capture_time_s)
    <name>_meta.json      settings, incl. the calibration used
    <name>.mp4            camera video on the same clock (frame k ~ t = k / fps)
    <name>_<camera>.mp4   (optional) further cameras on the same clock

Output: one LeRobot episode per demo, at the video frame rate:

    observation.images.<camera>  RGB video per camera (resized, see --image-size)
    observation.state            flange pose at frame t   (9 floats, pose_format.py)
    action                       flange pose at frame t+1 (absolute target pose)
    task                         language instruction (prompt for pi0.5)

The flange pose comes from the same pipeline the replay uses
(rollout_trajectory.py): calibration from the recording's _meta.json, still
time at the start/end trimmed, mocap jitter smoothed.  So what the policy
learns to output is what rollout_trajectory.py would have sent to the arm.

Examples (run with the preprocessing venv, from anywhere):

    # see what would be converted, nothing is written
    python convert_to_lerobot.py ../data_collection/recordings/11-12-2026 \\
        --task "pick up the fish" --dry-run

    # convert, then tar it up for CHTC
    python convert_to_lerobot.py ../data_collection/recordings/11-12-2026 \\
        --task "pick up the fish" --repo-id badfish/fish_v0 --tar

Per-demo instructions: put {"<name>": "<instruction>", ...} in a JSON file and
pass --tasks-file, or add a "task" key to the demo's _meta.json.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import math
import shutil
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("Needs OpenCV; use the preprocessing venv (see preprocessing.md)") from exc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "data_collection" / "PythonClient"))

from mocap_to_arm import MocapCalibration  # noqa: E402
from rollout_trajectory import (  # noqa: E402
    PlanError, calibration_for, load_demo, motion_window, resample,
)
from pose_format import POSE_DIM, POSE_NAMES, pose_to_vec  # noqa: E402

DEFAULT_OUT = HERE / "datasets"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_PREFIX = "observation.images."


# ----------------------------------------------------------------- planning
@dataclass
class Episode:
    name: str
    csv: Path
    meta: dict
    videos: Dict[str, Path]          # camera name -> mp4
    task: Optional[str] = None
    window: tuple = (0.0, 0.0)       # s, recording time kept
    num_frames: int = 0
    calibration: Optional[MocapCalibration] = None
    problems: List[str] = field(default_factory=list)


def find_recordings(targets: List[Path]) -> List[Path]:
    """Recording CSVs: given directly, or every <name>.csv with a <name>_meta.json
    in the given folders (recursively).  Plan / rollout CSVs are never picked up."""
    found: List[Path] = []
    for target in targets:
        target = target.resolve()
        if target.is_dir():
            for meta in sorted(target.rglob("*_meta.json")):
                csv = meta.with_name(meta.name[: -len("_meta.json")] + ".csv")
                if csv.exists():
                    found.append(csv)
        elif target.suffix == ".csv":
            found.append(target)
        elif target.name.endswith("_meta.json"):
            found.append(target.with_name(target.name[: -len("_meta.json")] + ".csv"))
        else:
            found.append(target.with_suffix(".csv"))
    unique = list(dict.fromkeys(found))
    missing = [p for p in unique if not p.exists()]
    if missing:
        raise SystemExit("Not found: " + ", ".join(map(str, missing)))
    return unique


def find_videos(csv: Path, meta: dict, main_camera: str) -> Dict[str, Path]:
    """Camera name -> video.  A "videos": {name: file} entry in _meta.json wins
    (for the multi-camera recorder); otherwise <name>.mp4 is `main_camera` and
    every <name>_<camera>.mp4 is another camera."""
    declared = meta.get("videos")
    if isinstance(declared, dict) and declared:
        return {cam: csv.with_name(Path(f).name) for cam, f in declared.items()}
    videos: Dict[str, Path] = {}
    main = csv.with_suffix(".mp4")
    if main.exists():
        videos[main_camera] = main
    for path in sorted(csv.parent.glob(f"{csv.stem}_*.mp4")):
        cam = path.stem[len(csv.stem) + 1:]
        if cam.endswith("visualization"):           # visualize_run.py output
            continue
        if path.with_suffix(".csv").exists():        # another recording (demo vs demo_2)
            continue
        videos[cam] = path
    return videos


def video_info(path: Path) -> tuple:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0, 0
        return cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def load_tasks_file(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object {{name or glob: instruction}}")
    return data


def task_for(name: str, meta: dict, tasks: Dict[str, str], default: Optional[str]) -> Optional[str]:
    if name in tasks:
        return tasks[name]
    for pattern, task in tasks.items():
        if fnmatch.fnmatch(name, pattern):
            return task
    return meta.get("task") or default


def plan_episode(csv: Path, args: argparse.Namespace, tasks: Dict[str, str]) -> Episode:
    meta_path = csv.with_name(csv.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ep = Episode(name=csv.stem, csv=csv, meta=meta,
                 videos=find_videos(csv, meta, args.main_camera))
    ep.task = task_for(ep.name, meta, tasks, args.task)

    if not meta:
        ep.problems.append("no _meta.json (can't tell which calibration it used)")
    if not ep.videos:
        ep.problems.append("no video")
    if not ep.task:
        ep.problems.append("no task instruction (--task, --tasks-file or meta 'task')")
    if ep.problems:
        return ep

    # Silence the rollout helpers' prints; problems are reported in the table.
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            demo = load_demo(csv)
            ep.calibration = calibration_for(csv, args.calibration)
    except SystemExit as exc:
        ep.problems.append(str(exc))
        return ep
    # calibration_for() falls back to the newest calibration when the recorded one
    # is gone; that's only right if Motive wasn't re-grounded in between.
    fallback = [line.strip() for line in log.getvalue().splitlines() if "using the newest" in line]
    if fallback and not args.allow_calibration_fallback and args.calibration is None:
        ep.problems.extend(fallback)
        return ep

    t0, t1 = demo.t[0], demo.t[-1]
    if not args.no_auto_trim:
        try:
            t0, t1 = motion_window(demo, args.still_mm, args.still_deg, args.still_pad)
        except PlanError as exc:
            ep.problems.append(str(exc))
            return ep

    # Every camera must cover the window.
    for cam, path in ep.videos.items():
        fps, frames = video_info(path)
        if fps <= 0 or frames <= 0:
            ep.problems.append(f"can't read video {path.name}")
            return ep
        t1 = min(t1, frames / fps - 1.0 / args.fps)

    gaps = np.diff(demo.t)
    bad = np.nonzero((gaps > args.max_gap) & (demo.t[1:] > t0) & (demo.t[:-1] < t1))[0]
    if len(bad):
        worst = bad[np.argmax(gaps[bad])]
        ep.problems.append(f"tracking gap of {gaps[worst]:.2f} s at t={demo.t[worst]:.1f} s "
                           f"(--max-gap {args.max_gap:g})")
    ep.window = (float(t0), float(t1))
    ep.num_frames = int(math.floor((t1 - t0) * args.fps)) + 1
    if (t1 - t0) < args.min_duration:
        ep.problems.append(f"only {t1 - t0:.1f} s of motion (--min-duration {args.min_duration:g})")
    ep._demo = demo  # type: ignore[attr-defined]
    return ep


# ----------------------------------------------------------------- frames
class FrameReader:
    """Sequential reader returning the frame shown at a given time, resized, RGB."""

    def __init__(self, path: Path, size: Optional[tuple]) -> None:
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"can't open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = size            # (width, height) or None
        self.pos = -1               # index of the last decoded frame
        self.frame: Optional[np.ndarray] = None

    def at_time(self, t: float) -> np.ndarray:
        # The recorder writes the camera frame grabbed in [k/fps, (k+1)/fps) as frame k.
        index = max(0, int(math.floor(t * self.fps + 1e-6)))
        if index < self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.pos = index - 1
        while self.pos < index:
            if self.pos < index - 1:
                ok = self.cap.grab()            # skip without converting
            else:
                ok, frame = self.cap.read()
                if ok:
                    self.frame = frame
            if not ok:
                break                           # past the end: keep the last frame
            self.pos += 1
        if self.frame is None:
            raise RuntimeError(f"{self.path.name}: no frame at t={t:.2f} s")
        img = self.frame
        if self.size is not None and (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


def episode_poses(ep: Episode, args: argparse.Namespace) -> tuple:
    """Times (recording clock) and (N, 9) flange-pose vectors at the dataset rate."""
    t0, t1 = ep.window
    tg, pos, quat = resample(ep._demo, t0, t1, args.fps, args.smooth)  # type: ignore[attr-defined]
    tg, pos, quat = tg[: ep.num_frames], pos[: ep.num_frames], quat[: ep.num_frames]
    flange = ep.calibration.flange_poses(pos, quat)
    return tg, pose_to_vec(flange).astype(np.float32)


# ----------------------------------------------------------------- writing
def build_features(cameras: List[str], size: tuple) -> dict:
    width, height = size
    features = {
        STATE_KEY: {"dtype": "float32", "shape": (POSE_DIM,), "names": list(POSE_NAMES)},
        ACTION_KEY: {"dtype": "float32", "shape": (POSE_DIM,), "names": list(POSE_NAMES)},
    }
    for cam in cameras:
        features[IMAGE_PREFIX + cam] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def write_dataset(episodes: List[Episode], cameras: List[str], size: tuple,
                  args: argparse.Namespace, root: Path) -> None:
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=build_features(cameras, size),
        root=root, robot_type="xarm6", use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )
    sources = []
    for index, ep in enumerate(episodes):
        times, poses = episode_poses(ep, args)
        # action[t] = pose at t+1; the last frame holds its pose (demos end at rest).
        actions = np.concatenate([poses[1:], poses[-1:]], axis=0)
        readers = {cam: FrameReader(ep.videos[cam], size) for cam in cameras}
        try:
            for k, t in enumerate(times):
                frame = {STATE_KEY: poses[k], ACTION_KEY: actions[k], "task": ep.task}
                for cam, reader in readers.items():
                    frame[IMAGE_PREFIX + cam] = reader.at_time(t)
                dataset.add_frame(frame)
        finally:
            for reader in readers.values():
                reader.close()
        dataset.save_episode()
        sources.append({
            "episode_index": index, "recording": str(ep.csv.relative_to(REPO)),
            "window_s": [round(ep.window[0], 3), round(ep.window[1], 3)],
            "frames": len(times), "task": ep.task,
            "videos": {cam: ep.videos[cam].name for cam in cameras},
            "calibration": str(ep.calibration.path),
        })
        print(f"  [{index + 1}/{len(episodes)}] {ep.name}: {len(times)} frames")
    dataset.finalize()

    # pi0.5 normalises state/action with q01/q99 quantiles by default; the stats
    # aggregated while writing are only a conservative envelope, so recompute.
    recompute_stats(LeRobotDataset(args.repo_id, root=root))

    (root / "meta" / "badfish_sources.json").write_text(json.dumps({
        "pose_format": "flange pose in arm base frame: x,y,z mm + first two rotation "
                       "matrix columns (preprocessing/pose_format.py)",
        "action": "absolute flange pose of the next frame",
        "smooth_s": args.smooth, "auto_trim": not args.no_auto_trim,
        "image_size": list(size), "episodes": sources,
    }, indent=2))


def make_tar(root: Path) -> Path:
    # Videos are already compressed; a plain tar is as small and much faster.
    tar_path = root.with_name(root.name + ".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(root, arcname=root.name)
    return tar_path


# ----------------------------------------------------------------- cli
def parse_size(value: str) -> Optional[tuple]:
    if value.lower() in ("none", "original"):
        return None
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 320x180, or 'original'")
    return w, h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", type=Path,
                   help="recording CSVs (or their _meta.json / base path), or folders to "
                        "take every recording in")
    p.add_argument("--repo-id", default="badfish/demos",
                   help="dataset id; use <hf_user>/<name> if you will --push-to-hub")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                   help="the dataset goes to <output-dir>/<repo-id> (default: preprocessing/datasets)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    p.add_argument("--dry-run", action="store_true", help="only list what would be converted")

    g = p.add_argument_group("content")
    g.add_argument("--task", help="instruction for every demo without a more specific one")
    g.add_argument("--tasks-file", type=Path,
                   help='JSON {"<demo name or glob>": "<instruction>"}')
    g.add_argument("--exclude", nargs="+", default=[], metavar="GLOB",
                   help="skip demos whose name matches")
    g.add_argument("--main-camera", default="main",
                   help="camera name for <name>.mp4; becomes observation.images.<name> "
                        "(default: main)")
    g.add_argument("--cameras", nargs="+",
                   help="cameras to include (default: all, and every demo must have the same set)")
    g.add_argument("--image-size", type=parse_size, default=(320, 180),
                   help="stored WIDTHxHEIGHT (default 320x180; pi0.5 resizes to 224 anyway), "
                        "or 'original'")
    g.add_argument("--fps", type=int, default=30, help="dataset frame rate (default 30, the camera's)")

    g = p.add_argument_group("trajectory (same meaning/defaults as rollout_trajectory.py)")
    g.add_argument("--no-auto-trim", action="store_true", help="keep still time at start/end")
    g.add_argument("--still-mm", type=float, default=5.0)
    g.add_argument("--still-deg", type=float, default=3.0)
    g.add_argument("--still-pad", type=float, default=0.5)
    g.add_argument("--smooth", type=float, default=0.04, help="Gaussian sigma in s (0 = off)")
    g.add_argument("--max-gap", type=float, default=0.1, help="longest tracking dropout allowed, s")
    g.add_argument("--min-duration", type=float, default=2.0, help="skip shorter demos, s")
    g.add_argument("--calibration", type=Path, help="override the calibration for every demo")
    g.add_argument("--allow-calibration-fallback", action="store_true",
                   help="convert demos whose recorded calibration is missing using the newest one")

    g = p.add_argument_group("output")
    g.add_argument("--tar", action="store_true", help="also write <dataset>.tar (to copy to CHTC)")
    g.add_argument("--push-to-hub", action="store_true",
                   help="upload to the Hugging Face Hub as a private dataset (needs `hf auth login`)")
    g.add_argument("--image-writer-threads", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_tasks_file(args.tasks_file)
    csvs = [c for c in find_recordings(args.recordings)
            if not any(fnmatch.fnmatch(c.stem, g) for g in args.exclude)]
    if not csvs:
        raise SystemExit("No recordings found")

    episodes = [plan_episode(csv, args, tasks) for csv in csvs]

    # Camera set: every episode in a LeRobot dataset has the same image keys.
    usable = [ep for ep in episodes if not ep.problems]
    if args.cameras:
        cameras = list(args.cameras)
        for ep in usable:
            missing = [c for c in cameras if c not in ep.videos]
            if missing:
                ep.problems.append(f"missing camera(s) {missing}; has {sorted(ep.videos)}")
    else:
        sets = {tuple(sorted(ep.videos)) for ep in usable}
        if len(sets) > 1:
            print("Demos have different cameras: " + "; ".join(
                f"{ep.name}: {sorted(ep.videos)}" for ep in usable))
            raise SystemExit("Pick the cameras to use with --cameras")
        cameras = list(sets.pop()) if sets else []
    usable = [ep for ep in episodes if not ep.problems]

    size = args.image_size
    if size is None and usable:
        cap = cv2.VideoCapture(str(usable[0].videos[cameras[0]]))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

    print(f"{'demo':<28} {'frames':>6} {'kept (s)':>13}  task / problem")
    for ep in episodes:
        rel = ep.csv.parent.name + "/" + ep.name
        if ep.problems:
            print(f"{rel:<28} {'-':>6} {'-':>13}  SKIP: {'; '.join(ep.problems)}")
        else:
            win = f"{ep.window[0]:.1f}-{ep.window[1]:.1f}"
            print(f"{rel:<28} {ep.num_frames:>6} {win:>13}  {ep.task}")
    total = sum(ep.num_frames for ep in usable)
    print(f"\n{len(usable)} of {len(episodes)} demos, {total} frames "
          f"({total / args.fps:.0f} s) at {args.fps} fps, cameras {cameras}, images {size}")
    if not usable:
        return 1
    if args.dry_run:
        return 0

    root = (args.output_dir / args.repo_id).resolve()
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {root}")
    write_dataset(usable, cameras, size, args, root)
    print(f"Done: {root}")
    if args.tar:
        print(f"Tarball: {make_tar(root)}")
    if args.push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LeRobotDataset(args.repo_id, root=root).push_to_hub(private=True)
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
