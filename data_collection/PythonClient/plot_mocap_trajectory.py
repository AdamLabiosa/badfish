#!/usr/bin/env python3
"""Plot the mocap trajectory logged by xarm_mocap_auto_calibration.py.

Reads <name>_stream.csv (every mocap frame of the rigid body, tagged with the
phase of the run), saves <name>_trajectory.png next to it and opens an
interactive window (drag to rotate the 3D view).  The figure has:
  * the 3D path in Motive's frame (Y up), coloured by phase, with the averaged
    calibration samples from <name>.csv as black diamonds;
  * x / y / z over time, with the phases shaded;
  * distance from the ready pose over time (from the run start if the log
    has no move to ready), for both the mocap and the arm's reported TCP.  Distance doesn't depend on the frame, so the two
    curves should overlap whenever the tool only translates.

    python plot_mocap_trajectory.py                 # latest run
    python plot_mocap_trajectory.py path/to/xarm_mocap_..._stream.csv
    python plot_mocap_trajectory.py path/to/xarm_mocap_....csv   # its stream
    python plot_mocap_trajectory.py ../calibration/11-05-2026    # latest in dir
    python plot_mocap_trajectory.py --no-show       # just write the PNG
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit("Plotting requires: python -m pip install matplotlib numpy") from exc

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"
GAP_S = 0.05  # a jump between valid frames longer than this counts as a dropout

PHASE_COLORS = {
    "to ready": "#7f7f7f",
    "idle": "#bcbd22",
    "move": "#1f77b4",
    "sample": "#d62728",
    "return": "#2ca02c",
}
SHADED = ("to ready", "move", "sample", "return")


def phase_kind(phase: str) -> str:
    return phase.split(" ")[0] if phase.startswith(("move", "sample")) else phase


def resolve_stream(target: Optional[Path]) -> Path:
    if target is None:
        target = CALIBRATION_DIR
    if target.is_dir():
        streams = sorted(target.rglob("*_stream.csv"), key=lambda p: p.stat().st_mtime)
        if not streams:
            raise SystemExit(f"No *_stream.csv under {target}; re-run the auto calibration "
                             "(runs before the stream logger was added have none)")
        return streams[-1]
    if target.name.endswith("_stream.csv"):
        return target
    stream = target.with_name(target.stem + "_stream.csv")
    if not stream.exists():
        raise SystemExit(f"{stream} not found (runs before the stream logger was added "
                         "only have averaged samples)")
    return stream


def _f(value: Optional[str]) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def phase_runs(phases: List[str]) -> List[tuple]:
    """(start, end_exclusive, phase) for each contiguous run of one phase."""
    runs, start = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            runs.append((start, i, phases[start]))
            start = i
    return runs


def summarize(t_all: np.ndarray, t: np.ndarray, xyz: np.ndarray, n_samples: int) -> List[str]:
    duration = float(t_all[-1] - t_all[0]) if len(t_all) > 1 else 0.0
    gaps = np.diff(t)[np.diff(t) > GAP_S] if len(t) > 1 else np.array([])
    lines = [
        f"{len(t_all)} frames over {duration:.1f} s (~{len(t_all) / max(duration, 1e-9):.0f} Hz), "
        f"{len(t)} with valid tracking ({100 * len(t) / len(t_all):.1f}%)",
        f"dropouts > {GAP_S * 1000:.0f} ms: {len(gaps)}"
        + (f" (longest {gaps.max() * 1000:.0f} ms)" if len(gaps) else ""),
    ]
    if len(xyz):
        span = xyz.max(axis=0) - xyz.min(axis=0)
        lines.append("motion extent x/y/z: " + " / ".join(f"{s:.1f}" for s in span) + " mm")
        lines.append(f"farthest from start: {np.linalg.norm(xyz - xyz[0], axis=1).max():.1f} mm")
    lines.append(f"averaged samples: {n_samples}")
    return lines


def shade_phases(ax, t_all: np.ndarray, phases: List[str]) -> None:
    for a, b, phase in phase_runs(phases):
        kind = phase_kind(phase)
        if kind in SHADED:
            ax.axvspan(t_all[a], t_all[b - 1], color=PHASE_COLORS[kind], alpha=0.12, lw=0)


def plot(stream: Path, show: bool) -> Path:
    rows = load_csv(stream)
    if not rows:
        raise SystemExit(f"{stream} has no frames")
    name = stream.name[: -len("_stream.csv")]
    samples = load_csv(stream.with_name(name + ".csv"))

    t_all = np.array([_f(r["time_s"]) for r in rows])
    phases_all = [r["phase"] for r in rows]
    valid = [r for r in rows if r["tracking_valid"] == "1" and not math.isnan(_f(r["x_m"]))]
    if not valid:
        raise SystemExit("No frames with valid tracking in the stream")
    t = np.array([_f(r["time_s"]) for r in valid])
    xyz = np.array([[_f(r[k]) * 1000 for k in ("x_m", "y_m", "z_m")] for r in valid])
    phases = [r["phase"] for r in valid]
    # Reference for distances: the ready pose, taken from the *last* frame parked
    # there before the first move (the first ones can still be settling, and the
    # arm's cached report lags by up to a few hundred ms).
    idle = [i for i, ph in enumerate(phases) if ph == "idle"]
    first_move = next((i for i, ph in enumerate(phases)
                       if ph.startswith(("move", "sample", "return"))), len(phases))
    idle = [i for i in idle if i < first_move]
    ref_i = idle[-1] if idle else next(
        (i for i, ph in enumerate(phases) if ph not in ("connect", "to ready")), 0)

    summary = summarize(t_all, t, xyz, len(samples))
    print(stream)
    for line in summary:
        print("  " + line)

    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(f"{name}\n" + "   |   ".join(summary), fontsize=9)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])

    # 3D path, one line per contiguous phase run (joined to the next run).
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    labelled = set()
    for a, b, phase in phase_runs(phases):
        kind = phase_kind(phase)
        seg = xyz[a:min(b + 1, len(xyz))]
        ax3.plot(*seg.T, color=PHASE_COLORS.get(kind, "#9467bd"), lw=2,
                 label=None if kind in labelled else kind)
        labelled.add(kind)
    ax3.scatter(*xyz[ref_i], color="#ff7f0e", s=40,
                label="ready" if ref_i else "start", depthshade=False)
    if samples:
        sx = np.array([[_f(s[k]) * 1000 for k in ("mocap_x_m", "mocap_y_m", "mocap_z_m")]
                       for s in samples])
        ax3.scatter(*sx.T, color="k", marker="D", s=50, label="averaged samples",
                    depthshade=False)
        for s, p in zip(samples, sx):
            ax3.text(*p, f" {s['sample']}", fontsize=8)
    span = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 10.0)  # >=10 mm per axis
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    half = span.max() / 2 * 1.1
    for setter, c in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), center):
        setter(c - half, c + half)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=20, azim=-60, vertical_axis="y")  # Motive is Y-up
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    ax3.set_zlabel("z (mm)")
    ax3.set_title("Mocap path (Motive frame)")
    ax3.legend(loc="upper left", fontsize=8)

    ax_xyz = fig.add_subplot(grid[0, 1])
    for i, axis in enumerate("xyz"):
        ax_xyz.plot(t, xyz[:, i], lw=1.2, label=axis)
    shade_phases(ax_xyz, t_all, phases_all)
    ax_xyz.set_ylabel("mm")
    ax_xyz.set_title("Position over time (shaded: grey to ready, blue move, red sample, "
                     "green return)", fontsize=9)
    ax_xyz.legend(fontsize=8)
    ax_xyz.grid(alpha=0.3)

    ax_d = fig.add_subplot(grid[1, 1], sharex=ax_xyz)
    ax_d.plot(t, np.linalg.norm(xyz - xyz[ref_i], axis=1), lw=1.2, label="mocap")
    arm = np.array([[_f(r["time_s"])] + [_f(r[k]) for k in ("arm_x_mm", "arm_y_mm", "arm_z_mm")]
                    for r in rows])
    arm = arm[~np.isnan(arm).any(axis=1)]
    if len(arm):
        # Same reference time as the mocap.
        ref = arm[min(np.searchsorted(arm[:, 0], t[ref_i]), len(arm) - 1), 1:]
        ax_d.plot(arm[:, 0], np.linalg.norm(arm[:, 1:] - ref, axis=1), "--", lw=1.2,
                  label="arm TCP (reported, few Hz)")
    shade_phases(ax_d, t_all, phases_all)
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("mm")
    ax_d.set_title(f"Distance from {'ready' if ref_i else 'start'} (frame-independent; matches the arm when the "
                   "tool only translates)", fontsize=9)
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)

    fig.tight_layout()
    out = stream.with_name(name + "_trajectory.png")
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", type=Path,
                        help="stream CSV, samples CSV, or directory (default: latest run)")
    parser.add_argument("--no-show", action="store_true",
                        help="only save the PNG, don't open a window")
    args = parser.parse_args()
    plot(resolve_stream(args.target), show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
