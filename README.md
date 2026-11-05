# Badfish repository

This repo should contain all of the data, preprocessing, and training data for the Badfish VLA-based system.

# Data collection: calibration, demo recording and replay

The workflow is:

1. **Calibrate** the motion-capture frame against the xArm 6 (once per session, or whenever
   Motive's ground plane is re-set or the end effector's markers move).
2. **Record** a demonstration: hold the end effector by hand and move it. Motive tracks its rigid
   body while a camera records video.
3. **Replay** the demonstration on the arm: the end effector is mounted back on the arm, which
   follows the recorded trajectory.

All scripts live in `data_collection/PythonClient/` and are run from that folder. Data is written to
`data_collection/calibration/<MM-DD-YYYY>/` and `data_collection/recordings/<MM-DD-YYYY>/`. Both
folders are git-ignored.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r data_collection/PythonClient/requirements.txt
cd data_collection/PythonClient
```

Network (robot ethernet, 192.168.1.x):

| Device | Address |
|---|---|
| Motive / NatNet server | `192.168.1.1` |
| xArm controller | `192.168.1.103` |
| This computer | any static `192.168.1.x`, e.g. `192.168.1.15` (picked automatically) |

In Motive, enable data streaming (multicast) and make sure the end-effector rigid body
(streaming ID **7**) is tracked. If the scripts can't connect, `python find_mocap.py` lists the
correct `--client-ip` / `--server-ip`.

### Flags shared by the scripts

| Flag | Meaning |
|---|---|
| `rigid_id` (positional) | Motive streaming ID of the end-effector rigid body (default `7`). In `rollout_trajectory.py` it's `--rigid-id`. |
| `--server-ip` | Motive address (default `192.168.1.1`). |
| `--client-ip` | This computer's address on the mocap network (default: detected automatically). |
| `--unicast` | Use NatNet unicast instead of multicast; must match Motive's streaming setting. |
| `--arm-ip` | xArm controller address (default `192.168.1.103`). |
| `--tool-length` | **Required** for arm motion. How far the end effector sticks out from the flange along its axis, in mm (`300` for the current end effector). Used for collision checks. |
| `--output-dir`, `--name` | Where to write the files and their base name (default: today's folder and a timestamp). |

**Safety:** whenever the arm is connected, **SPACE** (or Ctrl+C) is a software e-stop. It goes over
the network, so always keep the hardware e-stop within reach.

---

## 1. Calibration

### 1a. Collect calibration samples (arm moves automatically)

With the end effector mounted on the arm:

```bash
python xarm_mocap_auto_calibration.py --dry-run --tool-length 300 --min-z ready \
    --z-range 0 200 --tilt 30 --roll 90 --per-point 1        # preview the plan, nothing moves
python xarm_mocap_auto_calibration.py --tool-length 300 --min-z ready \
    --z-range 0 200 --tilt 30 --roll 90 --per-point 1
```

The arm moves to a "ready" pose (tool pointing forward) and prints the plan. Press Enter to
run it. The arm then visits a grid of positions around the ready pose, tilting and rolling the tool
at each one, and averages the mocap pose at every stop. Each move is checked against a kinematic
and collision model before anything moves.

Outputs: `xarm_mocap_<time>.csv` (one row per sample), `_stream.csv` (every mocap frame) and
`_config.json`.

| Flag | What it does |
|---|---|
| `--dry-run` | Print the planned poses and any skipped ones; doesn't connect to anything. |
| `--x-range / --y-range / --z-range MIN MAX` | Size of the sampled box relative to the ready position, mm. Bigger boxes give a better calibration over a larger workspace. |
| `--grid N` | Points per axis (N³ positions). |
| `--per-point N` | How many orientations to record at each position. |
| `--tilt DEG`, `--roll DEG` | How far the tool tilts and rolls at each sample. Rotation is needed to solve the marker-to-tool offset; 30° tilt and 90° roll work well. |
| `--min-z MM\|ready` | Floor for the flange and tool tip; `ready` = no lower than at the ready pose. |
| `--speed`, `--mvacc` | Linear speed (mm/s, default 150) and acceleration of the moves. |
| `--settle S`, `--mocap-frames N` | Wait time before each sample and number of mocap frames averaged. |
| `--max-wrist-rotation DEG` | Max J4/J6 turn from the ready pose (cable wrap), default 120. |
| `--table-z`, `--table-margin`, `--tool-radius`, `--self-margin`, `--base-margin` | Collision-model settings (table height, clearances, tool thickness). |
| `--no-mocap` | Run the moves without Motive (arm-only test; records nothing). |
| `-y` | Skip the Enter prompts. |

Manual alternative: `python xarm_mocap_calibration.py` lets you jog the arm with the keyboard
(w/s/a/d/r/f to move, u/o/i/k/j/l to rotate) and press SPACE to record each sample. It has no
collision checks, so jog carefully.

### 1b. Solve the calibration

```bash
python solve_mocap_calibration.py                 # newest samples CSV under calibration/
python solve_mocap_calibration.py ../calibration/11-05-2026/xarm_mocap_20261105_114641.csv
```

Fits the transform from Motive's frame to the arm base frame, the offset from the markers to the
tool, and a mocap scale factor. It prints the residual error; around 2 mm / 1° RMS is good. It
writes `<name>_solution.json`, which the recorder and replay scripts load automatically (newest
first).

| Flag | What it does |
|---|---|
| `target` | Samples CSV, its `_stream.csv`, or a folder (newest). |
| `--max-std MM` | Drop samples whose mocap position jittered more than this (arm still moving). |

### 1c. (Optional) Plot the calibration run

```bash
python plot_mocap_trajectory.py            # newest run; --no-show only writes the PNG
```

---

## 2. Record a demonstration

```bash
python record_mocap_camera.py --list-cameras      # find the camera index (saves a snapshot of each)
python record_mocap_camera.py --name pick_cup_01
```

1. A status line shows whether the rigid body is tracked and where the flange would be in the arm
   frame. It also says whether the arm can reach that pose (`reachable`, `OUT OF REACH`, or
   `UNSAFE: <reason>`).
2. Press **Enter**; after a short countdown, recording starts.
3. **Hold the end effector still for 1–2 s**, do the motion, hold still again.
4. Press **Enter** (or Ctrl+C) to stop and save.

Outputs in `recordings/<date>/`: `<name>.csv` (every mocap frame, raw and in the arm frame),
`<name>.mp4` (video synchronized to the CSV) and `<name>_meta.json` (settings and which calibration
was used).

| Flag | What it does |
|---|---|
| `--name` | Base file name (default `demo_<timestamp>`). Refuses to overwrite an existing recording. |
| `--camera N` | Camera index (default `0`) or a stream URL. Indices can change when devices are plugged in, so re-check with `--list-cameras`. |
| `--list-cameras` | Probe cameras: resolution, actual frame rate, and a snapshot of each (to tell them apart). |
| `--width`, `--height`, `--fps` | Request a camera resolution / frame rate. |
| `--countdown S` | Delay between pressing Enter and recording (default 3 s), so you can pick up the end effector. |
| `--calibration PATH` | Calibration `_solution.json` to use (default: newest). |
| `--no-calibration` | Record raw mocap only (no arm-frame columns or reach check). |
| `--tool-length MM` | Tool length for the reach check (default: from the calibration run). |

To view a recording with its 3D path: `python visualize_run.py <name>`.

---

## 3. Replay a demonstration on the arm

**Always dry-run first.** It plans and checks everything without touching the arm:

```bash
python rollout_trajectory.py ../recordings/11-05-2026/pick_cup_01.csv --tool-length 300 --dry-run
```

Then mount the end effector on the arm and run it:

```bash
python rollout_trajectory.py ../recordings/11-05-2026/pick_cup_01.csv --tool-length 300 \
    --speed-scale 0.5 --mocap
```

Press Enter to move (slowly) to the demo's start pose, then Enter again to replay.

How it works:
- Every recorded mocap pose is converted, via the calibration, into the flange pose that puts the
  end effector where it was held.
- The trajectory is smoothed, solved with inverse kinematics, and checked at every sample: joint
  limits, tool-to-arm clearance, folding onto the base, wrist cable wrap, singularities.
- It is then time-scaled to respect the speed limits and streamed to the arm at 100 Hz in servo mode.

If anything fails, the dry-run says when in the demo it happens and why. The plan is written to
`<name>_plan.csv`, and the replay log to `<name>_rollout_<time>.csv`; with `--mocap`, the tracked
replay goes to `_mocap.csv`.

**Speed:** the replay runs at `--speed-scale` of the demo's speed. Where that would exceed a
limit, it slows down only that part of the trajectory. The dry-run prints both, e.g.
`49% of demo speed; slows to 8% for 38.5 s where limited by joint speed`.

| Flag | What it does |
|---|---|
| `recording` | Recording CSV (or its `.mp4` / `_meta.json`, or a folder for the newest). Default: newest recording. |
| `--dry-run` | Plan and check only; doesn't connect to the arm. |
| `--speed-scale F` | Replay speed as a fraction of the demo's speed (default `0.5`; `0.25` = very slow, `1` = real time). |
| `--max-joint-speed DEG/S` | Joint speed limit (default 60, arm max 180). Slow stretches during big wrist rotations are set by this. |
| `--max-joint-acc DEG/S²` | Joint acceleration limit (default 300). |
| `--max-speed MM/S` | Flange and tool-tip speed limit (default 250). |
| `--mocap` | Track the end effector in Motive during the replay and report how far the replay was from the demo. |
| `--control joint\|cartesian` | `joint` (default): stream the planned joint angles. `cartesian`: stream end-effector poses and let the arm's built-in IK solve them; near a singularity the controller may move differently or stop. |
| `--trim START END` | Only replay this part of the recording (seconds). |
| `--no-auto-trim` | Keep the still time at the start/end (by default it is trimmed, leaving `--still-pad` s, default 0.5). |
| `--offset DX DY DZ` | Shift the whole trajectory in the arm base frame, mm. |
| `--smooth S` | Smoothing of mocap jitter (Gaussian sigma, default 0.04 s; 0 = off). |
| `--max-gap S` | Longest tracking dropout to interpolate over (default 0.1 s). |
| `--max-wrist-rotation DEG` | Max J4/J6 turn from 0 (default 120). Raise it (e.g. `210`) for demos that roll or tilt the end effector a lot, **if the wrist cables have the slack**. |
| `--table-margin MM` | Turn on the table-clearance check (off by default: the demo itself showed the end effector fits). |
| `--min-z MM` | Floor for the flange and tool tip, mm in the arm base frame. |
| `--singular-margin DEG` | How close J5 may get to 0°/180° (wrist singularity), default 10. |
| `--approach-speed DEG/S` | Joint speed for the move to the start pose (default 20). |
| `--calibration PATH` | Calibration to use (default: the one the recording was made with). |
| `--rate HZ` | Servo command rate (default 100). |
| `--no-controller-collision` | Skip setting up the arm controller's own self-collision detection. |

### Tips and common errors

- **"starts/ends with a joint moving … the arm will jerk":** hold the end effector still at the
  start and end of the demo, or `--trim` to a still moment.
- **"wrist turns N deg from ready (cable wrap)":** the demo needs more wrist rotation than allowed.
  This often happens when the end effector is tilted to point along the forearm (the wrist
  singularity): the arm must spin J4/J6 about 180° to follow it. Raise `--max-wrist-rotation` if
  the cables allow, or re-record.
- **"tracking gap":** markers were occluded; `--trim` around it or raise `--max-gap`.
- **"unreachable" / "IK jumped":** the demo left the arm's workspace. Watch the recorder's status
  line while demonstrating.
- If Motive's ground plane is re-set, the old calibration no longer applies: recalibrate.
