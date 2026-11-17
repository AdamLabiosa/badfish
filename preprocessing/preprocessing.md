# Preprocessing: recordings → LeRobot dataset for pi0.5

`convert_to_lerobot.py` turns demos from `data_collection/recordings/` into a
[LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset. That is the format
`lerobot-train --policy.type=pi05` fine-tunes on. The output is one self-contained folder
(a few MB per 30 s demo per camera at 320x180), easy to copy to CHTC or push to the Hugging Face Hub.

## Setup (separate venv: lerobot needs Python ≥ 3.12 and pulls in torch)

```bash
uv venv --python 3.12 preprocessing/.venv
uv pip install --python preprocessing/.venv/bin/python -r preprocessing/requirements.txt
```

## Convert

```bash
cd preprocessing
# check first: lists each demo, the time kept, and why any are skipped
.venv/bin/python convert_to_lerobot.py ../data_collection/recordings/11-12-2026/test_06.csv \
    --task "pick up the fish" --dry-run
.venv/bin/python convert_to_lerobot.py ../data_collection/recordings/11-12-2026/test_06.csv \
    --task "pick up the fish" --repo-id badfish/fish_v0 --tar
```

This writes `preprocessing/datasets/badfish/fish_v0/` (git-ignored) and, with `--tar`, `fish_v0.tar`.
Pass several CSVs, or a folder to take every recording in it (only demos with a `_meta.json`).

## What's in it

| Key | Content |
|---|---|
| `observation.images.<camera>` | RGB video per camera, resized to `--image-size` (default 320x180; pi0.5 resizes to 224 anyway) |
| `observation.state` | Flange pose at frame t, in the arm base frame: `[x, y, z (mm), r00, r10, r20, r01, r11, r21]`. The last six are the first two columns of the rotation matrix: continuous, unlike roll/pitch/yaw. |
| `action` | Flange pose at frame t+1 (absolute target, same 9 numbers) |
| `task` | Instruction, used as pi0.5's prompt |
| `meta/badfish_sources.json` | Which recording, time window and calibration each episode came from |

- **Rate:** 30 fps, the camera's rate. Mocap (~120 Hz) is interpolated to the video frame times.
- **Pose:** computed exactly like `rollout_trajectory.py` does: the recording's own calibration, still
  time at the start/end trimmed (`--still-*`), jitter smoothed (`--smooth`). Demos with a tracking gap
  over `--max-gap` are skipped.
- **Stats:** `meta/stats.json` is recomputed with the q01/q99 quantiles pi0.5 normalizes with by default.
- **Policy output:** `pose_format.vec_to_pose()` turns a 9-number action back into a 4x4 flange pose
  for IK.

## Instructions (`task`)

`--task` applies to every demo. For different tasks, use `--tasks-file tasks.json` with
`{"test_06": "...", "pick_*": "..."}` (names or globs). A `"task"` key in a demo's `_meta.json` also works.

## Cameras / the second view

Cameras are found per demo: `<name>.mp4` is named `--main-camera` (default `main`, so
`observation.images.main`). Every `<name>_<cam>.mp4` adds `observation.images.<cam>`. A
`"videos": {"wrist": "x.mp4", "scene": "x_scene.mp4"}` entry in `_meta.json` overrides both.
All demos in a dataset must have the same cameras (`--cameras` picks a subset).

**Requirement for the recorder:** each video must be on the recording's clock, with frame k
captured at t ≈ k / fps from the first mocap sample. This is what `record_mocap_camera.py` does now
for its one camera. A second camera written the same way needs no changes here.

Name cameras the way the deployed robot will (e.g. `--main-camera wrist`): the policy expects the
same keys and image size at inference.

## Sending to CHTC

Either works:
- **Tarball:** copy `datasets/badfish/<name>.tar` to the submit node (put it in `/staging` if it is over
  ~1 GB), untar it in the job, and train with
  `--dataset.repo_id=badfish/<name> --dataset.root=<untarred folder>`.
- **Hub:** `--repo-id <hf_user>/<name> --push-to-hub` (private, needs `hf auth login`). Then just
  `--dataset.repo_id=<hf_user>/<name>`. The job needs an HF token anyway: pi0.5's PaliGemma
  tokenizer is gated.

Training starting point (see the [LeRobot pi0.5 docs](https://huggingface.co/docs/lerobot/pi05)):
`lerobot-train --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --dataset.repo_id=...`.
Use the same lerobot version on CHTC as here (0.7.0) so the dataset format matches.

## Status and next steps (as of 2026-11-17)

- **Tested:** converted `11-12-2026/test_06` (1 episode, 1063 frames) and loaded it back with
  lerobot 0.7.0. Image shapes, q01/q99 stats and `action[t] == state[t+1]` all check out. No training
  run has been tried yet.
- **Which data to use:** only `test_06` is trusted so far. The other `11-05` / `11-12` demos are of
  unknown quality, and `10-27` has no `_meta.json`, so it is never picked up.
- **Next:** demos will be re-recorded with a **second camera**. `record_mocap_camera.py` only records
  one camera so far. It needs a second capture loop writing `<name>_<cam>.mp4` on the same clock
  (see "Cameras / the second view"), ideally also listing `"videos"` and `"task"` in `_meta.json`.
- **Step labels:** `record_mocap_camera.py --task "..." --steps A B C` now saves `task` and
  `steps` (start times, SPACE while recording) in `_meta.json`. The converter uses `task` as the
  label but ignores `steps` for now. To train on per-step prompts, add a flag that sets each frame's
  `task` to the current step (LeRobot stores the task per frame). The open pi0.5 can't predict the
  step itself, so at run time something else has to switch the prompt.
- **Things to watch:**
  - **Hand in view:** the current camera sees the demonstrator's hand holding the end effector. The
    hand isn't there when the arm runs the policy, so keep hands out of view if possible.
  - **Camera names:** decide them before collecting the real dataset (`--main-camera`); renaming later
    means re-converting, and the deployed policy needs the same keys.
  - **Dataset size:** one demo only exercises the pipeline. Fine-tuning needs many demos per task.
  - **Actions are the demo poses:** they are not the joint-limit-adjusted poses `rollout_trajectory.py`
    executes. Fine while demos stay in reach (watch the recorder's status line).
  - **Optional training choices:** `--policy.use_relative_actions=true` works with this pose format.
    If you use it, recompute the action stats in relative space
    (`lerobot-edit-dataset ... --operation.type recompute_stats --operation.relative_action true`).
- **On macOS:** a harmless `objc ... AVFFrameReceiver is implemented in both` warning appears
  because Homebrew's ffmpeg and PyAV's bundled copy both load.
