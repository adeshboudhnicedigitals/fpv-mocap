# M4 pull-out first target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `m4_pullout_single_shot.mp4` into a first-person arms animation on the Lone Asset M4 rig, exported as `.glb`.

**Architecture:** Laptop-side CPU stages (`frames`, `camera`, `align`, `overlay`) are plain Python with pytest-tested maths. GPU stages (WiLoR, SAM 2, Depth Anything V2, FoundationPose) run in one script on Kaggle/Colab and exchange only `rgb/`, `camera.json` and `.npz` files, validated by `mocap/contracts.py`. Retargeting runs in headless Blender.

**Tech Stack:** Python 3.11 (uv), numpy, scipy, OpenCV, pytest; Blender (`bpy`); WiLoR-mini, SAM 2, Depth Anything V2 metric, FoundationPose on a CUDA GPU.

## Global Constraints

- Python is `>=3.11,<3.12`; run everything through `uv run`.
- All tracking output is OpenCV camera space (+x right, +y down, +z forward), in metres, until `blender/retarget.py`.
- Stage files per plan.md: `camera.json`, `gun.npz`, `hands.npz`, `aligned.npz`, under `clips/m4_pullout/`.
- Gun pose is never smoothed. Fingers are smoothed with Savitzky-Golay.
- `valid = False` frames are never an error. Missing or malformed input files fail with a clear message.
- Gun mesh for tracking is `m4a1_single.glb` (repo root), exported to `rigs/m4_gun.obj` by `blender/export_gun_mesh.py`, in metres, in the glb's Blender world frame (see `docs/rig-notes.md`). Arms rig: `fps_arms.glb`.
- Camera: `--fov 90` guess. The clip's camera is locked.
- The footage belongs to someone else; the animation is reference-only until its licence is checked.
- Git commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

Spec: `docs/superpowers/specs/2026-11-16-m4-pullout-first-target-design.md`. One refinement to the spec's test line: because the camera sits on the rig's camera bone, the CV-to-Blender conversion is `T_world = cam_world @ FLIP @ T_cam_obj` (Task 2), not `M @ T @ M^-1`. It is tested against the known CV-to-Blender axis mapping.

## File Structure

```
mocap/transforms.py        pose maths: invert/relative, CV camera -> Blender world
mocap/contracts.py         load/save/validate hands.npz and gun.npz
mocap/align.py             grip weight, wrist locking, gun fill-in, finger smoothing, run()
mocap/overlay.py           project and draw gun axes + hands onto frames -> overlay.mp4
mocap/cli.py               + `align`, `overlay` subcommands
blender/list_bones.py      prints objects/bones of a .fbx
blender/export_gun_mesh.py exports the gun mesh as .obj in metres
blender/retarget.py        aligned.npz -> animated fps_arms + m4a1 -> .blend + .glb
notebooks/gpu_stages.py    percent-format script for Kaggle/Colab
clips/m4_pullout/          meta.json, SOURCE.md, camera.json, (npz outputs)
tests/test_transforms.py, test_contracts.py, test_align.py, test_overlay.py, test_cli_align.py
```

---

### Task 1: Prepare the clip, dependencies and docs

**Files:**
- Modify: `pyproject.toml`, `docs/plan.md`, `docs/superpowers/specs/2026-11-16-m4-pullout-first-target-design.md`
- Create: `clips/m4_pullout/meta.json`, `clips/m4_pullout/SOURCE.md`, `clips/m4_pullout/camera.json` (generated)

**Interfaces:**
- Produces: `clips/m4_pullout/rgb/00001.png ...` (137 frames), `camera.json` (`fov_guess`, 1920x1080, fx 960), `meta.json` with keys `init_frame` (int, 1-based frame number), `gun_click` (`[x, y]` pixels on the gun at the init frame), `gun_mesh` (path string).

- [ ] **Step 1: Add scipy**

Run: `uv add scipy`
Expected: `pyproject.toml` lists `scipy`; `uv.lock` updates.

- [ ] **Step 2: Extract frames and write camera.json**

Run:
```bash
uv run mocap frames m4_pullout m4_pullout_single_shot.mp4 --fps 60
uv run mocap camera m4_pullout --fov 90
```
Expected: `137 frames at 60 fps -> clips\m4_pullout\rgb` and `1920x1080, fx 960.0, cx 960.0, cy 540.0, horizontal FOV 90.0 deg`.

- [ ] **Step 3: Pick the init frame by eye**

Run: `ffmpeg -v error -y -i m4_pullout_single_shot.mp4 -vf "select='between(n,40,100)*not(mod(n,6))',scale=640:-1,tile=4x3" -frames:v 1 clips/m4_pullout/init_choice.png`
Open `clips/m4_pullout/init_choice.png`. Each tile is every 6th frame from n=40 (0-based) to 100, left to right, top to bottom. Pick the first tile where the rifle is mostly in view. Frame number for `meta.json` is `n + 1` (the `rgb/` files are 1-based). Estimate the pixel `[x, y]` of a point on the rifle body in that frame at 1920x1080 by opening `clips/m4_pullout/rgb/<NNNNN>.png`.

- [ ] **Step 4: Write meta.json and SOURCE.md**

`clips/m4_pullout/meta.json` (replace the three values with the ones chosen in Step 3):
```json
{
  "init_frame": 70,
  "gun_click": [1100, 800],
  "gun_mesh": "../lone-asset/content/weapons/m4/source/M4.fbx"
}
```
`clips/m4_pullout/SOURCE.md`:
```markdown
# m4_pullout

- File: m4_pullout_single_shot.mp4 (1920x1080, 60 fps, 137 frames), rendered game footage, locked camera.
- Origin: provided by the developer; original author and licence not recorded. Reference only until checked.
- Camera: guessed 90 degree horizontal FOV.
```

- [ ] **Step 5: Update plan.md and the spec to the new milestone order**

In `docs/plan.md`, replace the whole "Milestones" section (from `## Milestones` through the line `  align and retarget stages.`) with:
```markdown
## Milestones

Each one ends with something to look at. First target: `clips/m4_pullout` (see
`docs/superpowers/specs/2026-11-16-m4-pullout-first-target-design.md`).

- M0 Setup (done): uv project, `frames` and `camera` stages.
- M1 Hands: WiLoR on a GPU -> `hands.npz`. Check: overlay shows both hands' keypoints.
- M2 Gun: SAM 2 + Depth Anything + FoundationPose with the M4 mesh -> `gun.npz`. Check: overlay axes follow the gun.
- M3 Align: grip locking and smoothing, with unit tests. Check: gripping hands stay on the gun.
- M4 Retarget: `retarget.py` on the M4 rig. Check: the `.blend` plays the clip; the `.glb` imports into Godot.
- M5 In game: the clip plays on the rifle in Lone Asset.
- M6 Marker/PnP path (`gun_markers.py`) for self-recorded clips, and moving-camera footage.
```
In the spec, replace the bullet beginning `- `pytest` on the pure maths in `align.py`: CV-to-Blender basis change (`M @ T @ M^-1`), grip` with:
```markdown
- `pytest` on the pure maths: camera-to-Blender conversion (`cam_world @ FLIP @ T`, in `transforms.py`), grip
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock docs clips/m4_pullout/meta.json clips/m4_pullout/SOURCE.md clips/m4_pullout/camera.json
git commit -m "feat: prepare m4_pullout clip, scipy, reorder milestones"
```
(Frames are gitignored.)

---

### Task 2: Pose maths (`transforms.py`)

**Files:**
- Create: `mocap/transforms.py`, `tests/test_transforms.py`

**Interfaces:**
- Produces:
  - `FLIP: np.ndarray` (4,4) diag(1,-1,-1,1), OpenCV camera axes to Blender camera axes.
  - `relative(T_a: ndarray (...,4,4), T_b: ndarray (...,4,4)) -> ndarray` = `inv(T_a) @ T_b`.
  - `cam_to_world(cam_world: ndarray (4,4), T_cam_obj: ndarray (...,4,4)) -> ndarray (...,4,4)` = `cam_world @ FLIP @ T_cam_obj`.

- [ ] **Step 1: Write the failing tests**

`tests/test_transforms.py`:
```python
import numpy as np

from mocap.transforms import FLIP, cam_to_world, relative


def _pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_relative_recovers_the_offset():
    gun = _pose(_rot_x(30), [0.1, 0.2, 0.9])
    offset = _pose(_rot_x(-10), [0.0, -0.05, 0.12])
    assert np.allclose(relative(gun, gun @ offset), offset)


def test_relative_is_batched():
    gun = np.stack([_pose(_rot_x(10 * i), [0, 0, 1 + i]) for i in range(4)])
    offset = _pose(np.eye(3), [0, 0.1, 0])
    out = relative(gun, gun @ offset)
    assert out.shape == (4, 4, 4)
    assert np.allclose(out, offset)


def test_camera_forward_down_and_right_map_to_blender_axes():
    # Blender camera bone looking along world +Y with +Z up = rotate 90 deg about X.
    cam_world = _pose(_rot_x(90), [0, 0, 0])
    forward = cam_to_world(cam_world, _pose(np.eye(3), [0, 0, 2]))[:3, 3]
    down = cam_to_world(cam_world, _pose(np.eye(3), [0, 1, 0]))[:3, 3]
    right = cam_to_world(cam_world, _pose(np.eye(3), [1, 0, 0]))[:3, 3]
    assert np.allclose(forward, [0, 2, 0])  # +Z forward -> +Y forward
    assert np.allclose(down, [0, 0, -1])  # +Y down -> -Z
    assert np.allclose(right, [1, 0, 0])


def test_cam_to_world_moves_with_the_camera():
    cam_world = _pose(_rot_x(90), [1, 2, 3])
    out = cam_to_world(cam_world, _pose(np.eye(3), [0, 0, 2]))[:3, 3]
    assert np.allclose(out, [1, 4, 3])


def test_flip_is_a_pure_reflection_pair():
    assert np.allclose(FLIP @ FLIP, np.eye(4))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_transforms.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'mocap.transforms'`.

- [ ] **Step 3: Implement**

`mocap/transforms.py`:
```python
"""Pose maths shared by align, overlay and retarget. Poses are 4x4 matrices, metres."""

import numpy as np

# OpenCV camera (+x right, +y down, +z forward) to Blender camera (+x right, +y up, -z forward).
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def relative(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Pose of b expressed in a's frame: inv(T_a) @ T_b. Works on batches."""
    return np.linalg.inv(T_a) @ T_b


def cam_to_world(cam_world: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """Object pose in the world, given the camera's world matrix in Blender convention."""
    return cam_world @ FLIP @ T_cam_obj
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_transforms.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add mocap/transforms.py tests/test_transforms.py
git commit -m "feat: pose maths for camera-to-world conversion"
```

---

### Task 3: Data contracts (`contracts.py`)

**Files:**
- Create: `mocap/contracts.py`, `tests/test_contracts.py`

**Interfaces:**
- Produces:
  - `class ContractError(ValueError)`.
  - `SIDES = ("left", "right")`.
  - `save_gun(path: Path, gun: dict[str, ndarray])`, `load_gun(path: Path) -> dict[str, ndarray]` with keys `T_cam_gun` (N,4,4), `valid` (N,) bool, `score` (N,).
  - `save_hands(path: Path, hands: dict[str, dict[str, ndarray]])`, `load_hands(path: Path) -> dict[str, dict[str, ndarray]]` keyed by side, then `T_cam_wrist` (N,4,4), `finger_rot` (N,15,3,3), `keypoints` (N,21,3), `valid` (N,) bool, `score` (N,). On disk the keys are flat: `left_T_cam_wrist`, `right_valid`, etc.

- [ ] **Step 1: Write the failing tests**

`tests/test_contracts.py`:
```python
from pathlib import Path

import numpy as np
import pytest

from mocap.contracts import ContractError, load_gun, load_hands, save_gun, save_hands


def _gun(n: int) -> dict:
    return {"T_cam_gun": np.tile(np.eye(4), (n, 1, 1)), "valid": np.ones(n, bool), "score": np.ones(n)}


def _hand(n: int) -> dict:
    return {
        "T_cam_wrist": np.tile(np.eye(4), (n, 1, 1)),
        "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
        "keypoints": np.zeros((n, 21, 3)),
        "valid": np.ones(n, bool),
        "score": np.ones(n),
    }


def test_gun_round_trip(tmp_path: Path):
    save_gun(tmp_path / "gun.npz", _gun(5))
    out = load_gun(tmp_path / "gun.npz")
    assert out["T_cam_gun"].shape == (5, 4, 4)
    assert out["valid"].dtype == bool


def test_hands_round_trip(tmp_path: Path):
    save_hands(tmp_path / "hands.npz", {"left": _hand(4), "right": _hand(4)})
    out = load_hands(tmp_path / "hands.npz")
    assert set(out) == {"left", "right"}
    assert out["right"]["finger_rot"].shape == (4, 15, 3, 3)


def test_missing_key_names_the_key(tmp_path: Path):
    hands = {"left": _hand(3), "right": _hand(3)}
    del hands["right"]["keypoints"]
    save_hands(tmp_path / "hands.npz", hands)
    with pytest.raises(ContractError, match="right_keypoints"):
        load_hands(tmp_path / "hands.npz")


def test_wrong_shape_is_rejected(tmp_path: Path):
    gun = _gun(3)
    gun["T_cam_gun"] = np.zeros((3, 3, 3))
    save_gun(tmp_path / "gun.npz", gun)
    with pytest.raises(ContractError, match="T_cam_gun"):
        load_gun(tmp_path / "gun.npz")


def test_frame_count_must_match_across_fields(tmp_path: Path):
    gun = _gun(3)
    gun["score"] = np.ones(4)
    save_gun(tmp_path / "gun.npz", gun)
    with pytest.raises(ContractError, match="score"):
        load_gun(tmp_path / "gun.npz")


def test_missing_file_is_a_contract_error(tmp_path: Path):
    with pytest.raises(ContractError, match="gun.npz"):
        load_gun(tmp_path / "gun.npz")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'mocap.contracts'`.

- [ ] **Step 3: Implement**

`mocap/contracts.py`:
```python
"""Load, save and validate the .npz files the stages exchange (see docs/plan.md, Data contracts)."""

from pathlib import Path

import numpy as np

SIDES = ("left", "right")
GUN_SHAPES = {"T_cam_gun": (4, 4), "valid": (), "score": ()}
HAND_SHAPES = {
    "T_cam_wrist": (4, 4),
    "finger_rot": (15, 3, 3),
    "keypoints": (21, 3),
    "valid": (),
    "score": (),
}


class ContractError(ValueError):
    """A stage file is missing, incomplete or has the wrong shape."""


def _load(path: Path, prefix: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    if not Path(path).exists():
        raise ContractError(f"{path} not found")
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    n: int | None = None
    for field, tail in shapes.items():
        key = f"{prefix}{field}"
        if key not in data:
            raise ContractError(f"{path}: missing {key}")
        arr = data[key]
        n = len(arr) if n is None else n
        if arr.shape != (n, *tail):
            raise ContractError(f"{path}: {key} has shape {arr.shape}, expected {(n, *tail)}")
        out[field] = arr.astype(bool) if field == "valid" else arr
    return out


def save_gun(path: Path, gun: dict[str, np.ndarray]) -> None:
    np.savez(path, **gun)


def load_gun(path: Path) -> dict[str, np.ndarray]:
    return _load(path, "", GUN_SHAPES)


def save_hands(path: Path, hands: dict[str, dict[str, np.ndarray]]) -> None:
    np.savez(path, **{f"{side}_{k}": v for side, fields in hands.items() for k, v in fields.items()})


def load_hands(path: Path) -> dict[str, dict[str, np.ndarray]]:
    return {side: _load(path, f"{side}_", HAND_SHAPES) for side in SIDES}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add mocap/contracts.py tests/test_contracts.py
git commit -m "feat: validated loaders for gun.npz and hands.npz"
```

---

### Task 4: Alignment maths (`align.py`)

**Files:**
- Create: `mocap/align.py`, `tests/test_align.py`

**Interfaces:**
- Consumes: `mocap.transforms.relative`; the dicts returned by `mocap.contracts.load_gun` / `load_hands`.
- Produces:
  - `grip_weight(rel: (N,4,4), valid: (N,) bool, window=15, eps_t=0.01, eps_r=np.radians(8), blend=9, min_valid=5) -> (N,) float in [0,1]`
  - `grip_offset(rel, weight) -> (4,4) | None` (mean pose over frames with weight >= 0.99).
  - `blend_pose(A: (N,4,4), B: (N,4,4), w: (N,)) -> (N,4,4)`: translation lerp, rotation slerp.
  - `smooth_fingers(finger_rot: (N,15,3,3), valid: (N,) bool, window=9, order=2) -> (N,15,3,3)`
  - `run(gun: dict, hands: dict) -> dict[str, ndarray]`: the `aligned.npz` contents. Keys: `T_cam_gun`, `gun_valid`, `gun_source` (int8: 0 tracked, 1 from right hand, -1 none), and per side `{side}_T_cam_wrist`, `{side}_finger_rot`, `{side}_keypoints`, `{side}_valid`, `{side}_grip_weight`.

Notes on the design: one fixed hand-in-gun offset per hand for the whole clip (enough for a pull-out where each hand keeps its grip place). The right hand (trigger hand) is the fallback for gun frames where tracking lost lock.

- [ ] **Step 1: Write the failing tests**

`tests/test_align.py`:
```python
import numpy as np

from mocap.align import blend_pose, grip_offset, grip_weight, run, smooth_fingers
from mocap.transforms import relative


def _pose(t, yaw_deg=0.0) -> np.ndarray:
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    T[:3, 3] = t
    return T


def test_grip_weight_is_high_for_a_steady_offset_and_low_when_it_moves():
    n = 60
    rel = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        rel[i, :3, 3] = [0.05, 0, 0.1] if i < 30 else [0.05 + 0.05 * (i - 30), 0, 0.1]
    w = grip_weight(rel, np.ones(n, bool))
    assert w[10] > 0.99
    assert w[50] < 0.01


def test_grip_weight_is_zero_on_invalid_frames():
    rel = np.tile(np.eye(4), (30, 1, 1))
    valid = np.ones(30, bool)
    valid[10:20] = False
    w = grip_weight(rel, valid, blend=1)
    assert np.all(w[10:20] == 0)


def test_grip_offset_none_without_a_grip():
    rel = np.tile(np.eye(4), (10, 1, 1))
    assert grip_offset(rel, np.zeros(10)) is None


def test_grip_offset_recovers_the_constant_offset():
    off = _pose([0.05, 0.0, 0.1], 20)
    rel = np.tile(off, (20, 1, 1))
    assert np.allclose(grip_offset(rel, np.ones(20)), off, atol=1e-9)


def test_blend_pose_endpoints_and_midpoint():
    A = np.stack([_pose([0, 0, 0], 0)] * 3)
    B = np.stack([_pose([2, 0, 0], 90)] * 3)
    out = blend_pose(A, B, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out[0], A[0])
    assert np.allclose(out[2], B[0])
    assert np.allclose(out[1, :3, 3], [1, 0, 0])
    assert np.allclose(out[1], _pose([1, 0, 0], 45))


def test_smooth_fingers_reduces_jitter_and_keeps_valid_rotations():
    rng = np.random.default_rng(0)
    n = 40
    from scipy.spatial.transform import Rotation

    base = Rotation.from_rotvec([0.3, 0.0, 0.0])
    noisy = np.empty((n, 15, 3, 3))
    for i in range(n):
        for j in range(15):
            noisy[i, j] = (Rotation.from_rotvec(rng.normal(0, 0.05, 3)) * base).as_matrix()
    out = smooth_fingers(noisy, np.ones(n, bool))
    to_vec = lambda m: Rotation.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().reshape(n, 15, 3)
    assert to_vec(out).std(axis=0).mean() < 0.5 * to_vec(noisy).std(axis=0).mean()
    assert np.allclose(out @ np.swapaxes(out, -1, -2), np.eye(3), atol=1e-6)


def _synthetic(n=60):
    rng = np.random.default_rng(1)
    gun = np.stack([_pose([0.0, 0.0 + 0.002 * i, 1.0], 0.2 * i) for i in range(n)])
    off_r = _pose([0.05, 0.1, 0.0])
    off_l = _pose([-0.05, 0.1, 0.3])
    right = gun @ off_r
    right[:, :3, 3] += rng.normal(0, 0.001, (n, 3))
    left = gun @ off_l
    for i in range(30, n):  # support hand leaves the gun and moves fast
        left[i, :3, 3] += [0.05 * (i - 30), 0.0, 0.0]

    def hand(T):
        return {
            "T_cam_wrist": T,
            "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
            "keypoints": np.zeros((n, 21, 3)),
            "valid": np.ones(n, bool),
            "score": np.ones(n),
        }

    return (
        {"T_cam_gun": gun, "valid": np.ones(n, bool), "score": np.ones(n)},
        {"left": hand(left), "right": hand(right)},
        gun,
        off_r,
    )


def test_run_locks_the_gripping_hand_and_releases_the_support_hand():
    gun, hands, gun_truth, off_r = _synthetic()
    out = run(gun, hands)
    assert out["right_grip_weight"][30] > 0.99
    assert np.allclose(out["right_T_cam_wrist"][30, :3, 3], (gun_truth[30] @ off_r)[:3, 3], atol=1e-6)
    assert out["left_grip_weight"][10] > 0.99
    assert out["left_grip_weight"][55] < 0.01
    assert np.allclose(out["left_T_cam_wrist"][55], hands["left"]["T_cam_wrist"][55])
    assert np.allclose(out["T_cam_gun"], gun_truth)  # gun is never smoothed
    assert np.all(out["gun_source"] == 0)


def test_run_fills_lost_gun_frames_from_the_right_hand():
    gun, hands, gun_truth, _ = _synthetic()
    gun["valid"][25:30] = False
    gun["T_cam_gun"][25:30] = np.eye(4)
    out = run(gun, hands)
    assert np.all(out["gun_source"][25:30] == 1)
    assert np.allclose(out["T_cam_gun"][25:30, :3, 3], gun_truth[25:30, :3, 3], atol=0.01)
    assert out["gun_valid"][25:30].all()


def test_run_leaves_frames_with_no_gun_and_no_hand_invalid():
    gun, hands, _, _ = _synthetic()
    gun["valid"][:5] = False
    hands["right"]["valid"][:5] = False
    out = run(gun, hands)
    assert np.all(out["gun_source"][:5] == -1)
    assert not out["gun_valid"][:5].any()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_align.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'mocap.align'`.

- [ ] **Step 3: Implement**

`mocap/align.py`:
```python
"""Join gun and hands: grip detection, wrist locking, gun fill-in and finger smoothing.

Pure maths on arrays; no files. All poses are camera space, metres.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from mocap.contracts import SIDES
from mocap.transforms import relative


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def grip_weight(
    rel: np.ndarray,
    valid: np.ndarray,
    window: int = 15,
    eps_t: float = 0.01,
    eps_r: float = np.radians(8),
    blend: int = 9,
    min_valid: int = 5,
) -> np.ndarray:
    """1 where the wrist's pose in the gun's frame barely changes over a window, else 0, blended."""
    n = len(rel)
    hard = np.zeros(n)
    half = window // 2
    for i in range(n):
        if not valid[i]:
            continue
        idx = [j for j in range(max(0, i - half), min(n, i + half + 1)) if valid[j]]
        if len(idx) < min_valid:
            continue
        dt = max(np.linalg.norm(rel[j, :3, 3] - rel[i, :3, 3]) for j in idx)
        dr = max(_angle(rel[i, :3, :3], rel[j, :3, :3]) for j in idx)
        hard[i] = float(dt < eps_t and dr < eps_r)
    weight = uniform_filter1d(hard, blend, mode="nearest")
    weight[~valid] = 0.0
    return weight


def grip_offset(rel: np.ndarray, weight: np.ndarray) -> np.ndarray | None:
    """Mean wrist-in-gun pose over the frames that are firmly gripping, or None if there are none."""
    idx = np.flatnonzero(weight >= 0.99)
    if len(idx) == 0:
        return None
    T = np.eye(4)
    T[:3, 3] = rel[idx, :3, 3].mean(axis=0)
    T[:3, :3] = Rotation.from_matrix(rel[idx, :3, :3]).mean().as_matrix()
    return T


def blend_pose(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-frame blend from A (w=0) to B (w=1): translation lerp, rotation slerp."""
    out = np.tile(np.eye(4), (len(A), 1, 1))
    out[:, :3, 3] = (1 - w)[:, None] * A[:, :3, 3] + w[:, None] * B[:, :3, 3]
    for i in range(len(A)):
        pair = Rotation.from_matrix(np.stack([A[i, :3, :3], B[i, :3, :3]]))
        out[i, :3, :3] = Slerp([0.0, 1.0], pair)(w[i]).as_matrix()
    return out


def smooth_fingers(finger_rot: np.ndarray, valid: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay on finger rotation vectors; invalid frames are interpolated first."""
    n = len(finger_rot)
    vec = Rotation.from_matrix(finger_rot.reshape(-1, 3, 3)).as_rotvec().reshape(n, 15 * 3)
    if valid.any() and not valid.all():
        frames = np.arange(n)
        for c in range(vec.shape[1]):
            vec[:, c] = np.interp(frames, frames[valid], vec[valid, c])
    if n >= window:
        vec = savgol_filter(vec, window, order, axis=0)
    return Rotation.from_rotvec(vec.reshape(-1, 3)).as_matrix().reshape(n, 15, 3, 3)


def run(gun: dict[str, np.ndarray], hands: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Produce the contents of aligned.npz from gun and hands data (see Interfaces in the plan)."""
    T_gun = gun["T_cam_gun"].copy()
    gun_valid = gun["valid"].copy()
    source = np.where(gun_valid, 0, -1).astype(np.int8)

    rel, weight, offset = {}, {}, {}
    for side in SIDES:
        h = hands[side]
        both = gun["valid"] & h["valid"]
        rel[side] = relative(gun["T_cam_gun"], h["T_cam_wrist"])
        weight[side] = grip_weight(rel[side], both)
        offset[side] = grip_offset(rel[side], weight[side])

    right = hands["right"]
    if offset["right"] is not None:
        lost = ~gun["valid"] & right["valid"]
        T_gun[lost] = right["T_cam_wrist"][lost] @ np.linalg.inv(offset["right"])
        gun_valid |= lost
        source[lost] = 1

    out: dict[str, np.ndarray] = {"T_cam_gun": T_gun, "gun_valid": gun_valid, "gun_source": source}
    for side in SIDES:
        h = hands[side]
        wrist = h["T_cam_wrist"]
        if offset[side] is not None:
            wrist = blend_pose(wrist, T_gun @ offset[side], weight[side])
        out[f"{side}_T_cam_wrist"] = wrist
        out[f"{side}_finger_rot"] = smooth_fingers(h["finger_rot"], h["valid"])
        out[f"{side}_keypoints"] = h["keypoints"]
        out[f"{side}_valid"] = h["valid"]
        out[f"{side}_grip_weight"] = weight[side]
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_align.py -v`
Expected: 9 passed. If `test_run_fills_lost_gun_frames_from_the_right_hand` fails on the 0.01 m tolerance, the cause is the right-hand jitter (1 mm) plus the offset average; investigate the offset, do not loosen the tolerance above 0.01.

- [ ] **Step 5: Commit**

```bash
git add mocap/align.py tests/test_align.py
git commit -m "feat: grip locking, gun fill-in and finger smoothing"
```

---

### Task 5: Overlay (`overlay.py`)

**Files:**
- Create: `mocap/overlay.py`, `tests/test_overlay.py`

**Interfaces:**
- Consumes: `mocap.camera.Camera` (`K`, `fps`, `width`, `height`), `mocap.frames.list_frames(rgb_dir) -> list[Path]`, the `aligned` dict from `align.run`.
- Produces:
  - `project(K: (3,3), pts: (M,3)) -> (M,2)` pixel coordinates.
  - `draw_frame(img: ndarray BGR, K, aligned: dict, i: int) -> ndarray` (copy with drawings).
  - `write_video(rgb_dir: Path, camera: Camera, aligned: dict, out_path: Path) -> int` (frames written).

What is drawn: gun axes (X red, Y green, Z blue, 0.15 m) when `gun_valid[i]`, yellow if the pose came from a hand (`gun_source == 1`); each hand's 21 keypoints as small dots, and the aligned wrist as a larger circle whose colour runs from red (free) to green (grip locked) by `grip_weight`.

- [ ] **Step 1: Write the failing tests**

`tests/test_overlay.py`:
```python
import numpy as np

from mocap.overlay import draw_frame, project

K = np.array([[960.0, 0, 960], [0, 960.0, 540], [0, 0, 1]])


def _aligned(n=2, valid=True):
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, 2, 3] = 1.0
    out = {
        "T_cam_gun": pose,
        "gun_valid": np.full(n, valid),
        "gun_source": np.zeros(n, np.int8),
    }
    for side in ("left", "right"):
        out[f"{side}_T_cam_wrist"] = pose.copy()
        out[f"{side}_keypoints"] = np.tile(np.array([0.0, 0.0, 1.0]), (n, 21, 1))
        out[f"{side}_valid"] = np.full(n, valid)
        out[f"{side}_grip_weight"] = np.ones(n)
    return out


def test_project_principal_point_and_offset():
    uv = project(K, np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]))
    assert np.allclose(uv, [[960, 540], [1440, 540]])


def test_draw_frame_marks_valid_frames_without_touching_the_input():
    img = np.zeros((1080, 1920, 3), np.uint8)
    out = draw_frame(img, K, _aligned(), 0)
    assert img.sum() == 0
    assert out.sum() > 0


def test_draw_frame_skips_invalid_frames():
    img = np.zeros((1080, 1920, 3), np.uint8)
    out = draw_frame(img, K, _aligned(valid=False), 0)
    assert out.sum() == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_overlay.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'mocap.overlay'`.

- [ ] **Step 3: Implement**

`mocap/overlay.py`:
```python
"""Draw the tracked gun and hands back onto the video so each stage can be judged by eye."""

from pathlib import Path

import cv2
import numpy as np

from mocap.camera import Camera
from mocap.frames import list_frames

AXIS_LENGTH_M = 0.15


def project(K: np.ndarray, pts: np.ndarray) -> np.ndarray:
    uvw = pts @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def _pt(uv: np.ndarray) -> tuple[int, int]:
    return int(round(uv[0])), int(round(uv[1]))


def draw_frame(img: np.ndarray, K: np.ndarray, aligned: dict, i: int) -> np.ndarray:
    out = img.copy()
    if aligned["gun_valid"][i]:
        T = aligned["T_cam_gun"][i]
        tips = np.vstack([np.zeros(3), np.eye(3) * AXIS_LENGTH_M])
        uv = project(K, tips @ T[:3, :3].T + T[:3, 3])
        from_hand = aligned["gun_source"][i] == 1
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: x red, y green, z blue
        for k in range(3):
            cv2.line(out, _pt(uv[0]), _pt(uv[k + 1]), (0, 255, 255) if from_hand else colours[k], 3)
    for side in ("left", "right"):
        if not aligned[f"{side}_valid"][i]:
            continue
        for uv in project(K, aligned[f"{side}_keypoints"][i]):
            cv2.circle(out, _pt(uv), 3, (255, 255, 255), -1)
        w = float(aligned[f"{side}_grip_weight"][i])
        wrist = project(K, aligned[f"{side}_T_cam_wrist"][i][None, :3, 3])[0]
        cv2.circle(out, _pt(wrist), 10, (0, int(255 * w), int(255 * (1 - w))), 3)  # red free -> green locked
    return out


def write_video(rgb_dir: Path, camera: Camera, aligned: dict, out_path: Path) -> int:
    paths = list_frames(rgb_dir)
    if len(paths) != len(aligned["gun_valid"]):
        raise SystemExit(f"{len(paths)} frames in {rgb_dir} but aligned.npz has {len(aligned['gun_valid'])}")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), camera.fps,
                             (camera.width, camera.height))
    try:
        for i, path in enumerate(paths):
            writer.write(draw_frame(cv2.imread(str(path)), camera.K, aligned, i))
    finally:
        writer.release()
    return len(paths)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_overlay.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mocap/overlay.py tests/test_overlay.py
git commit -m "feat: overlay of tracked gun and hands on the video"
```

---

### Task 6: `align` and `overlay` commands

**Files:**
- Modify: `mocap/cli.py` (imports, two `cmd_*` functions, two subparsers)
- Create: `tests/test_cli_align.py`

**Interfaces:**
- Consumes: `contracts.load_gun/load_hands`, `align.run`, `overlay.write_video`, `Camera.load`, `ContractError`.
- Produces: `mocap align <clip>` reads `clips/<clip>/gun.npz` and `hands.npz`, writes `aligned.npz`. `mocap overlay <clip>` reads `rgb/`, `camera.json`, `aligned.npz`, writes `overlay.mp4`. Both exit with a one-line message on a `ContractError`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_align.py`:
```python
import cv2
import numpy as np
import pytest

from mocap import cli
from mocap.camera import Camera, intrinsics_from_fov
from mocap.contracts import save_gun, save_hands


def _make_clip(root, n=12):
    clip = root / "c"
    (clip / "rgb").mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(clip / "rgb" / f"{i + 1:05d}.png"), np.zeros((108, 192, 3), np.uint8))
    Camera(192, 108, 60.0, intrinsics_from_fov(192, 108, 90)).save(clip / "camera.json")
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, 2, 3] = 1.0
    save_gun(clip / "gun.npz", {"T_cam_gun": pose, "valid": np.ones(n, bool), "score": np.ones(n)})
    hand = {
        "T_cam_wrist": pose.copy(),
        "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
        "keypoints": np.tile([0.0, 0.0, 1.0], (n, 21, 1)),
        "valid": np.ones(n, bool),
        "score": np.ones(n),
    }
    save_hands(clip / "hands.npz", {"left": hand, "right": hand})
    return clip


def test_align_then_overlay_write_their_files(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path)
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    cli.main(["align", "c"])
    assert (clip / "aligned.npz").exists()
    cli.main(["overlay", "c"])
    cap = cv2.VideoCapture(str(clip / "overlay.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12


def test_align_without_inputs_fails_with_a_message(tmp_path, monkeypatch):
    (tmp_path / "c").mkdir()
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    with pytest.raises(SystemExit, match="gun.npz"):
        cli.main(["align", "c"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_align.py -v`
Expected: FAIL (`argparse` error: invalid choice `align`, raised as `SystemExit`).

- [ ] **Step 3: Implement**

In `mocap/cli.py`, replace the imports block lines 3-10 with:
```python
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from mocap import align, frames, overlay, sheets
from mocap.camera import Camera, calibrate, intrinsics_from_fov
from mocap.contracts import ContractError, load_gun, load_hands
```
Add these functions after `cmd_sheets`:
```python
def cmd_align(args: argparse.Namespace) -> None:
    clip = _clip_dir(args.clip)
    try:
        aligned = align.run(load_gun(clip / "gun.npz"), load_hands(clip / "hands.npz"))
    except ContractError as e:
        raise SystemExit(str(e))
    np.savez(clip / "aligned.npz", **aligned)
    gripping = {s: int((aligned[f"{s}_grip_weight"] > 0.5).sum()) for s in ("left", "right")}
    print(f"gun valid on {int(aligned['gun_valid'].sum())} of {len(aligned['gun_valid'])} frames; "
          f"gripping frames {gripping} -> {clip / 'aligned.npz'}")


def cmd_overlay(args: argparse.Namespace) -> None:
    clip = _clip_dir(args.clip)
    for needed in ("camera.json", "aligned.npz"):
        if not (clip / needed).exists():
            raise SystemExit(f"{clip / needed} not found")
    count = overlay.write_video(clip / "rgb", Camera.load(clip / "camera.json"),
                                dict(np.load(clip / "aligned.npz")), clip / "overlay.mp4")
    print(f"{count} frames -> {clip / 'overlay.mp4'}")
```
Add before `args = parser.parse_args(argv)`:
```python
    p = sub.add_parser("align", help="join gun.npz and hands.npz into aligned.npz")
    p.add_argument("clip")
    p.set_defaults(func=cmd_align)

    p = sub.add_parser("overlay", help="draw aligned.npz onto the frames as overlay.mp4")
    p.add_argument("clip")
    p.set_defaults(func=cmd_overlay)

```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass (existing camera/frames tests included).

- [ ] **Step 5: Commit and update README**

Add to the README's "Stages so far" block:
```
uv run mocap align <clip>                             # clips/<clip>/aligned.npz from gun.npz + hands.npz
uv run mocap overlay <clip>                           # clips/<clip>/overlay.mp4
```
```bash
git add mocap/cli.py tests/test_cli_align.py README.md
git commit -m "feat: align and overlay commands"
```

---

### Task 7: Blender setup and the gun mesh

**Files:**
- Create: `blender/list_bones.py`, `blender/export_gun_mesh.py`
- Output (not committed): `rigs/M4.fbx` (copy), `rigs/m4_gun.obj`

**Interfaces:**
- Produces: `rigs/m4_gun.obj`, the gun mesh in metres with its own origin and axes kept, for FoundationPose. A recorded list of the rig's object and bone names (written into the plan notes in Step 4) used by Task 9.

- [ ] **Step 1: Install Blender and point at it**

Run: `winget install BlenderFoundation.Blender`, then in PowerShell: `$env:BLENDER = "C:\Program Files\Blender Foundation\Blender 4.x\blender.exe"` (use the real path; `Get-ChildItem "C:\Program Files\Blender Foundation"` shows it).
Verify: `& $env:BLENDER --version` prints a Blender version.

- [ ] **Step 2: Write the inspection scripts**

`blender/list_bones.py`:
```python
"""blender -b -P blender/list_bones.py -- <file.fbx>: print objects, armatures and bone names."""

import sys

import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path)
for obj in bpy.data.objects:
    dims = tuple(round(d, 3) for d in obj.dimensions)
    print(f"OBJECT {obj.name} type={obj.type} dims={dims} parent={obj.parent.name if obj.parent else None}")
    if obj.type == "ARMATURE":
        for bone in obj.data.bones:
            print(f"BONE {bone.name} parent={bone.parent.name if bone.parent else None}")
```
`blender/export_gun_mesh.py`:
```python
"""blender -b -P blender/export_gun_mesh.py -- <rig.fbx> <mesh object name> <out.obj>

Exports one mesh in metres, with the object's own origin and axes, for FoundationPose.
"""

import sys

import bpy

rig, name, out = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=rig)
obj = bpy.data.objects[name]
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(
    filepath=out, export_selected_objects=True, export_materials=False, apply_modifiers=True,
    global_scale=1.0, forward_axis="Y", up_axis="Z",
)
print("DIMS_M", tuple(round(d, 3) for d in obj.dimensions))
```

- [ ] **Step 3: Inspect the rig**

Run (PowerShell):
```powershell
New-Item -ItemType Directory -Force rigs | Out-Null
Copy-Item ..\lone-asset\content\weapons\m4\source\M4.fbx rigs\M4.fbx
& $env:BLENDER -b -P blender/list_bones.py -- rigs\M4.fbx | Select-String "OBJECT|BONE"
```
Expected: lines for objects and bones including `Main_j`, `RightHand`, `LeftHand`, `RightHandIndex1` (names as in plan.md). Note the exact name of the gun mesh object and whether its `dims` longest side is about 0.8-0.9 (metres) or 80-90 (centimetres).

- [ ] **Step 4: Export the gun mesh and check its size**

Run: `& $env:BLENDER -b -P blender/export_gun_mesh.py -- rigs\M4.fbx <gun mesh object name> rigs\m4_gun.obj`
Expected output ends with `DIMS_M (...)` where the longest value is between 0.7 and 0.95. If it is about 80-90, the FBX is in centimetres: rerun with `global_scale=0.01` in the script. Record the object name, the bone names, and the scale used as a comment block at the top of `blender/retarget.py` in Task 9.

Note: the FBX probably carries its own animation (a binary grep finds `AnimationStack`), and `export_gun_mesh.py` exports the mesh at the import's current pose, not necessarily the rest pose. Task 9 Step 2 therefore first verifies that the imported rig's gun bone pose matrix equals its rest matrix at the frame used (or sets `arm.data.pose_position = 'REST'` in both scripts). `blender/export_gun_mesh.py` itself is not changed by this note; if the check fails, re-export the mesh with the rest pose.

- [ ] **Step 5: Commit**

```bash
git add blender/list_bones.py blender/export_gun_mesh.py
git commit -m "feat: blender scripts to inspect the M4 rig and export the gun mesh"
```

---

### Task 8: GPU stages script (`notebooks/gpu_stages.py`)

This task is the least certain: the external repositories' APIs below are written from their public READMEs and must be confirmed on the first GPU run. The printed key listing in Step 2 exists for that purpose. The CPU-side contract (`save_hands`, `save_gun`) is already tested, so any mismatch is confined to the adapter lines marked `ADAPTER`.

**Files:**
- Create: `notebooks/gpu_stages.py`

**Interfaces:**
- Consumes: `clips/m4_pullout/rgb/`, `camera.json`, `meta.json`, `rigs/m4_gun.obj`, `mocap.contracts.save_hands`, `mocap.contracts.save_gun`.
- Produces: `hands.npz`, `gun.npz`, `masks/gun_mask.png`, `depth/*.npy` (float32 metres) under `clips/m4_pullout/`.
- Upload to Kaggle/Colab: the `mocap/` package, `clips/m4_pullout/{rgb,camera.json,meta.json}`, `rigs/m4_gun.obj`. Download after: `hands.npz`, `gun.npz`.

- [ ] **Step 1: Write the script**

`notebooks/gpu_stages.py` (percent format; each `# %%` is a notebook cell; set `ROOT` to the uploaded folder):
```python
# %% setup
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/kaggle/working/fpv-mocap")  # Colab: Path("/content/fpv-mocap")
CLIP = ROOT / "clips" / "m4_pullout"
sys.path.insert(0, str(ROOT))
from mocap.contracts import save_gun, save_hands
from mocap.frames import list_frames

meta = json.loads((CLIP / "meta.json").read_text())
cam = json.loads((CLIP / "camera.json").read_text())
K = np.array(cam["K"])
frames = list_frames(CLIP / "rgb")
N = len(frames)
init = meta["init_frame"] - 1  # 0-based index

# %% install (once per session)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wilor-mini", "sam2", "trimesh"], check=True)
# Depth Anything V2 and FoundationPose are cloned and built as their READMEs describe:
#   git clone https://github.com/DepthAnything/Depth-Anything-V2 ; cd metric_depth ; download the
#   depth_anything_v2_metric_hypersim_vitl.pth checkpoint
#   git clone https://github.com/NVlabs/FoundationPose ; build_all.sh (compiles CUDA extensions;
#   save the build as a Kaggle Dataset so later sessions skip it)

# %% hands: WiLoR (ADAPTER: confirm output keys on the first run)
import torch
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

pipe = WiLorHandPose3dEstimationPipeline(device="cuda", dtype=torch.float16)


def empty_hand():
    return {
        "T_cam_wrist": np.tile(np.eye(4), (N, 1, 1)),
        "finger_rot": np.tile(np.eye(3), (N, 15, 1, 1)),
        "keypoints": np.zeros((N, 21, 3)),
        "valid": np.zeros(N, bool),
        "score": np.zeros(N),
    }


hands = {"left": empty_hand(), "right": empty_hand()}
for i, path in enumerate(frames):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    for out in pipe.predict(rgb):
        if i == init:
            print("WiLoR keys:", out.keys(), out["wilor_preds"].keys())  # ADAPTER check
        side = "right" if out["is_right"] else "left"
        p = out["wilor_preds"]
        h = hands[side]
        T = np.eye(4)
        T[:3, :3] = np.asarray(p["global_orient"]).reshape(3, 3)
        T[:3, 3] = np.asarray(p["pred_cam_t_full"]).reshape(3)
        h["T_cam_wrist"][i] = T
        h["finger_rot"][i] = np.asarray(p["hand_pose"]).reshape(15, 3, 3)
        h["keypoints"][i] = np.asarray(p["pred_keypoints_3d"]).reshape(21, 3) + T[:3, 3]
        h["valid"][i] = True
        h["score"][i] = 1.0
save_hands(CLIP / "hands.npz", hands)
print({s: int(hands[s]["valid"].sum()) for s in hands}, "valid frames of", N)

# %% depth: Depth Anything V2 metric (run from the cloned metric_depth folder; ADAPTER)
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20)
model.load_state_dict(torch.load("checkpoints/depth_anything_v2_metric_hypersim_vitl.pth", map_location="cpu"))
model = model.to("cuda").eval()
(CLIP / "depth").mkdir(exist_ok=True)
for i, path in enumerate(frames):
    np.save(CLIP / "depth" / f"{i + 1:05d}.npy", model.infer_image(cv2.imread(str(path))).astype(np.float32))

# %% gun mask on the init frame: SAM 2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
init_rgb = cv2.cvtColor(cv2.imread(str(frames[init])), cv2.COLOR_BGR2RGB)
predictor.set_image(init_rgb)
masks, scores, _ = predictor.predict(
    point_coords=np.array([meta["gun_click"]]), point_labels=np.array([1]), multimask_output=False
)
(CLIP / "masks").mkdir(exist_ok=True)
cv2.imwrite(str(CLIP / "masks" / "gun_mask.png"), masks[0].astype(np.uint8) * 255)
overlay = init_rgb.copy()
overlay[masks[0] > 0] = (0.5 * overlay[masks[0] > 0] + [127, 0, 0]).astype(np.uint8)
cv2.imwrite(str(CLIP / "masks" / "gun_mask_check.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
# Look at gun_mask_check.png: the mask must cover the rifle only. If not, change gun_click in meta.json.

# %% gun: FoundationPose, forward then backward from the init frame (ADAPTER)
import trimesh
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor, dr  # FoundationPose repo

mesh = trimesh.load(ROOT / "rigs" / "m4_gun.obj", force="mesh")
est = FoundationPose(
    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
    scorer=ScorePredictor(), refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
)
mask = cv2.imread(str(CLIP / "masks" / "gun_mask.png"), cv2.IMREAD_GRAYSCALE) > 0

T_gun = np.tile(np.eye(4), (N, 1, 1))
gun_valid = np.zeros(N, bool)


def rgb_depth(i):
    return (cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB), np.load(CLIP / "depth" / f"{i + 1:05d}.npy"))


for order in (range(init, N), range(init, -1, -1)):
    rgb, depth = rgb_depth(init)
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    for i in order:
        if i != init:
            rgb, depth = rgb_depth(i)
            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)
        T_gun[i] = pose.reshape(4, 4)
        gun_valid[i] = True
# Frames before the gun enters the view are not tracked reliably: set them invalid after looking at
# overlay.mp4, e.g. gun_valid[:first_visible] = False, then save again.
save_gun(CLIP / "gun.npz", {"T_cam_gun": T_gun, "valid": gun_valid, "score": gun_valid.astype(float)})
```

- [ ] **Step 2: Run on Kaggle/Colab in stages, checking as you go**

1. Upload the files listed under Interfaces. Run the `setup`, `install` and `hands` cells. Expected: a printed `WiLoR keys:` line; compare it with the fields the adapter reads (`is_right`, `wilor_preds`, `global_orient`, `hand_pose`, `pred_keypoints_3d`, `pred_cam_t_full`) and fix any mismatch in the adapter lines only. Then `{'left': N1, 'right': N2} valid frames of 137` with both counts well above 0.
2. Download `hands.npz` and validate it on the laptop: `uv run python -c "from mocap.contracts import load_hands; load_hands('clips/m4_pullout/hands.npz')"` must print nothing and exit 0. (`mocap align` needs `gun.npz` too, so the hands overlay waits for step 3.)
3. Run the `depth`, `gun mask` and `gun` cells. Look at `masks/gun_mask_check.png` before the `gun` cell. Download `gun.npz`.

Adapter notes (check on the first run, fix in the adapter lines only):

- `global_orient` and `hand_pose` may be axis-angle (shapes `(3,)` and `(45,)`) rather than rotation matrices. If so convert with `scipy.spatial.transform.Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix()` before writing `T_cam_wrist` and `finger_rot`.
- Confirm how left-hand outputs are mirrored (WiLoR flips left-hand crops, so left outputs may be in the mirrored frame and need un-mirroring before use).
- Consider keypoint 0 (the wrist joint, `pred_keypoints_3d[0] + pred_cam_t_full`) instead of `pred_cam_t_full` as the wrist translation, since it is the point the IK target should sit on (grip locking does not care which fixed point is used).
- Frames invalidated before the gun or hand enters the view must keep identity poses (`align.run` now tolerates zeros on invalid frames, but keep identity anyway).
- If the first `mocap align` prints `gripping 0 frames` for a side, loosen `--eps-t` / `--eps-r-deg` before anything else, and read the printed median wrist-to-gun distance (expect 0.05-0.3 m).
- Depth comes from three independent scales: the guessed FOV 90, Depth Anything Hypersim indoor metric depth, and the WiLoR focal prior. If the wrist-to-gun distance is off, suspect these first.

- [ ] **Step 3: Check on the laptop**

Run:
```bash
uv run mocap align m4_pullout
uv run mocap overlay m4_pullout
```
Expected: `gun valid on X of 137 frames -> ...`, then per side a line like `right: gripping 41 frames, median wrist-to-gun distance 0.17 m`, and `137 frames -> clips\m4_pullout\overlay.mp4`. If a side prints `gripping 0 frames`, see the first-align bullet in the adapter notes under Step 2. Open `overlay.mp4`: gun axes should sit on the rifle through the clip and the hand dots should sit on the hands. Fix by editing `gun_valid[:first_visible]`/`meta.json` and rerunning the affected cell; if the axes are consistently the wrong size, adjust `--fov` and rerun `camera`, then the GPU cells.

- [ ] **Step 4: Commit**

```bash
git add notebooks/gpu_stages.py clips/m4_pullout/hands.npz clips/m4_pullout/gun.npz clips/m4_pullout/aligned.npz
git commit -m "feat: GPU stage script and tracked hands/gun for m4_pullout"
```

---

### Task 9: Retarget to the FPS arms and the M4 (`blender/retarget.py`)

Developed and verified against `fps_arms.glb` and `m4a1_single.glb` (repo root, uncommitted Sketchfab assets; facts
in `docs/rig-notes.md`), which replace the Lone Asset `M4.fbx` rig. The script is the source of truth; this section
only describes it.

**Files:**
- Create: `blender/retarget.py`; `mocap/transforms.py` gains `finger_local_rotation` (pure numpy);
  `tests/test_retarget_script.py` (compile check + that helper; Blender is not launched).
- Output, in the clip dir: `retarget_setup.blend` (before baking: wrist empties, IK and Copy Rotation live, for
  hand fixes), `retarget.blend` (baked, constraints cleared), `<clip>.glb` (arms mesh + armature + rifle, one
  animation named after the clip; no camera, no empties).

**What it does:** imports both glbs; scales the arms by `ARM_SCALE` = 0.01 (cm to m, times `ARMS_EXTRA_SCALE`) and
applies it; parents the rifle node `base` under an empty `weapon_root` at the origin (gun rest = identity, so the
gun pose is `cam_world @ FLIP @ T_cam_gun`); adds a camera 0.15 m ahead of and 0.22 m above the shoulder midpoint,
looking -Y, with the footage's field of view; sets the upper-arm and forearm tails to their child heads (IK drives
the tail); finds bones by prefix (`f"{S}_{finger}{k}_"`); keys `weapon_root` per frame (invalid frames hold), one
empty per wrist (`ik_target_{side}`), and MANO finger rotations on bones 1..3 as `C^-1 R C` (`FINGER_BASIS`,
`MIRROR_LEFT`, both at the top of the file for tuning); a 2-bone IK on each forearm and a Copy Rotation on each hand
follow the empty; then bakes with visual keying and exports.

- [x] **Step 1: Run it**

Run (PowerShell): `& $env:BLENDER -b -P blender/retarget.py -- clips\m4_pullout fps_arms.glb m4a1_single.glb`
Expected: ends with `RETARGET exported ...m4_pullout.glb`. A `retarget: expected one bone starting with ...` exit
means a bone prefix does not match the rig.

Verified on a synthetic clip (30 frames, gun 0.45 m ahead): `weapon_root` matches `cam_to_world` to 2e-5 (float32)
in both the `.blend` and the re-imported `.glb`; the right wrist lands on its empty to 0.01 cm; index bone 1
carries the keyed 40 deg. Known gaps for real data: `FINGER_BASIS` = identity un-curls the fingers on this rig,
and the support hand may be out of reach (see `docs/rig-notes.md`).

- [ ] **Step 2: Judge by eye and tune**

Open `clips/m4_pullout/retarget_setup.blend` in Blender and play it. Check: (a) the rifle comes up from below as in
the source clip; (b) the hands follow the rifle and the forearms bend sensibly; (c) fingers curl rather than twist.
If the rifle is rotated relative to the clip, compare with `overlay.mp4` first (the pose is upstream); if the
hands drift off the rifle, check `overlay.mp4`; if fingers twist or un-curl, tune `FINGER_BASIS` (and
`MIRROR_LEFT` for the left hand); if the support hand stops short, look at the tracked distances before raising
`ARMS_EXTRA_SCALE`. Re-run the script after each change.

- [ ] **Step 3: Check the `.glb` in Godot**

Copy `clips/m4_pullout/m4_pullout.glb` into a scratch Godot project (not Lone Asset yet), open it, and play the
`m4_pullout` animation in an `AnimationPlayer` on its imported scene. Expected: the same motion as the `.blend`.

- [ ] **Step 4: Commit**

```bash
git add clips/m4_pullout/retarget.blend clips/m4_pullout/m4_pullout.glb
git commit -m "feat: retarget m4_pullout to the FPS arms and export glb"
```

M5 (copying the `.glb` into Lone Asset and registering the animation name in the rifle weapon file) is done by the developer after this check, per `docs/plan.md` step 7, and is outside this plan.

---

## Self-Review

- **Spec coverage:** decisions (markerless path, M4 mesh, fov 90, locked camera, hands-first order, SOURCE.md, reference-only) are in Global Constraints and Tasks 1, 7, 8. Milestones M1-M4 map to Tasks 8 (hands, gun), 4+6 (align), 9 (retarget); M5 is the developer's step; M6 is deferred as the spec says. Stage table is covered by Tasks 1, 6, 8, 9. Init-frame choice and forward/backward tracking are in Tasks 1 and 8. Right-hand fallback for lost gun frames is `align.run` (Task 4), with `gun_source` as the score record (the spec's wording about `score` is satisfied by `gun_source`, which keeps `score` for detector confidence). Errors: `ContractError` plus clear `SystemExit` messages (Tasks 3, 6). Testing: Tasks 2-6 pytest; GPU and Blender by overlay and playback.
- **Spec deviation recorded:** the camera-to-Blender conversion formula (Task 1 Step 5 amends the spec).
- **Placeholder scan:** the adapter lines in Task 8 are known unknowns, stated as such with a verification step, not hidden TODOs. `GUN_REST` (the gun bone's rest matrix, applied to the gun bone only) is read from the rig, and the per-bone finger change of basis `R_bone = C^-1 @ R_mano @ C` is tuned by eye in Task 9 Step 4 with concrete instructions.
- **Type consistency:** `align.run` output keys match what `overlay.draw_frame` reads (`gun_valid`, `gun_source`, `T_cam_gun`, `{side}_valid`, `{side}_keypoints`, `{side}_T_cam_wrist`, `{side}_grip_weight`) and what `retarget.py` reads. `contracts` field names match the notebook's `empty_hand()`.
