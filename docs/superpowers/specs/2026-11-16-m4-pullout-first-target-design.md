# M4 pull-out as the first target

Make `m4_pullout_single_shot.mp4` (1920x1080, 60 fps, 137 frames, rendered game footage, locked
camera, no markers) the first end-to-end clip. The pipeline and data contracts are those in
`docs/plan.md`; this spec only records what changes for this clip.

## Decisions

- The clip uses the markerless path (plan.md M6) first. The marker/PnP path (`gun_markers.py`) is deferred.
- Gun mesh for tracking: `../lone-asset/content/weapons/m4/source/M4.fbx`, rescaled to metres on load.
  The clip's rifle looks different from a stock M4, so gun tracking is approximate.
- Camera: `mocap camera m4_pullout --fov 90` (guessed). Checked against the overlay; adjust if the gun looks the wrong size.
- The camera is locked, so the camera bone is static in Blender.
- Order: hands first, then gun, so the first overlay shows hands early.
- The footage belongs to someone else: `clips/m4_pullout/SOURCE.md` records its origin; the animation
  is reference-only until its licence is checked.

## Milestones (replace plan.md's M1-M6 order)

- M0 Setup: done (`frames`, `camera`).
- M1 Hands: WiLoR on Kaggle/Colab -> `hands.npz`. Check: overlay shows both hands' keypoints; note lost frames.
- M2 Gun: SAM 2 mask + Depth Anything depth + FoundationPose with the M4 mesh -> `gun.npz`. Check: overlay outline follows the gun.
- M3 Align: `align.py` + `overlay.py` -> `aligned.npz`, `overlay.mp4`. Check: gripping hands stay on the gun.
- M4 Retarget: `blender/retarget.py` on the M4 rig -> `.blend`, `.glb`. Check: `.blend` plays the pull-out; `.glb` imports into Godot.
- M5 In game: plays on the rifle in Lone Asset.
- M6 Markers/PnP path and moving-camera footage.

## Stages

| Stage | Runs on | Reads | Writes |
|---|---|---|---|
| frames | laptop (done) | the mp4 | `clips/m4_pullout/rgb/` |
| camera | laptop (done) | frame size | `camera.json` |
| WiLoR | Kaggle/Colab | `rgb/`, `camera.json` | `hands.npz` |
| SAM 2 + Depth Anything | Kaggle/Colab | `rgb/`, a click on the gun at the init frame | `gun_mask.png`, `depth/` |
| FoundationPose | Kaggle/Colab | `rgb/`, `depth/`, mask, M4 mesh, `camera.json` | `gun.npz` |
| align | laptop | `hands.npz`, `gun.npz` | `aligned.npz` |
| overlay | laptop | `rgb/`, `aligned.npz` | `overlay.mp4` |
| retarget | laptop, Blender | `aligned.npz`, `M4.fbx` | `.blend`, `.glb` |

Data contracts (OpenCV camera space, metres): see plan.md "Data contracts".

## Gun initialisation

The gun enters from below the frame: a sliver is visible around frame 23 and it is mostly in view by
about frame 70. FoundationPose cannot start on frame 0.

- The init frame is chosen by eye from a contact sheet (the first frame where the gun is mostly in view) and stored in clip metadata.
- Tracking runs forward from it and backward to the first frame where the gun appears.
- Frames before the gun appears are `valid = False`.
- Where tracking loses lock, the gun pose comes from the gripping hand's wrist pose plus the fixed
  hand-in-gun offset measured on well-tracked frames. `score` records which source each frame used.

## Errors

Each stage checks its inputs and fails with a clear message when one is missing (no `camera.json`,
unreadable mesh, init frame without a mask). Lost-tracking frames are `valid = False`, never an error.

## Testing

- `pytest` on the pure maths: camera-to-Blender conversion (`cam_world @ FLIP @ T`, in `transforms.py`), grip
  detection, blending, Savitzky-Golay smoothing, using synthetic poses with known answers.
- GPU stages and Blender are checked by the overlay and by playing the `.blend`, not by pytest.

## Done when

`overlay.mp4` shows the gun and both hands stuck on the video, the `.blend` plays the pull-out on the
M4 rig, and the `.glb` imports into Godot.

## Out of scope

Markers/PnP, moving-camera footage, tuning the M4 mesh to the clip's rifle.
