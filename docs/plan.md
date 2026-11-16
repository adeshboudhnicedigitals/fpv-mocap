# fpv-mocap plan

Turn a first-person video of someone handling a rifle into a first-person arms animation on the
M4 pack rig used by Lone Asset (`../lone-asset/content/weapons/m4/source/M4.fbx`), exported as `.glb`.

First target: a rifle reload.

## Approach

Track the gun and the hands separately, join them where the hand grips the gun, and let IK solve the
elbows and shoulders.

1. Frames and camera: extract frames; get the camera intrinsics `K` (checkerboard calibration for
   our own recordings, a guessed field of view for other footage).
2. Gun pose: a 6-DoF pose per frame, camera to gun.
   - Markers on a replica: OpenCV marker detection + `solvePnP` on the laptop CPU.
   - Footage without markers (films, real soldier POV): FoundationPose on a GPU with a mesh close
     to the gun in the clip, a SAM 2 mask on the first frame, and Depth Anything V2 depth.
3. Hands: WiLoR per frame on a GPU gives each hand's wrist pose, MANO finger rotations and
   21 keypoints.
4. Align: store each wrist in the gun's space; while it is gripping (small change over a window) lock
   it to the gun, otherwise let it move freely in camera space, and blend between the two. Smooth
   the fingers (Savitzky-Golay); leave the gun unsmoothed.
5. Overlay check: draw the tracked gun and hands back onto the video. Every stage is judged here
   by eye before moving on.
6. Retarget in Blender (headless): put the camera on the rig's camera bone, the gun on `Main_j`,
   the wrists on IK targets for `RightHand` / `LeftHand`, the MANO fingers on the
   `*HandIndex1..3`, `*HandMiddle1..3`, ... bones. Bake and export one action as `.glb`. The
   `.blend` is kept so the clip can be cleaned by hand.
7. Into the game: copy the `.glb` into Lone Asset and add the animation name to the rifle weapon
   file. `FpsArms` copies the rig bones as it does for every other clip.

## Where things run

| Step | Where | Why |
|---|---|---|
| Frames, calibration, markers + PnP, align, overlay | Laptop (CPU) | No NVIDIA GPU here (Intel UHD) |
| SAM 2, Depth Anything, FoundationPose, WiLoR | Kaggle (2x T4) or Colab (T4) | Needs CUDA; 16 GB per card is enough |
| Retarget, bake, export | Laptop, Blender | Blender is not on PATH yet; install it or point `BLENDER` at it |

Kaggle's two T4s do not pool memory. Use them in parallel: WiLoR on half the frames each, or WiLoR
on one card and FoundationPose (which must run frames in order) on the other. One notebook works on
both Kaggle and Colab with only the paths changed. Save compiled CUDA extensions as a Kaggle Dataset
so each session does not rebuild them.

## Layout

```
fpv-mocap/
  pyproject.toml           uv project, Python 3.11 (the GPU libraries lag behind 3.14)
  mocap/
    frames.py              video -> clips/<name>/rgb/%05d.png at a fixed fps
    camera.py              checkerboard calibration or FOV guess -> camera.json
    gun_markers.py         markers + solvePnP -> gun.npz
    align.py               pure maths: frame conversions, grip detection, blending, smoothing
    overlay.py             draws gun and hands onto the frames -> overlay.mp4
    cli.py                 `mocap <stage> <clip>`
  notebooks/
    gpu_stages.ipynb       SAM 2, Depth Anything, FoundationPose, WiLoR -> gun.npz, hands.npz
  blender/
    retarget.py            run with `blender -b -P blender/retarget.py -- <clip>`
  rigs/                    a copy of M4.fbx and the gun mesh (not committed if the licence asks)
  clips/<name>/            per clip: source video, rgb/, camera.json, gun.npz, hands.npz,
                           aligned.npz, overlay.mp4, retarget.blend, <name>.glb, SOURCE.md
  tests/                   unit tests for align.py and the coordinate conversions
```

Each stage reads the previous stage's files and writes its own, so any stage can be rerun alone
and the GPU notebook only has to exchange `rgb/`, `camera.json` and the `.npz` files.

## Data contracts

All tracking output stays in OpenCV camera space (+x right, +y down, +z forward), in metres, until
the Blender step.

- `camera.json`: `width`, `height`, `fps`, `K` (3x3), `dist` (distortion, zeros when guessed),
  `source` (`calibrated` or `fov_guess`).
- `gun.npz`: `T_cam_gun` (N, 4, 4), `valid` (N,) bool, `score` (N,).
- `hands.npz`: for `left` and `right`: `T_cam_wrist` (N, 4, 4), `finger_rot` (N, 15, 3, 3) in MANO
  joint order, `keypoints` (N, 21, 3), `valid` (N,), `score` (N,).
- `aligned.npz`: the same hands after grip locking and smoothing, plus `grip_weight` (N,) per hand.

Conversion to Blender happens once, in `retarget.py`. A pose changes basis with
`M @ T @ M^-1` (not `M @ T`, which only converts the position). The
`pose_bone.matrix` is armature space, so camera-space poses also go through the camera bone's
matrix. Quaternion keys need `rotation_mode = 'QUATERNION'`, and `view_layer.update()` must run
between bones so constraints see the new values.

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

## Source footage

- Self-recorded: phone on the head (or chest), replica or airsoft rifle with a few printed ArUco
  markers, checkerboard shots for calibration. Clean to use, and the best tracking.
- Films and real soldier POV footage: possible through M6, but the camera is unknown, there are no
  markers and the gun mesh must be close to the gun in the clip. The footage belongs to someone
  else: each clip's `SOURCE.md` records where it came from, and animation made from it is used as
  reference until we have checked whether it may ship in the game.

## Licences to keep in mind

FoundationPose (NVIDIA source code licence), MANO (used by WiLoR) and the large Depth Anything V2
models are non-commercial. SAM 2 is Apache 2.0. That matches the game's current non-commercial
assets, but anything made with these tools must be revisited before a commercial release.

## Risks

- Hands are lost during the magazine swap: keep the `.blend` for hand cleanup; fill short gaps by
  interpolating in gun space.
- Guessed FOV gives wrong depth and scale for footage without markers: the gun mesh's known size
  sets the scale in FoundationPose.
- MANO's finger rest pose differs from the rig's: compute a fixed offset per finger bone from a flat
  hand pose once.
- The rig's proportions differ from the performer's: IK keeps the wrists on target; shoulders sit at
  the rig's own positions.
- A single global grip offset per hand fails if the support hand regrips during the draw (mitigation when it
  shows up: one offset per connected run of grip weight >= 0.99).
- Gun fill-in from the right hand does not check that the hand is gripping (mitigation: only fill within or
  next to grip runs).
