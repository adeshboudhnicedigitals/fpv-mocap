# fpv-mocap

A helper tool for Lone Asset: turns first-person video of rifle handling into first-person arms
animations on the game's M4 pack rig, exported as `.glb`.

Plan and status: [`docs/plan.md`](docs/plan.md).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and ffmpeg on PATH. `uv sync` installs Python 3.11 and the
dependencies; `uv run pytest` runs the tests.

## Stages so far

```
uv run mocap sheets                                   # sheets/aruco_markers.png, sheets/checkerboard.png
uv run mocap frames <clip> <video> --fps 60           # clips/<clip>/rgb/00001.png ...
uv run mocap camera <clip> --checkerboard <folder>    # clips/<clip>/camera.json, calibrated
uv run mocap camera <clip> --fov 90                   # or a guessed horizontal FOV
```

## Recording a clip

1. Print both sheets at 100 % scale (no "fit to page"). Measure one marker and one square with a
   ruler; pass the square size as `--square-mm`.
2. Camera: phone strapped to the forehead or a head mount, landscape, 60 fps, fixed focus and
   exposure if the camera app allows it, stabilisation and wide-angle "lens correction" off.
   Use the same settings for the clip and the checkerboard.
3. Calibration: 15 to 20 photos (or a slow video) of the checkerboard from different angles and
   distances, filling different parts of the frame. Put the frames in one folder.
4. Markers: glue 4 to 6 markers flat on the replica where the hands do not cover them (stock side,
   receiver side, top of the handguard). Note which id went where.
5. The clip: start and end in the idle hold for a second, then the reload at normal speed. Good,
   even light; avoid strong backlight.
6. Write `clips/<clip>/SOURCE.md`: who filmed it, when, which replica and where each marker sits.
