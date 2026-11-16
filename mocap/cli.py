"""`mocap <stage> ...`: each stage reads and writes files under clips/<name>/."""

import argparse
import math
from pathlib import Path

import cv2

from mocap import frames, sheets
from mocap.camera import Camera, calibrate, intrinsics_from_fov

CLIPS = Path("clips")


def _clip_dir(name: str) -> Path:
    return CLIPS / name


def cmd_frames(args: argparse.Namespace) -> None:
    rgb = _clip_dir(args.clip) / "rgb"
    count = frames.extract(Path(args.video), rgb, args.fps)
    print(f"{count} frames at {args.fps:g} fps -> {rgb}")


def cmd_camera(args: argparse.Namespace) -> None:
    clip = _clip_dir(args.clip)
    first = frames.list_frames(clip / "rgb")
    if not first:
        raise SystemExit(f"no frames in {clip / 'rgb'}; run `mocap frames` first")
    height, width = cv2.imread(str(first[0])).shape[:2]
    if args.fov is not None:
        camera = Camera(width, height, args.fps, intrinsics_from_fov(width, height, args.fov))
    else:
        paths = sorted(Path(args.checkerboard).glob("*.*"))
        images = [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None]
        if images and images[0].shape[:2] != (height, width):
            raise SystemExit("checkerboard images must have the clip's resolution")
        pattern = tuple(int(v) for v in args.pattern.split("x"))
        K, dist, rms, used = calibrate(images, pattern, args.square_mm / 1000.0)
        print(f"checkerboard found in {used} of {len(images)} images, RMS {rms:.3f} px")
        camera = Camera(width, height, args.fps, K, dist, "calibrated", rms)
    camera.save(clip / "camera.json")
    hfov = 2 * math.degrees(math.atan(width / (2 * camera.K[0, 0])))
    print(f"{width}x{height}, fx {camera.K[0, 0]:.1f}, cx {camera.K[0, 2]:.1f}, cy {camera.K[1, 2]:.1f}, "
          f"horizontal FOV {hfov:.1f} deg -> {clip / 'camera.json'}")


def cmd_sheets(args: argparse.Namespace) -> None:
    for path in sheets.write_sheets(Path(args.out)):
        print(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mocap")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("frames", help="extract frames from a video")
    p.add_argument("clip")
    p.add_argument("video")
    p.add_argument("--fps", type=float, default=60.0)
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("camera", help="write camera.json from a checkerboard or a guessed FOV")
    p.add_argument("clip")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fov", type=float, help="guessed horizontal field of view in degrees")
    source.add_argument("--checkerboard", help="folder of checkerboard photos or frames from the same camera")
    p.add_argument("--pattern", default="9x6", help="inner corners, cols x rows")
    p.add_argument("--square-mm", type=float, default=25.0, help="measured size of one printed square")
    p.add_argument("--fps", type=float, default=60.0)
    p.set_defaults(func=cmd_camera)

    p = sub.add_parser("sheets", help="write printable ArUco and checkerboard sheets")
    p.add_argument("--out", default="sheets")
    p.set_defaults(func=cmd_sheets)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
