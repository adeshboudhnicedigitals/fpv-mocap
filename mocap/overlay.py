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
