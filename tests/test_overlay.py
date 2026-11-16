import cv2
import numpy as np
import pytest

from mocap.camera import Camera, intrinsics_from_fov
from mocap.overlay import draw_frame, project, write_video

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


W, H = 192, 108


def _camera():
    return Camera(W, H, 60.0, intrinsics_from_fov(W, H, 90))


def _write_frames(rgb_dir, sizes):
    rgb_dir.mkdir()
    for k, (w, h) in enumerate(sizes):
        cv2.imwrite(str(rgb_dir / f"{k:04d}.png"), np.zeros((h, w, 3), np.uint8))


def test_write_video_writes_every_frame(tmp_path):
    rgb_dir = tmp_path / "rgb"
    _write_frames(rgb_dir, [(W, H)] * 3)
    out_path = tmp_path / "overlay.mp4"
    assert write_video(rgb_dir, _camera(), _aligned(n=3), out_path) == 3
    cap = cv2.VideoCapture(str(out_path))
    try:
        frames = 0
        while cap.read()[0]:
            frames += 1
    finally:
        cap.release()
    assert frames == 3


def test_write_video_rejects_frame_of_wrong_size(tmp_path):
    rgb_dir = tmp_path / "rgb"
    _write_frames(rgb_dir, [(W, H), (100, 50), (W, H)])
    with pytest.raises(SystemExit) as excinfo:
        write_video(rgb_dir, _camera(), _aligned(n=3), tmp_path / "overlay.mp4")
    msg = str(excinfo.value)
    assert "0001.png" in msg
    assert "size" in msg


def test_draw_frame_keypoint_at_zero_depth_does_not_raise():
    img = np.zeros((H, W, 3), np.uint8)
    aligned = _aligned()
    aligned["left_keypoints"][0, 0] = [0.0, 0.0, 0.0]
    draw_frame(img, _camera().K, aligned, 0)
