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
