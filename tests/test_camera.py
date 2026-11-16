import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from mocap.camera import Camera, calibrate, intrinsics_from_fov

PATTERN = (9, 6)
SQUARE_M = 0.03


def test_fov_90_puts_focal_length_at_half_width():
    K = intrinsics_from_fov(1920, 1080, 90.0)
    assert K[0, 0] == pytest.approx(960.0)
    assert K[1, 1] == K[0, 0]
    assert (K[0, 2], K[1, 2]) == (960.0, 540.0)


def test_fov_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        intrinsics_from_fov(1920, 1080, 180.0)


def test_camera_json_round_trip(tmp_path: Path):
    camera = Camera(1280, 720, 60.0, intrinsics_from_fov(1280, 720, 100.0), np.arange(5.0), "calibrated", 0.21)
    camera.save(tmp_path / "camera.json")
    loaded = Camera.load(tmp_path / "camera.json")
    assert np.allclose(loaded.K, camera.K)
    assert np.allclose(loaded.dist, camera.dist)
    assert (loaded.width, loaded.height, loaded.fps, loaded.source, loaded.rms) == (1280, 720, 60.0, "calibrated", 0.21)


def _render_board_views(K: np.ndarray, size: tuple[int, int]) -> list[np.ndarray]:
    """Pictures of a flat checkerboard seen by a distortion-free camera K from several angles."""
    px, border = 40, 40
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    board = np.full((rows * px + 2 * border, cols * px + 2 * border), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[border + r * px : border + (r + 1) * px, border + c * px : border + (c + 1) * px] = 0
    # Board plane metres -> board image pixels. The first inner corner is plane (0, 0).
    S = np.array([[px / SQUARE_M, 0, border + px], [0, px / SQUARE_M, border + px], [0, 0, 1]])
    centre = np.array([(PATTERN[0] - 1) * SQUARE_M / 2, (PATTERN[1] - 1) * SQUARE_M / 2, 0.0])
    views = []
    for yaw, pitch, roll, x, y, z in [
        (0, 0, 0, 0.0, 0.0, 0.6), (25, 0, 5, 0.05, 0.0, 0.65), (-25, 5, -5, -0.05, 0.02, 0.65),
        (0, 25, 10, 0.0, -0.05, 0.7), (0, -25, -10, 0.0, 0.05, 0.7), (20, 20, 0, 0.1, 0.05, 0.75),
        (-20, -20, 15, -0.1, -0.05, 0.75), (10, -15, 30, 0.08, -0.06, 0.55),
    ]:
        R, _ = cv2.Rodrigues(np.radians([pitch, yaw, roll]).astype(np.float64))
        t = np.array([x, y, z]) - R @ centre
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.linalg.inv(S)
        views.append(cv2.warpPerspective(board, H, size, flags=cv2.INTER_AREA, borderValue=180))
    return views


def test_calibration_recovers_a_known_camera():
    size = (1280, 720)
    K_true = intrinsics_from_fov(*size, 90.0)
    K, dist, rms, used = calibrate(_render_board_views(K_true, size), PATTERN, SQUARE_M)
    assert used == 8
    assert rms < 0.5
    assert K[0, 0] == pytest.approx(K_true[0, 0], rel=0.01)
    assert K[1, 1] == pytest.approx(K_true[1, 1], rel=0.01)
    assert math.hypot(K[0, 2] - K_true[0, 2], K[1, 2] - K_true[1, 2]) < 8.0


def test_calibration_needs_three_boards():
    blank = [np.full((720, 1280), 128, np.uint8)] * 4
    with pytest.raises(ValueError, match="at least 3"):
        calibrate(blank, PATTERN, SQUARE_M)
