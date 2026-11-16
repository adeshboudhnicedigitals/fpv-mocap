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
