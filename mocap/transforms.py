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


# Reflection across the x = 0 plane: un-mirrors a rotation estimated on a horizontally flipped image.
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def finger_local_rotation(R: np.ndarray, C: np.ndarray, mirror: bool = False) -> np.ndarray:
    """MANO joint rotation R as a rig bone's local rotation: C^-1 @ R @ C, after un-mirroring if asked.

    C is the bone's change of basis (MANO joint axes expressed in the bone's rest axes). Works on batches.
    """
    if mirror:
        R = MIRROR_X @ R @ MIRROR_X
    return np.linalg.inv(C) @ R @ C
