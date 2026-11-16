"""blender/retarget.py runs only inside Blender; here: it compiles, and its pure maths is right."""

import ast
from pathlib import Path

import numpy as np

from mocap.transforms import MIRROR_X, finger_local_rotation

SCRIPT = Path(__file__).resolve().parent.parent / "blender" / "retarget.py"


def _rot(axis: int, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def test_script_compiles_and_exposes_the_tunables():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(compile(source, str(SCRIPT), "exec", ast.PyCF_ONLY_AST))
    names = {t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets
             if isinstance(t, ast.Name)}
    assert {"ARM_SCALE", "ARMS_EXTRA_SCALE", "CAM_AHEAD_OF_SHOULDERS", "CAM_ABOVE_SHOULDERS",
            "FINGER_BASIS", "MIRROR_LEFT"} <= names


def test_identity_basis_keeps_the_rotation():
    R = _rot(0, 40)
    assert np.allclose(finger_local_rotation(R, np.eye(3)), R)


def test_basis_change_moves_the_rotation_axis():
    # C^-1 R C is the same rotation seen in the bone's axes: its axis becomes C^-1 @ (R's axis), angle kept.
    C = _rot(2, 90)
    out = finger_local_rotation(_rot(0, 40), C)
    axis_bone = np.linalg.inv(C) @ np.array([1.0, 0, 0])
    assert np.allclose(out @ axis_bone, axis_bone)  # the rotation axis is preserved in the bone frame
    assert np.isclose(np.degrees(np.arccos((np.trace(out) - 1) / 2)), 40)


def test_mirror_reflects_then_stays_a_rotation_and_works_on_batches():
    R = np.stack([_rot(1, 30), _rot(2, -20)])
    out = finger_local_rotation(R, np.eye(3), mirror=True)
    assert out.shape == (2, 3, 3)
    assert np.allclose(out, MIRROR_X @ R @ MIRROR_X)
    assert np.allclose(np.linalg.det(out), 1)
    assert np.allclose(out[0], _rot(1, -30))  # a rotation about y flips direction under x reflection
