"""Join gun and hands: grip detection, wrist locking, gun fill-in and finger smoothing.

Pure maths on arrays; no files. All poses are camera space, metres.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from mocap.contracts import SIDES, ContractError
from mocap.transforms import relative


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    cos = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def grip_weight(
    rel: np.ndarray,
    valid: np.ndarray,
    window: int = 15,
    eps_t: float = 0.01,
    eps_r: float = np.radians(8),
    blend: int = 9,
    min_valid: int = 5,
) -> np.ndarray:
    """1 where the wrist's pose in the gun's frame barely changes over a window, else 0, blended."""
    n = len(rel)
    hard = np.zeros(n)
    half = window // 2
    for i in range(n):
        if not valid[i]:
            continue
        idx = [j for j in range(max(0, i - half), min(n, i + half + 1)) if valid[j]]
        if len(idx) < min_valid:
            continue
        dt = max(np.linalg.norm(rel[j, :3, 3] - rel[i, :3, 3]) for j in idx)
        dr = max(_angle(rel[i, :3, :3], rel[j, :3, :3]) for j in idx)
        hard[i] = float(dt < eps_t and dr < eps_r)
    weight = uniform_filter1d(hard, blend, mode="nearest")
    weight[~valid] = 0.0
    return weight


def grip_offset(rel: np.ndarray, weight: np.ndarray) -> np.ndarray | None:
    """Mean wrist-in-gun pose over the frames that are firmly gripping, or None if there are none."""
    idx = np.flatnonzero(weight >= 0.99)
    if len(idx) == 0:
        return None
    T = np.eye(4)
    T[:3, 3] = rel[idx, :3, 3].mean(axis=0)
    T[:3, :3] = Rotation.from_matrix(rel[idx, :3, :3]).mean().as_matrix()
    return T


def blend_pose(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-frame blend from A (w=0) to B (w=1): translation lerp, rotation slerp."""
    out = np.tile(np.eye(4), (len(A), 1, 1))
    out[:, :3, 3] = (1 - w)[:, None] * A[:, :3, 3] + w[:, None] * B[:, :3, 3]
    for i in range(len(A)):
        pair = Rotation.from_matrix(np.stack([A[i, :3, :3], B[i, :3, :3]]))
        out[i, :3, :3] = Slerp([0.0, 1.0], pair)(w[i]).as_matrix()
    return out


def smooth_fingers(finger_rot: np.ndarray, valid: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay on finger rotation vectors; invalid frames are interpolated first."""
    n = len(finger_rot)
    vec = Rotation.from_matrix(finger_rot.reshape(-1, 3, 3)).as_rotvec().reshape(n, 15 * 3)
    if valid.any() and not valid.all():
        frames = np.arange(n)
        for c in range(vec.shape[1]):
            vec[:, c] = np.interp(frames, frames[valid], vec[valid, c])
    if n >= window:
        vec = savgol_filter(vec, window, order, axis=0)
    return Rotation.from_rotvec(vec.reshape(-1, 3)).as_matrix().reshape(n, 15, 3, 3)


def _identity_where_invalid(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Copy of arr (N, ..., r, r) with the trailing square matrix set to identity where valid is False."""
    out = arr.copy()
    out[~valid] = np.eye(arr.shape[-1])
    return out


def run(
    gun: dict[str, np.ndarray],
    hands: dict[str, dict[str, np.ndarray]],
    eps_t: float = 0.01,
    eps_r: float = np.radians(8),
) -> dict[str, np.ndarray]:
    """Produce the contents of aligned.npz from gun and hands data (see Interfaces in the plan).

    Poses on frames where their `valid` flag is False are ignored (they may be zeros or garbage): the
    inputs are not mutated, and those frames hold identity in the outputs (T_cam_gun unless filled from
    the right hand, and each side's T_cam_wrist). finger_rot on invalid frames is interpolated from
    neighbouring valid frames. eps_t (metres) and eps_r (radians) are the grip detection thresholds.
    """
    counts = (len(gun["T_cam_gun"]), len(hands["left"]["T_cam_wrist"]), len(hands["right"]["T_cam_wrist"]))
    if len(set(counts)) != 1:
        raise ContractError(f"gun has {counts[0]} frames, left hand {counts[1]}, right hand {counts[2]}")
    gun = {**gun, "T_cam_gun": _identity_where_invalid(gun["T_cam_gun"], gun["valid"])}
    hands = {
        side: {
            **h,
            "T_cam_wrist": _identity_where_invalid(h["T_cam_wrist"], h["valid"]),
            "finger_rot": _identity_where_invalid(h["finger_rot"], h["valid"]),
        }
        for side, h in hands.items()
    }
    T_gun = gun["T_cam_gun"].copy()
    gun_valid = gun["valid"].copy()
    source = np.where(gun_valid, 0, -1).astype(np.int8)

    rel, weight, offset = {}, {}, {}
    for side in SIDES:
        h = hands[side]
        both = gun["valid"] & h["valid"]
        rel[side] = relative(gun["T_cam_gun"], h["T_cam_wrist"])
        weight[side] = grip_weight(rel[side], both, eps_t=eps_t, eps_r=eps_r)
        offset[side] = grip_offset(rel[side], weight[side])

    right = hands["right"]
    if offset["right"] is not None:
        lost = ~gun["valid"] & right["valid"]
        T_gun[lost] = right["T_cam_wrist"][lost] @ np.linalg.inv(offset["right"])
        gun_valid |= lost
        source[lost] = 1

    out: dict[str, np.ndarray] = {"T_cam_gun": T_gun, "gun_valid": gun_valid, "gun_source": source}
    for side in SIDES:
        h = hands[side]
        wrist = h["T_cam_wrist"]
        if offset[side] is not None:
            wrist = blend_pose(wrist, T_gun @ offset[side], weight[side])
        out[f"{side}_T_cam_wrist"] = wrist
        out[f"{side}_finger_rot"] = smooth_fingers(h["finger_rot"], h["valid"])
        out[f"{side}_keypoints"] = h["keypoints"]
        out[f"{side}_valid"] = h["valid"]
        out[f"{side}_grip_weight"] = weight[side]
    return out
