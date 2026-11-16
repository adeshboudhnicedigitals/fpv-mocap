"""Load, save and validate the .npz files the stages exchange (see docs/plan.md, Data contracts)."""

from pathlib import Path

import numpy as np

SIDES = ("left", "right")
GUN_SHAPES = {"T_cam_gun": (4, 4), "valid": (), "score": ()}
HAND_SHAPES = {
    "T_cam_wrist": (4, 4),
    "finger_rot": (15, 3, 3),
    "keypoints": (21, 3),
    "valid": (),
    "score": (),
}


class ContractError(ValueError):
    """A stage file is missing, incomplete or has the wrong shape."""


def _load(path: Path, prefix: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    if not Path(path).exists():
        raise ContractError(f"{path} not found")
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    n: int | None = None
    for field, tail in shapes.items():
        key = f"{prefix}{field}"
        if key not in data:
            raise ContractError(f"{path}: missing {key}")
        arr = data[key]
        n = len(arr) if n is None else n
        if arr.shape != (n, *tail):
            raise ContractError(f"{path}: {key} has shape {arr.shape}, expected {(n, *tail)}")
        out[field] = arr.astype(bool) if field == "valid" else arr
    return out


def save_gun(path: Path, gun: dict[str, np.ndarray]) -> None:
    np.savez(path, **gun)


def load_gun(path: Path) -> dict[str, np.ndarray]:
    return _load(path, "", GUN_SHAPES)


def save_hands(path: Path, hands: dict[str, dict[str, np.ndarray]]) -> None:
    np.savez(path, **{f"{side}_{k}": v for side, fields in hands.items() for k, v in fields.items()})


def load_hands(path: Path) -> dict[str, dict[str, np.ndarray]]:
    return {side: _load(path, f"{side}_", HAND_SHAPES) for side in SIDES}
