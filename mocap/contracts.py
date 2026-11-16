"""Load, save and validate the .npz files the stages exchange (see docs/plan.md, Data contracts)."""

import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
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


_LOAD_ERRORS = (OSError, ValueError, zipfile.BadZipFile)


@contextmanager
def _open(path: Path) -> Iterator[np.lib.npyio.NpzFile]:
    """Open an npz so every failure is a ContractError and no file handle outlives the block."""
    if not Path(path).exists():
        raise ContractError(f"{path} not found")
    with open(path, "rb") as fh:  # np.load leaks its own handle when the archive is corrupt
        try:
            data = np.load(fh)
        except _LOAD_ERRORS as e:
            raise ContractError(f"{path}: not a valid npz ({e})") from e
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ContractError(f"{path}: not a valid npz (got a bare array)")
        with data:
            yield data


def _load(path: Path, prefix: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n: int | None = None
    with _open(path) as data:
        for field, tail in shapes.items():
            key = f"{prefix}{field}"
            if key not in data:
                raise ContractError(f"{path}: missing {key}")
            try:
                arr = data[key]
            except _LOAD_ERRORS as e:
                raise ContractError(f"{path}: not a valid npz ({e})") from e
            n = len(arr) if n is None else n
            if arr.shape != (n, *tail):
                raise ContractError(f"{path}: {key} has shape {arr.shape}, expected {(n, *tail)}")
            out[field] = arr.astype(bool) if field == "valid" else arr
    return out


ALIGNED_KEYS = ("gun_valid", "gun_source", "T_cam_gun") + tuple(
    f"{side}_{k}" for side in SIDES for k in ("T_cam_wrist", "keypoints", "valid", "grip_weight")
)


def load_aligned(path: Path) -> dict[str, np.ndarray]:
    """Load the keys overlay.draw_frame needs from aligned.npz."""
    out: dict[str, np.ndarray] = {}
    with _open(path) as data:
        for key in ALIGNED_KEYS:
            if key not in data:
                raise ContractError(f"{path}: missing {key}")
            try:
                out[key] = data[key]
            except _LOAD_ERRORS as e:
                raise ContractError(f"{path}: not a valid npz ({e})") from e
    return out


def save_gun(path: Path, gun: dict[str, np.ndarray]) -> None:
    np.savez(path, **gun)


def load_gun(path: Path) -> dict[str, np.ndarray]:
    return _load(path, "", GUN_SHAPES)


def save_hands(path: Path, hands: dict[str, dict[str, np.ndarray]]) -> None:
    np.savez(path, **{f"{side}_{k}": v for side, fields in hands.items() for k, v in fields.items()})


def load_hands(path: Path) -> dict[str, dict[str, np.ndarray]]:
    return {side: _load(path, f"{side}_", HAND_SHAPES) for side in SIDES}
