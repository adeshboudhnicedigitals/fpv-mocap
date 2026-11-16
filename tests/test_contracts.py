from pathlib import Path

import numpy as np
import pytest

from mocap.contracts import ContractError, load_gun, load_hands, save_gun, save_hands


def _gun(n: int) -> dict:
    return {"T_cam_gun": np.tile(np.eye(4), (n, 1, 1)), "valid": np.ones(n, bool), "score": np.ones(n)}


def _hand(n: int) -> dict:
    return {
        "T_cam_wrist": np.tile(np.eye(4), (n, 1, 1)),
        "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
        "keypoints": np.zeros((n, 21, 3)),
        "valid": np.ones(n, bool),
        "score": np.ones(n),
    }


def test_gun_round_trip(tmp_path: Path):
    save_gun(tmp_path / "gun.npz", _gun(5))
    out = load_gun(tmp_path / "gun.npz")
    assert out["T_cam_gun"].shape == (5, 4, 4)
    assert out["valid"].dtype == bool


def test_hands_round_trip(tmp_path: Path):
    save_hands(tmp_path / "hands.npz", {"left": _hand(4), "right": _hand(4)})
    out = load_hands(tmp_path / "hands.npz")
    assert set(out) == {"left", "right"}
    assert out["right"]["finger_rot"].shape == (4, 15, 3, 3)


def test_missing_key_names_the_key(tmp_path: Path):
    hands = {"left": _hand(3), "right": _hand(3)}
    del hands["right"]["keypoints"]
    save_hands(tmp_path / "hands.npz", hands)
    with pytest.raises(ContractError, match="right_keypoints"):
        load_hands(tmp_path / "hands.npz")


def test_wrong_shape_is_rejected(tmp_path: Path):
    gun = _gun(3)
    gun["T_cam_gun"] = np.zeros((3, 3, 3))
    save_gun(tmp_path / "gun.npz", gun)
    with pytest.raises(ContractError, match="T_cam_gun"):
        load_gun(tmp_path / "gun.npz")


def test_frame_count_must_match_across_fields(tmp_path: Path):
    gun = _gun(3)
    gun["score"] = np.ones(4)
    save_gun(tmp_path / "gun.npz", gun)
    with pytest.raises(ContractError, match="score"):
        load_gun(tmp_path / "gun.npz")


def test_missing_file_is_a_contract_error(tmp_path: Path):
    with pytest.raises(ContractError, match="gun.npz"):
        load_gun(tmp_path / "gun.npz")


def test_garbage_npz_is_a_contract_error(tmp_path: Path):
    (tmp_path / "gun.npz").write_bytes(b"not a zip at all")
    with pytest.raises(ContractError, match="gun.npz"):
        load_gun(tmp_path / "gun.npz")


def test_truncated_npz_is_a_contract_error(tmp_path: Path):
    save_gun(tmp_path / "gun.npz", _gun(5))
    raw = (tmp_path / "gun.npz").read_bytes()
    (tmp_path / "gun.npz").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ContractError, match="gun.npz"):
        load_gun(tmp_path / "gun.npz")


def test_npy_is_a_contract_error(tmp_path: Path):
    np.save(tmp_path / "gun.npz.npy", np.zeros(3))
    (tmp_path / "gun.npz.npy").rename(tmp_path / "gun.npz")
    with pytest.raises(ContractError, match="gun.npz"):
        load_gun(tmp_path / "gun.npz")


def _aligned_arrays(n: int) -> dict:
    out = {"gun_valid": np.ones(n, bool), "gun_source": np.zeros(n, np.int8),
           "T_cam_gun": np.tile(np.eye(4), (n, 1, 1))}
    for side in ("left", "right"):
        out[f"{side}_T_cam_wrist"] = np.tile(np.eye(4), (n, 1, 1))
        out[f"{side}_keypoints"] = np.zeros((n, 21, 3))
        out[f"{side}_valid"] = np.ones(n, bool)
        out[f"{side}_grip_weight"] = np.zeros(n)
    return out


def test_load_aligned_round_trip_and_errors(tmp_path: Path):
    from mocap.contracts import load_aligned

    np.savez(tmp_path / "aligned.npz", **_aligned_arrays(3))
    assert load_aligned(tmp_path / "aligned.npz")["T_cam_gun"].shape == (3, 4, 4)
    broken = _aligned_arrays(3)
    del broken["left_grip_weight"]
    np.savez(tmp_path / "bad.npz", **broken)
    with pytest.raises(ContractError, match="left_grip_weight"):
        load_aligned(tmp_path / "bad.npz")
    with pytest.raises(ContractError, match="nope.npz"):
        load_aligned(tmp_path / "nope.npz")
