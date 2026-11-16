import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mocap.align import blend_pose, grip_offset, grip_weight, run, smooth_fingers
from mocap.contracts import ContractError
from mocap.transforms import relative


def _pose(t, yaw_deg=0.0) -> np.ndarray:
    a = np.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    T[:3, 3] = t
    return T


def test_grip_weight_is_high_for_a_steady_offset_and_low_when_it_moves():
    n = 60
    rel = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        rel[i, :3, 3] = [0.05, 0, 0.1] if i < 30 else [0.05 + 0.05 * (i - 30), 0, 0.1]
    w = grip_weight(rel, np.ones(n, bool))
    assert w[10] > 0.99
    assert w[50] < 0.01


def test_grip_weight_is_zero_on_invalid_frames():
    rel = np.tile(np.eye(4), (30, 1, 1))
    valid = np.ones(30, bool)
    valid[10:20] = False
    w = grip_weight(rel, valid, blend=1)
    assert np.all(w[10:20] == 0)


def test_grip_offset_none_without_a_grip():
    rel = np.tile(np.eye(4), (10, 1, 1))
    assert grip_offset(rel, np.zeros(10)) is None


def test_grip_offset_recovers_the_constant_offset():
    off = _pose([0.05, 0.0, 0.1], 20)
    rel = np.tile(off, (20, 1, 1))
    assert np.allclose(grip_offset(rel, np.ones(20)), off, atol=1e-9)


def test_blend_pose_endpoints_and_midpoint():
    A = np.stack([_pose([0, 0, 0], 0)] * 3)
    B = np.stack([_pose([2, 0, 0], 90)] * 3)
    out = blend_pose(A, B, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out[0], A[0])
    assert np.allclose(out[2], B[0])
    assert np.allclose(out[1, :3, 3], [1, 0, 0])
    assert np.allclose(out[1], _pose([1, 0, 0], 45))


def test_smooth_fingers_reduces_jitter_and_keeps_valid_rotations():
    rng = np.random.default_rng(0)
    n = 40
    base = Rotation.from_rotvec([0.3, 0.0, 0.0])
    noisy = np.empty((n, 15, 3, 3))
    for i in range(n):
        for j in range(15):
            noisy[i, j] = (Rotation.from_rotvec(rng.normal(0, 0.05, 3)) * base).as_matrix()
    out = smooth_fingers(noisy, np.ones(n, bool))
    to_vec = lambda m: Rotation.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().reshape(n, 15, 3)
    assert to_vec(out).std(axis=0).mean() < 0.6 * to_vec(noisy).std(axis=0).mean()
    assert np.allclose(out @ np.swapaxes(out, -1, -2), np.eye(3), atol=1e-6)


def _synthetic(n=60):
    rng = np.random.default_rng(1)
    gun = np.stack([_pose([0.0, 0.0 + 0.002 * i, 1.0], 0.2 * i) for i in range(n)])
    off_r = _pose([0.05, 0.1, 0.0])
    off_l = _pose([-0.05, 0.1, 0.3])
    right = gun @ off_r
    right[:, :3, 3] += rng.normal(0, 0.001, (n, 3))
    left = gun @ off_l
    for i in range(30, n):  # support hand leaves the gun and moves fast
        left[i, :3, 3] += [0.05 * (i - 30), 0.0, 0.0]

    def hand(T):
        return {
            "T_cam_wrist": T,
            "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
            "keypoints": np.zeros((n, 21, 3)),
            "valid": np.ones(n, bool),
            "score": np.ones(n),
        }

    return (
        {"T_cam_gun": gun.copy(), "valid": np.ones(n, bool), "score": np.ones(n)},
        {"left": hand(left), "right": hand(right)},
        gun,
        off_r,
    )


def test_run_locks_the_gripping_hand_and_releases_the_support_hand():
    gun, hands, gun_truth, off_r = _synthetic()
    out = run(gun, hands)
    assert out["right_grip_weight"][30] > 0.99
    locked = out["right_grip_weight"] > 0.99
    rel_t = relative(out["T_cam_gun"], out["right_T_cam_wrist"])[locked][:, :3, 3]
    assert rel_t.std(axis=0).max() < 1e-9
    assert np.allclose(rel_t.mean(axis=0), off_r[:3, 3], atol=1e-3)
    assert out["left_grip_weight"][10] > 0.99
    assert out["left_grip_weight"][55] < 0.01
    assert np.allclose(out["left_T_cam_wrist"][55], hands["left"]["T_cam_wrist"][55])
    assert np.allclose(out["T_cam_gun"], gun_truth)  # gun is never smoothed
    assert np.all(out["gun_source"] == 0)


def test_run_fills_lost_gun_frames_from_the_right_hand():
    gun, hands, gun_truth, _ = _synthetic()
    gun["valid"][25:30] = False
    gun["T_cam_gun"][25:30] = np.eye(4)
    out = run(gun, hands)
    assert np.all(out["gun_source"][25:30] == 1)
    assert np.allclose(out["T_cam_gun"][25:30, :3, 3], gun_truth[25:30, :3, 3], atol=0.01)
    assert out["gun_valid"][25:30].all()


def test_run_leaves_frames_with_no_gun_and_no_hand_invalid():
    gun, hands, _, _ = _synthetic()
    gun["valid"][:5] = False
    hands["right"]["valid"][:5] = False
    out = run(gun, hands)
    assert np.all(out["gun_source"][:5] == -1)
    assert not out["gun_valid"][:5].any()


def test_run_rejects_mismatched_frame_counts():
    gun, hands, _, _ = _synthetic()
    gun = {k: v[:50] for k, v in gun.items()}
    with pytest.raises(ContractError, match="gun has 50"):
        run(gun, hands)


def test_run_tolerates_zeros_on_invalid_frames():
    gun, hands, _, _ = _synthetic()
    gun["valid"][5:10] = False
    hands["left"]["valid"][20:25] = False
    hands["right"]["valid"][40:45] = False
    gun_i = {k: v.copy() for k, v in gun.items()}
    hands_i = {s: {k: v.copy() for k, v in h.items()} for s, h in hands.items()}
    gun_i["T_cam_gun"][5:10] = np.eye(4)
    hands_i["left"]["T_cam_wrist"][20:25] = np.eye(4)
    hands_i["left"]["finger_rot"][20:25] = np.eye(3)
    hands_i["right"]["T_cam_wrist"][40:45] = np.eye(4)
    hands_i["right"]["finger_rot"][40:45] = np.eye(3)
    gun_z = {k: v.copy() for k, v in gun_i.items()}
    hands_z = {s: {k: v.copy() for k, v in h.items()} for s, h in hands_i.items()}
    gun_z["T_cam_gun"][5:10] = 0
    hands_z["left"]["T_cam_wrist"][20:25] = 0
    hands_z["left"]["finger_rot"][20:25] = 0
    hands_z["right"]["T_cam_wrist"][40:45] = 0
    hands_z["right"]["finger_rot"][40:45] = 0
    before = gun_z["T_cam_gun"].copy()
    expected = run(gun_i, hands_i)
    out = run(gun_z, hands_z)
    assert np.array_equal(gun_z["T_cam_gun"], before)  # inputs not mutated
    assert np.all(hands_z["left"]["T_cam_wrist"][20:25] == 0)
    for key, val in expected.items():
        assert np.allclose(out[key], val), key
    assert np.allclose(out["left_T_cam_wrist"][20:25], np.eye(4))


def test_run_grip_thresholds_change_the_weight():
    gun, hands, _, _ = _synthetic()
    rng = np.random.default_rng(2)
    hands["right"]["T_cam_wrist"][:, :3, 3] += rng.normal(0, 0.01, (60, 3))
    tight = run(gun, hands, eps_t=1e-4)["right_grip_weight"]
    loose = run(gun, hands, eps_t=0.2)["right_grip_weight"]
    assert (loose > 0.5).sum() > (tight > 0.5).sum()
