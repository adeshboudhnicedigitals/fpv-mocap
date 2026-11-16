"""The GPU script (notebooks/gpu_stages.py) cannot run here; this mirrors the arrays it builds and
proves its output layout round-trips through the contracts and is accepted by align.run."""
import ast
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mocap import align
from mocap.contracts import HAND_SHAPES, load_gun, load_hands, save_gun, save_hands

N = 20
SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "gpu_stages.py"


def empty_hand():  # same shapes/dtypes as the script's empty_hand()
    return {
        "T_cam_wrist": np.tile(np.eye(4), (N, 1, 1)),
        "finger_rot": np.tile(np.eye(3), (N, 15, 1, 1)),
        "keypoints": np.zeros((N, 21, 3)),
        "valid": np.zeros(N, bool),
        "score": np.zeros(N),
    }


# The float16 cast below documents the cast WiLoR output needs; it does not guard script behaviour.
def test_script_layout_round_trips_and_aligns(tmp_path):
    rng = np.random.default_rng(0)
    hands = {"left": empty_hand(), "right": empty_hand()}
    for h in hands.values():
        for i in range(5, N):  # first frames undetected: identity + invalid
            rots = Rotation.random(15, random_state=i).as_matrix()
            h["finger_rot"][i] = rots.astype(np.float16)  # float16 into float64
            h["keypoints"][i] = rng.normal(size=(21, 3)).astype(np.float16)
            T = np.eye(4)
            T[:3, 3] = [0.1, 0.0, 1.0]
            h["T_cam_wrist"][i] = T
            h["valid"][i] = True
            h["score"][i] = 1.0
        assert h["finger_rot"].dtype == np.float64 and h["keypoints"].dtype == np.float64
    T_gun = np.tile(np.eye(4), (N, 1, 1))
    gun_valid = np.zeros(N, bool)
    for i in range(3, N):
        T_gun[i] = np.eye(4, dtype=np.float32).reshape(4, 4)
        T_gun[i, 2, 3] = 1.0
        gun_valid[i] = True
    save_hands(tmp_path / "hands.npz", hands)
    save_gun(tmp_path / "gun.npz", {"T_cam_gun": T_gun, "valid": gun_valid, "score": gun_valid.astype(float)})
    loaded_hands, loaded_gun = load_hands(tmp_path / "hands.npz"), load_gun(tmp_path / "gun.npz")
    out = align.run(loaded_gun, loaded_hands)
    assert out["T_cam_gun"].shape == (N, 4, 4)
    assert np.isfinite(out["T_cam_gun"]).all()


def test_script_compiles():
    compile(SCRIPT.read_text(encoding="utf-8"), "gpu_stages.py", "exec")


def test_script_empty_hand_matches_contract_shapes():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "empty_hand")
    ns = {"np": np, "N": N}
    exec(compile(ast.Module([fn], []), "empty_hand", "exec"), ns)
    hand = ns["empty_hand"]()
    assert set(hand) == set(HAND_SHAPES)
    for key, shape in HAND_SHAPES.items():
        assert hand[key].shape == (N, *shape), key
