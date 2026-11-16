import cv2
import numpy as np
import pytest

from mocap import cli
from mocap.camera import Camera, intrinsics_from_fov
from mocap.contracts import save_gun, save_hands


def _make_clip(root, n=12):
    clip = root / "c"
    (clip / "rgb").mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(clip / "rgb" / f"{i + 1:05d}.png"), np.zeros((108, 192, 3), np.uint8))
    Camera(192, 108, 60.0, intrinsics_from_fov(192, 108, 90)).save(clip / "camera.json")
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, 2, 3] = 1.0
    save_gun(clip / "gun.npz", {"T_cam_gun": pose, "valid": np.ones(n, bool), "score": np.ones(n)})
    hand = {
        "T_cam_wrist": pose.copy(),
        "finger_rot": np.tile(np.eye(3), (n, 15, 1, 1)),
        "keypoints": np.tile([0.0, 0.0, 1.0], (n, 21, 1)),
        "valid": np.ones(n, bool),
        "score": np.ones(n),
    }
    save_hands(clip / "hands.npz", {"left": hand, "right": hand})
    return clip


def test_align_then_overlay_write_their_files(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path)
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    cli.main(["align", "c"])
    assert (clip / "aligned.npz").exists()
    cli.main(["overlay", "c"])
    cap = cv2.VideoCapture(str(clip / "overlay.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12


def test_align_without_inputs_fails_with_a_message(tmp_path, monkeypatch):
    (tmp_path / "c").mkdir()
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    with pytest.raises(SystemExit, match="gun.npz"):
        cli.main(["align", "c"])
