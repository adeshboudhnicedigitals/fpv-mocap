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


def test_overlay_with_incomplete_aligned_fails_with_a_message(tmp_path, monkeypatch):
    clip = _make_clip(tmp_path)
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    cli.main(["align", "c"])
    data = dict(np.load(clip / "aligned.npz"))
    del data["right_keypoints"]
    np.savez(clip / "aligned.npz", **data)
    with pytest.raises(SystemExit, match="right_keypoints"):
        cli.main(["overlay", "c"])


def test_align_prints_wrist_to_gun_distance(tmp_path, monkeypatch, capsys):
    _make_clip(tmp_path)
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    cli.main(["align", "c"])
    out = capsys.readouterr().out
    assert "right: gripping 12 frames, median wrist-to-gun distance 0.00 m" in out
    assert "left: gripping 12 frames" in out


def test_align_hints_when_nothing_grips(tmp_path, monkeypatch, capsys):
    clip = _make_clip(tmp_path)
    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    hands = dict(np.load(clip / "hands.npz"))
    hands["right_T_cam_wrist"] = hands["right_T_cam_wrist"].copy()
    hands["right_T_cam_wrist"][:, 0, 3] = np.arange(12) * 0.1
    np.savez(clip / "hands.npz", **hands)
    cli.main(["align", "c"])
    out = capsys.readouterr().out
    assert "right: gripping 0 frames" in out
    assert "--eps-t" in out and "--eps-r-deg" in out
