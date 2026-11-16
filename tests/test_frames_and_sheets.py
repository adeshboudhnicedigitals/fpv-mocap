import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from mocap import frames, sheets


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_extract_writes_numbered_frames_at_the_requested_rate(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
                    str(video)], check=True)
    rgb = tmp_path / "rgb"
    (rgb).mkdir()
    (rgb / "99999.png").write_bytes(b"stale")
    assert frames.extract(video, rgb, 60.0) == 60
    names = [p.name for p in frames.list_frames(rgb)]
    assert names[0] == "00001.png" and names[-1] == "00060.png"


def test_aruco_sheet_markers_are_detectable_and_true_size():
    dpi = 150
    page = sheets.aruco_sheet(marker_mm=30.0, count=12, dpi=dpi)
    assert page.shape == (round(297 / 25.4 * dpi), round(210 / 25.4 * dpi))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(sheets.DICTIONARY))
    corners, ids, _ = detector.detectMarkers(page)
    assert sorted(ids.ravel().tolist()) == list(range(12))
    side_px = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    assert side_px == pytest.approx(30.0 / 25.4 * dpi, abs=2.0)


def test_checkerboard_sheet_is_detectable():
    page = sheets.checkerboard_sheet((9, 6), 25.0, dpi=100)
    found, _ = cv2.findChessboardCorners(page, (9, 6))
    assert found
