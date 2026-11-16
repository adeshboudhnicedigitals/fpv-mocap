"""Camera intrinsics: from a checkerboard calibration or from a guessed horizontal field of view."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fps: float
    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    source: str = "fov_guess"  # "calibrated" or "fov_guess"
    rms: float | None = None  # reprojection error in pixels, calibrated only

    def save(self, path: Path) -> None:
        data = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "K": self.K.tolist(),
            "dist": np.ravel(self.dist).tolist(),
            "source": self.source,
            "rms": self.rms,
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: Path) -> "Camera":
        data = json.loads(path.read_text())
        return Camera(
            width=data["width"],
            height=data["height"],
            fps=data["fps"],
            K=np.array(data["K"], dtype=np.float64),
            dist=np.array(data["dist"], dtype=np.float64),
            source=data["source"],
            rms=data.get("rms"),
        )


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole K with square pixels and the principal point at the image centre."""
    if not 1.0 < hfov_deg < 179.0:
        raise ValueError(f"horizontal FOV out of range: {hfov_deg}")
    f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def checkerboard_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """Inner-corner positions on the board plane (z = 0), in metres, in OpenCV's corner order."""
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def calibrate(
    images: list[np.ndarray], pattern: tuple[int, int], square_m: float
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Calibrates from grayscale checkerboard images. `pattern` counts inner corners (cols, rows).

    Returns K, distortion, RMS reprojection error in pixels and the number of images used.
    """
    if not images:
        raise ValueError("no images")
    size = images[0].shape[1], images[0].shape[0]
    board = checkerboard_points(pattern, square_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for gray in images:
        if (gray.shape[1], gray.shape[0]) != size:
            raise ValueError("calibration images differ in size")
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        object_points.append(board)
        image_points.append(corners)
    if len(object_points) < 3:
        raise ValueError(f"checkerboard found in {len(object_points)} images; need at least 3")
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    return K, dist.ravel(), float(rms), len(object_points)
