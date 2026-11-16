"""Printable sheets: ArUco markers for the replica rifle and a checkerboard for calibration.

Print at 100 % scale (no "fit to page") and measure one marker or square afterwards; the measured
size is what the tracking and calibration steps need.
"""

from pathlib import Path

import cv2
import numpy as np

A4_MM = (210.0, 297.0)
DICTIONARY = cv2.aruco.DICT_4X4_50


def _mm(value_mm: float, dpi: int) -> int:
    return round(value_mm / 25.4 * dpi)


def aruco_sheet(marker_mm: float = 30.0, count: int = 12, dpi: int = 300) -> np.ndarray:
    """A4 page of ArUco markers ids 0..count-1, each with a white margin and its id underneath."""
    page_w, page_h = _mm(A4_MM[0], dpi), _mm(A4_MM[1], dpi)
    page = np.full((page_h, page_w), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    side = _mm(marker_mm, dpi)
    cell = side + _mm(15.0, dpi)
    margin = _mm(15.0, dpi)
    per_row = (page_w - 2 * margin) // cell
    rows = (page_h - 2 * margin) // cell
    if per_row * rows < count:
        raise ValueError(f"{count} markers of {marker_mm} mm do not fit on one page")
    for marker_id in range(count):
        x = margin + (marker_id % per_row) * cell
        y = margin + (marker_id // per_row) * cell
        page[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
        cv2.putText(page, f"id {marker_id}", (x, y + side + _mm(6.0, dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                    dpi / 300.0, 0, max(1, dpi // 150))
    caption = f"ArUco 4x4_50, {marker_mm:g} mm per marker at 100 % scale"
    cv2.putText(page, caption, (margin, page_h - margin // 2), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0, 0,
                max(1, dpi // 150))
    return page


def checkerboard_sheet(pattern: tuple[int, int] = (9, 6), square_mm: float = 25.0, dpi: int = 300) -> np.ndarray:
    """A4 landscape checkerboard with `pattern` inner corners (cols, rows)."""
    page_w, page_h = _mm(A4_MM[1], dpi), _mm(A4_MM[0], dpi)
    page = np.full((page_h, page_w), 255, np.uint8)
    square = _mm(square_mm, dpi)
    cols, rows = pattern[0] + 1, pattern[1] + 1
    if cols * square > page_w or rows * square > page_h:
        raise ValueError("checkerboard does not fit on one page")
    x0, y0 = (page_w - cols * square) // 2, (page_h - rows * square) // 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                page[y0 + r * square : y0 + (r + 1) * square, x0 + c * square : x0 + (c + 1) * square] = 0
    caption = f"{pattern[0]}x{pattern[1]} inner corners, {square_mm:g} mm squares at 100 % scale"
    cv2.putText(page, caption, (x0, page_h - _mm(6.0, dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0, 0,
                max(1, dpi // 150))
    return page


def write_sheets(out_dir: Path, dpi: int = 300) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "aruco_markers.png", out_dir / "checkerboard.png"]
    cv2.imwrite(str(paths[0]), aruco_sheet(dpi=dpi))
    cv2.imwrite(str(paths[1]), checkerboard_sheet(dpi=dpi))
    return paths
