"""Video to numbered frames at a fixed rate."""

import shutil
import subprocess
from pathlib import Path


def extract(video: Path, rgb_dir: Path, fps: float) -> int:
    """Writes the video's frames to rgb_dir/00001.png, 00002.png ... at `fps`. Returns the frame count.

    Old frames in rgb_dir are removed first so a rerun never mixes two extractions.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")
    if not video.is_file():
        raise FileNotFoundError(video)
    if rgb_dir.exists():
        shutil.rmtree(rgb_dir)
    rgb_dir.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", str(rgb_dir / "%05d.png")],
        check=True,
    )
    return len(list_frames(rgb_dir))


def list_frames(rgb_dir: Path) -> list[Path]:
    return sorted(rgb_dir.glob("*.png"))
