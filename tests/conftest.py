"""Test fixtures. All media is generated on the fly, so the repository holds no images of people."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from clipsonar.common import Reporter

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


class RecordingReporter(Reporter):
    def __init__(self, stop_after_checks: int | None = None) -> None:
        self.messages: list[str] = []
        self._checks_left = stop_after_checks

    def log(self, message: str) -> None:
        self.messages.append(message)

    def should_stop(self) -> bool:
        if self._checks_left is None:
            return False
        self._checks_left -= 1
        return self._checks_left < 0

    def text(self) -> str:
        return "\n".join(self.messages)


class FakeMatcher:
    """Scores an image by its blue level: a blue value of 200 means a score of 0.80, and 0 means no face."""

    def score(self, image: np.ndarray) -> float | None:
        blue = float(image[..., 0].mean())
        return None if blue < 10 else round(blue / 250, 3)


def make_image(path: Path, blue: int = 200, size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[..., 0] = blue
    ok, data = cv2.imencode(path.suffix, image)
    assert ok
    data.tofile(str(path))
    return path


def make_video(path: Path, color: str = "blue", seconds: int = 3, audio: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=64x48:r=10:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=d={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def rep() -> RecordingReporter:
    return RecordingReporter()
