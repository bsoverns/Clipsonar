"""End-to-end checks with the real face model, skipped unless it is already downloaded.

These use the public-domain astronaut photo that ships inside scikit-image, so no image of a
person is stored in this repository.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from clipsonar.common import PhotoLayout, VideoLayout
from clipsonar.faces import MODEL_NAME, FaceMatcher, read_image
from clipsonar.photos import PhotoScanConfig, scan_photos
from clipsonar.videos import VideoScanConfig, scan_videos

from conftest import make_image, needs_ffmpeg

MODEL_DIR = Path.home() / ".insightface" / "models" / MODEL_NAME

pytestmark = [pytest.mark.model, pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="face model is not downloaded")]


def write_person(path: Path, flip: bool = False) -> Path:
    from skimage import data

    image = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    if flip:
        image = cv2.flip(image, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def library(tmp_path):
    write_person(tmp_path / "samples" / "person.png")
    write_person(tmp_path / "lib" / "sub" / "same_person.png", flip=True)
    make_image(tmp_path / "lib" / "no_person.png", blue=120, size=(400, 400))
    return tmp_path


def test_matcher_recognizes_the_person(library, rep):
    matcher = FaceMatcher.load(library / "samples", False, rep)
    assert matcher.score(read_image(library / "lib" / "sub" / "same_person.png")) > 0.6
    assert matcher.score(np.zeros((400, 400, 3), dtype=np.uint8)) is None


def test_photo_scan_end_to_end(library, rep):
    scan_photos(PhotoScanConfig(library / "samples", [library / "lib"], library / "dest"), rep)
    names = [p.name for p in PhotoLayout.under(library / "dest").matches.iterdir()]
    assert len(names) == 1 and names[0].startswith("same_person__s")


@needs_ffmpeg
def test_video_scan_end_to_end(library, rep):
    import subprocess

    source = write_person(library / "frame.png")
    video = library / "lib" / "movie.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(source), "-t", "3", "-r", "5",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    cfg = VideoScanConfig(library / "samples", [library / "lib"], library / "dest", interval=1.0, pre=0.5, post=1.0,
                          min_gap=2.0)
    scan_videos(cfg, rep)
    names = [p.name for p in VideoLayout.under(library / "dest").clips.iterdir()]
    assert names and all(n.startswith("movie__") for n in names)
