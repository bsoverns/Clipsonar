import csv

import pytest

from clipsonar.common import Cancelled, VideoLayout, read_json, run_ffmpeg
from clipsonar.videos import VideoScanConfig, scan_videos

from conftest import FakeMatcher, RecordingReporter, make_image, make_video, needs_ffmpeg

pytestmark = needs_ffmpeg


def config(tmp_path, **options):
    make_image(tmp_path / "samples" / "ref.jpg")
    defaults = {"threshold": 0.5, "interval": 1.0, "pre": 1.0, "post": 1.0, "min_gap": 2.0}
    return VideoScanConfig(tmp_path / "samples", [tmp_path / "lib"], tmp_path / "dest", **{**defaults, **options})


def clips(tmp_path):
    return sorted(p.name for p in VideoLayout.under(tmp_path / "dest").clips.iterdir())


def test_extracts_clips_only_from_matching_videos(tmp_path, rep):
    make_video(tmp_path / "lib" / "match.mp4", "blue")
    make_video(tmp_path / "lib" / "other.mp4", "black")
    scan_videos(config(tmp_path), rep, FakeMatcher())
    names = clips(tmp_path)
    assert names and all(n.startswith("match__") and "__s" in n for n in names)


def test_same_file_name_in_two_folders_keeps_results_apart(tmp_path, rep):
    make_video(tmp_path / "lib" / "a" / "IMG_0001.mp4", "blue")
    make_video(tmp_path / "lib" / "b" / "IMG_0001.mp4", "blue")
    cfg = config(tmp_path, min_gap=100.0)
    scan_videos(cfg, rep, FakeMatcher())
    assert len(clips(tmp_path)) == 2
    with open(VideoLayout.under(cfg.destination).hits, newline="", encoding="utf-8") as f:
        assert len({row["video"] for row in csv.DictReader(f)}) == 2


def test_preview_then_real_run_extracts_without_rescanning(tmp_path, rep):
    make_video(tmp_path / "lib" / "match.mp4", "blue")
    scan_videos(config(tmp_path, dry_run=True), rep, FakeMatcher())
    assert clips(tmp_path) == []

    class NoScans:
        def score(self, image):
            raise AssertionError("videos should not be scanned again")

    scan_videos(config(tmp_path), rep, NoScans())
    assert clips(tmp_path)


def test_broken_video_is_logged_and_left_in_place_by_default(tmp_path, rep):
    broken = tmp_path / "lib" / "broken.mp4"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"this is not a video")
    cfg = config(tmp_path)
    scan_videos(cfg, rep, FakeMatcher())
    assert broken.exists()
    assert str(broken) in read_json(VideoLayout.under(cfg.destination).state)["failed"]


def test_broken_video_moves_only_when_asked(tmp_path, rep):
    broken = tmp_path / "lib" / "broken.mp4"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"this is not a video")
    cfg = config(tmp_path, move_broken_sources=True)
    scan_videos(cfg, rep, FakeMatcher())
    assert not broken.exists()
    assert (VideoLayout.under(cfg.destination).broken_sources / "broken.mp4").exists()


def test_retry_failed_scans_the_video_again(tmp_path, rep):
    video = tmp_path / "lib" / "flaky.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not readable yet")
    scan_videos(config(tmp_path), rep, FakeMatcher())
    assert clips(tmp_path) == []
    make_video(video, "blue")
    scan_videos(config(tmp_path), rep, FakeMatcher())
    assert clips(tmp_path) == []
    scan_videos(config(tmp_path, retry_failed=True), rep, FakeMatcher())
    assert clips(tmp_path)


SLOW_INPUT = ["-re", "-f", "lavfi", "-i", "color=s=64x48:d=60"]


def test_stopping_ffmpeg_leaves_no_partial_file(tmp_path):
    rep = RecordingReporter(stop_after_checks=0)
    with pytest.raises(Cancelled):
        run_ffmpeg(SLOW_INPUT, tmp_path / "out.mp4", rep)
    assert list(tmp_path.iterdir()) == []


def test_ffmpeg_timeout_fails_and_leaves_no_partial_file(tmp_path, rep):
    result = run_ffmpeg(SLOW_INPUT, tmp_path / "out.mp4", rep, timeout=0.5)
    assert not result.ok and "timed out" in result.error
    assert list(tmp_path.iterdir()) == []
