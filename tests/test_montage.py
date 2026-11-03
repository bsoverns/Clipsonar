import pytest

from clipsonar.common import JobError, read_json
from clipsonar.montage import (
    PhotoMontageSettings,
    VideoMontageSettings,
    build_photo_montage,
    build_video_montage,
    has_audio,
)
from clipsonar.review import build_review_video, filter_low_scores, set_aside_broken_photos

from conftest import make_image, make_video, needs_ffmpeg

pytestmark = needs_ffmpeg

SMALL = {"width": 160, "height": 120, "fps": 5}


def photos(folder, count=5):
    for i in range(count):
        make_image(folder / f"p{i}__s0.{50 + i}.jpg")
    return folder


def playlist(montages):
    return (montages / "playlist.m3u").read_text(encoding="utf-8").split()


def test_photo_montage_builds_batches_and_playlist(tmp_path, rep):
    montages = tmp_path / "m"
    settings = PhotoMontageSettings(batch_size=2, duration=0.2, **SMALL)
    build_photo_montage(photos(tmp_path / "p"), montages, settings, rep)
    assert playlist(montages) == ["#EXTM3U", "Montage_batch_001.mp4", "Montage_batch_002.mp4", "Montage_batch_003.mp4"]
    assert has_audio(montages / "Montage_batch_001.mp4", rep) is False


def test_photo_montage_resume_rebuilds_only_missing_batches(tmp_path, rep):
    montages = tmp_path / "m"
    settings = PhotoMontageSettings(batch_size=2, duration=0.2, **SMALL)
    folder = photos(tmp_path / "p")
    build_photo_montage(folder, montages, settings, rep)
    kept = (montages / "Montage_batch_001.mp4").stat().st_mtime_ns
    (montages / "Montage_batch_002.mp4").unlink()
    build_photo_montage(folder, montages, settings, rep)
    assert (montages / "Montage_batch_002.mp4").exists()
    assert (montages / "Montage_batch_001.mp4").stat().st_mtime_ns == kept


def test_changed_settings_require_start_over(tmp_path, rep):
    montages = tmp_path / "m"
    folder = photos(tmp_path / "p", 2)
    build_photo_montage(folder, montages, PhotoMontageSettings(batch_size=2, duration=0.2, **SMALL), rep)
    changed = PhotoMontageSettings(batch_size=2, duration=0.4, **SMALL)
    with pytest.raises(JobError, match="Start over"):
        build_photo_montage(folder, montages, changed, rep)
    build_photo_montage(folder, montages, changed, rep, restart=True)
    assert read_json(montages / "batches" / "order.json")["settings"]["duration"] == 0.4


def test_new_files_are_noted_on_resume(tmp_path, rep):
    montages = tmp_path / "m"
    folder = photos(tmp_path / "p", 2)
    settings = PhotoMontageSettings(batch_size=2, duration=0.2, **SMALL)
    build_photo_montage(folder, montages, settings, rep)
    make_image(folder / "extra__s0.70.jpg")
    build_photo_montage(folder, montages, settings, rep)
    assert "1 newer file(s)" in rep.text()


def test_score_filter_and_no_matching_files(tmp_path, rep):
    folder = photos(tmp_path / "p", 3)
    settings = PhotoMontageSettings(batch_size=10, duration=0.2, min_score=0.52, shuffle=False, **SMALL)
    build_photo_montage(folder, tmp_path / "m", settings, rep)
    assert len(read_json(tmp_path / "m" / "batches" / "order.json")["items"]) == 1
    with pytest.raises(JobError):
        strict = PhotoMontageSettings(batch_size=10, duration=0.2, min_score=0.99, **SMALL)
        build_photo_montage(folder, tmp_path / "m2", strict, rep)


def test_file_names_with_apostrophes_work(tmp_path, rep):
    folder = tmp_path / "it's here"
    make_image(folder / "bob's__s0.60.jpg")
    build_photo_montage(folder, tmp_path / "m", PhotoMontageSettings(batch_size=2, duration=0.2, **SMALL), rep)
    assert (tmp_path / "m" / "Montage_batch_001.mp4").exists()


def test_video_montage_mixes_clips_with_and_without_audio(tmp_path, rep):
    clips = tmp_path / "c"
    make_video(clips / "a__s0.60.mp4", audio=True)
    make_video(clips / "b__s0.70.mp4", audio=False)
    (clips / "c__s0.80.mp4").write_bytes(b"broken")
    montages = tmp_path / "m"
    build_video_montage(clips, montages, VideoMontageSettings(batch_size=10, **SMALL), rep,
                        broken_dir=tmp_path / "broken")
    output = montages / "Montage_batch_001.mp4"
    assert has_audio(output, rep) is True
    assert (tmp_path / "broken" / "c__s0.80.mp4").exists()
    assert list((montages / "batches" / "norm").iterdir()) == []


def test_review_video(tmp_path, rep):
    folder = photos(tmp_path / "p", 2)
    output = tmp_path / "it's a folder" / "Review.mp4"
    build_review_video(folder, output, rep, duration=0.2, width=320, height=240, fps=5)
    assert output.stat().st_size > 0
    assert sorted(p.name for p in output.parent.iterdir()) == ["Review.mp4"]


def test_filter_low_scores_moves_files_and_preview_does_not(tmp_path, rep):
    folder = photos(tmp_path / "p", 5)
    assert filter_low_scores(folder, 0.52, rep, dry_run=True) == 2
    assert not (folder / "low_score").exists()
    assert filter_low_scores(folder, 0.52, rep) == 2
    assert len(list((folder / "low_score").iterdir())) == 2


def test_broken_photos_are_set_aside(tmp_path, rep):
    folder = photos(tmp_path / "p", 2)
    (folder / "bad__s0.90.jpg").write_bytes(b"not an image")
    assert set_aside_broken_photos(folder, rep, tmp_path / "broken") == 1
    assert (tmp_path / "broken" / "bad__s0.90.jpg").exists()
