import csv

import pytest

from clipsonar.common import Cancelled, JobError, PhotoLayout
from clipsonar.photos import PhotoScanConfig, scan_photos

from conftest import FakeMatcher, RecordingReporter, make_image


def setup_library(tmp_path):
    samples = tmp_path / "samples"
    make_image(samples / "ref.jpg")
    make_image(tmp_path / "lib" / "match.jpg", blue=200)
    make_image(tmp_path / "lib" / "sub" / "low.jpg", blue=100)
    make_image(tmp_path / "lib" / "noface.jpg", blue=0)
    return samples, [tmp_path / "lib"], tmp_path / "dest"


def rows(layout):
    with open(layout.hits, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matches(layout):
    return sorted(p.name for p in layout.matches.iterdir())


def test_copies_only_matches_and_leaves_sources_alone(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5), rep, FakeMatcher())
    layout = PhotoLayout.under(dest)
    assert matches(layout) == ["match__s0.80.jpg"]
    assert len(rows(layout)) == 3
    assert (tmp_path / "lib" / "match.jpg").exists()


def test_resume_skips_scanned_photos(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    cfg = PhotoScanConfig(samples, sources, dest, threshold=0.5)
    scan_photos(cfg, rep, FakeMatcher())
    make_image(tmp_path / "lib" / "new.jpg", blue=225)
    scan_photos(cfg, rep, FakeMatcher())
    layout = PhotoLayout.under(dest)
    assert len(rows(layout)) == 4
    assert matches(layout) == ["match__s0.80.jpg", "new__s0.90.jpg"]


def test_cancelled_scan_resumes(tmp_path):
    samples, sources, dest = setup_library(tmp_path)
    cfg = PhotoScanConfig(samples, sources, dest, threshold=0.5)
    with pytest.raises(Cancelled):
        scan_photos(cfg, RecordingReporter(stop_after_checks=3), FakeMatcher())
    layout = PhotoLayout.under(dest)
    assert len(rows(layout)) == 1
    scan_photos(cfg, RecordingReporter(), FakeMatcher())
    assert len(rows(layout)) == 3
    assert len({r["source_file"] for r in rows(layout)}) == 3


def test_real_run_after_preview_copies_the_matches(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5, dry_run=True), rep, FakeMatcher())
    layout = PhotoLayout.under(dest)
    assert matches(layout) == []
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5), rep, FakeMatcher())
    assert matches(layout) == ["match__s0.80.jpg"]


def test_lower_threshold_applies_without_rescanning(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5), rep, FakeMatcher())

    class NoScans:
        def score(self, image):
            raise AssertionError("photos should not be scanned again")

    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.3), rep, NoScans())
    assert matches(PhotoLayout.under(dest)) == ["low__s0.40.jpg", "match__s0.80.jpg"]


def test_photos_deleted_during_review_stay_deleted(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    cfg = PhotoScanConfig(samples, sources, dest, threshold=0.5)
    scan_photos(cfg, rep, FakeMatcher())
    layout = PhotoLayout.under(dest)
    (layout.matches / "match__s0.80.jpg").unlink()
    scan_photos(cfg, rep, FakeMatcher())
    assert matches(layout) == []


def test_destination_inside_a_source_is_not_scanned(tmp_path, rep):
    samples, sources, _ = setup_library(tmp_path)
    cfg = PhotoScanConfig(samples, sources, tmp_path / "lib" / "out", threshold=0.5)
    scan_photos(cfg, rep, FakeMatcher())
    scan_photos(cfg, rep, FakeMatcher())
    assert matches(PhotoLayout.under(cfg.destination)) == ["match__s0.80.jpg"]


def test_non_ascii_paths_work(tmp_path, rep):
    samples = tmp_path / "samples"
    make_image(samples / "ref.jpg")
    make_image(tmp_path / "bibliothèque" / "été.jpg", blue=200)
    scan_photos(PhotoScanConfig(samples, [tmp_path / "bibliothèque"], tmp_path / "dest", threshold=0.5),
                rep, FakeMatcher())
    assert matches(PhotoLayout.under(tmp_path / "dest")) == ["été__s0.80.jpg"]


def test_restart_clears_results(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5), rep, FakeMatcher())
    layout = PhotoLayout.under(dest)
    (layout.matches / "stale__s0.99.jpg").touch()
    scan_photos(PhotoScanConfig(samples, sources, dest, threshold=0.5, restart=True), rep, FakeMatcher())
    assert matches(layout) == ["match__s0.80.jpg"]


def test_changed_samples_produce_a_warning(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    cfg = PhotoScanConfig(samples, sources, dest, threshold=0.5)
    scan_photos(cfg, rep, FakeMatcher())
    make_image(samples / "another.jpg")
    scan_photos(cfg, rep, FakeMatcher())
    assert "settings changed" in rep.text()


def test_missing_folders_are_reported(tmp_path, rep):
    samples, sources, dest = setup_library(tmp_path)
    with pytest.raises(JobError):
        scan_photos(PhotoScanConfig(tmp_path / "nope", sources, dest), rep, FakeMatcher())
    with pytest.raises(JobError):
        scan_photos(PhotoScanConfig(samples, [tmp_path / "nope"], dest), rep, FakeMatcher())
