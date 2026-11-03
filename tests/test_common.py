from pathlib import Path

from clipsonar.common import IMAGE_EXTS, format_hms, iter_files, partial_path, score_from_name, set_aside, unique_target
from clipsonar.videos import clip_name, dedupe_hits


def test_score_from_name():
    assert score_from_name(Path("IMG_0042__s0.62.jpg")) == 0.62
    assert score_from_name(Path("clip__ab12cd__00h01m00s__s0.45.mp4")) == 0.45
    assert score_from_name(Path("IMG_0042.jpg")) == 0.0


def test_unique_target_never_reuses_a_name(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "a_1.jpg").touch()
    assert unique_target(tmp_path / "a.jpg").name == "a_2.jpg"
    assert unique_target(tmp_path / "b.jpg").name == "b.jpg"


def test_set_aside_keeps_both_files_on_collision(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "a.jpg").write_text("first")
    (tmp_path / "a.jpg").write_text("second")
    target = set_aside(tmp_path / "a.jpg", tmp_path / "bad")
    assert target.name == "a_1.jpg"
    assert (tmp_path / "bad" / "a.jpg").read_text() == "first"
    assert target.read_text() == "second"


def test_format_hms():
    assert format_hms(3725.9) == "01h02m05s"


def test_iter_files_skips_output_system_and_partial_files(tmp_path, rep):
    source = tmp_path / "src"
    for name in ["a.JPG", "sub/b.png", "@eaDir/thumb.jpg", "out/copy.jpg", "note.txt"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (source / partial_path(Path("c.jpg")).name).touch()
    names = {p.name for p in iter_files([source, source / "sub"], IMAGE_EXTS, rep, exclude=[source / "out"])}
    assert names == {"a.JPG", "b.png"}


def test_dedupe_keeps_first_time_and_best_score():
    hits = [(10.0, 0.5), (12.0, 0.7), (60.0, 0.4), (70.0, 0.3)]
    assert dedupe_hits(hits, 45.0) == [(10.0, 0.7), (60.0, 0.4)]


def test_clip_names_differ_for_same_file_name_in_different_folders():
    first = clip_name(Path("/a/IMG_0001.mp4"), 5.0, 0.5)
    second = clip_name(Path("/b/IMG_0001.mp4"), 5.0, 0.5)
    assert first != second
    assert score_from_name(Path(first)) == 0.5
