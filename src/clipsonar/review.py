"""Tools for cleaning up matches: a review video, a score filter, and a broken-photo check."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .common import (
    IMAGE_EXTS,
    JobError,
    Reporter,
    require_tools,
    run_ffmpeg,
    run_tool,
    score_from_name,
    set_aside,
    top_level_files,
    write_concat_list,
)
from .montage import _video_filter

LABELS_FILE = "_review_labels.srt"
LIST_FILE = "_review_list.txt"


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _source_map(hits: Path) -> dict[str, str]:
    """Map copied file names back to the original photo paths recorded by the scan."""
    if not hits.exists():
        return {}
    with open(hits, newline="", encoding="utf-8") as f:
        return {row["copied_as"]: row["source_file"] for row in csv.DictReader(f) if row.get("copied_as")}


def build_review_video(photos_dir: Path, output: Path, rep: Reporter, hits: Path | None = None,
                       duration: float = 0.5, min_score: float = 0.0, max_score: float = 1.0,
                       shuffle: bool = False, width: int = 1920, height: int = 1080, fps: int = 30) -> None:
    """Build a fast video with each photo's file name on screen, for spotting false matches.

    Pause on a wrong photo, read its name, and delete that file from the matches folder.
    """
    require_tools("ffmpeg")
    if not Path(photos_dir).is_dir():
        raise JobError(f"Photos folder not found: {photos_dir}. Run the photo scan first.")
    photos = [p for p in top_level_files(photos_dir, IMAGE_EXTS)
              if min_score <= score_from_name(p) <= max_score]
    if not photos:
        raise JobError("No photos match the chosen score range.")
    if shuffle:
        random.shuffle(photos)

    sources = _source_map(hits) if hits else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = output.parent / LABELS_FILE
    list_path = output.parent / LIST_FILE
    with open(labels, "w", encoding="utf-8") as f:
        for i, photo in enumerate(photos):
            source = sources.get(photo.name, "(unavailable)")
            f.write(f"{i + 1}\n{_srt_time(i * duration)} --> {_srt_time((i + 1) * duration)}\n"
                    f"Match: {photo.name}\\NSource: {source}\n\n")
    write_concat_list(list_path, photos, duration)

    # ffmpeg runs from the output folder so the filter can name the labels file without path escaping.
    style = ("Fontname=Consolas,Fontsize=18,PrimaryColour=&H00FFFFFF&,BorderStyle=3,"
             "BackColour=&HA0000000&,Alignment=2,MarginV=30")
    vf = f"{_video_filter(width, height)},subtitles={LABELS_FILE}:force_style='{style}'"
    rep.log(f"Building a review video of {len(photos)} photo(s)...")
    try:
        result = run_ffmpeg(["-f", "concat", "-safe", "0", "-i", LIST_FILE, "-vf", vf, "-r", str(fps),
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart"], output.resolve(), rep, cwd=output.parent)
    finally:
        labels.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)
    if not result.ok:
        raise JobError(f"Could not build the review video: {result.error}")
    rep.log(f"Done: {output}")


def filter_low_scores(photos_dir: Path, min_score: float, rep: Reporter, reject_folder: str = "low_score",
                      dry_run: bool = False) -> int:
    """Move matches below min_score into a subfolder. Returns how many qualify."""
    if not Path(photos_dir).is_dir():
        raise JobError(f"Photos folder not found: {photos_dir}. Run the photo scan first.")
    low = [p for p in top_level_files(photos_dir, IMAGE_EXTS) if 0.0 < score_from_name(p) < min_score]
    if not dry_run:
        for photo in low:
            rep.check()
            set_aside(photo, photos_dir / reject_folder)
    action = "would move" if dry_run else "moved"
    rep.log(f"{len(low)} photo(s) below {min_score:.2f} {action} to {photos_dir / reject_folder}.")
    return len(low)


def set_aside_broken_photos(photos_dir: Path, rep: Reporter, broken_dir: Path | None = None,
                            timeout: float = 5.0, dry_run: bool = False) -> int:
    """Move photos ffmpeg can't decode in time out of the folder. Returns how many it found."""
    require_tools("ffmpeg")
    if not Path(photos_dir).is_dir():
        raise JobError(f"Photos folder not found: {photos_dir}. Run the photo scan first.")
    broken_dir = broken_dir or photos_dir / "broken_photos"
    photos = top_level_files(photos_dir, IMAGE_EXTS)
    bad = 0
    for i, photo in enumerate(photos, 1):
        rep.check()
        rep.progress(i, len(photos), f"{bad} broken | {photo.name}")
        result = run_tool(["ffmpeg", "-y", "-v", "error", "-i", str(photo), "-frames:v", "1", "-f", "null", "-"],
                          rep, timeout)
        if not result.ok:
            bad += 1
            rep.log(f"  broken: {photo.name}")
            if not dry_run:
                set_aside(photo, broken_dir)
    action = "would move" if dry_run else "moved"
    rep.log(f"Checked {len(photos)} photo(s): {bad} broken, {action} to {broken_dir}.")
    return bad
