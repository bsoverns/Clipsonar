"""Turn matched photos and clips into batched MP4 montages with an M3U playlist.

Batching keeps a long montage resumable: each finished batch is recorded, a failure costs one
batch rather than the whole run, and the playlist plays the batches back to back.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import (
    _NO_WINDOW,
    IMAGE_EXTS,
    VIDEO_EXTS,
    Cancelled,
    JobError,
    Reporter,
    partial_path,
    read_json,
    require_tools,
    run_ffmpeg,
    run_tool,
    score_from_name,
    set_aside,
    top_level_files,
    write_concat_list,
    write_json,
)

ORDER_FILE = "order.json"
STATE_FILE = "batch_state.json"
PLAYLIST_FILE = "playlist.m3u"
BATCH_GLOB = "Montage_batch_*.mp4"


@dataclass
class MontageSettings:
    """Options that define the result. Changing them requires starting the montage over."""

    batch_size: int
    shuffle: bool = True
    min_score: float = 0.0
    max_score: float = 1.0
    limit: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30


@dataclass
class PhotoMontageSettings(MontageSettings):
    batch_size: int = 500
    duration: float = 1.0


@dataclass
class VideoMontageSettings(MontageSettings):
    batch_size: int = 50


def _video_filter(width: int, height: int) -> str:
    return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:-1:-1:color=black,setsar=1")


def _select(items: list[Path], settings: MontageSettings) -> list[Path]:
    selected = [p for p in items if settings.min_score <= score_from_name(p) <= settings.max_score]
    if settings.shuffle:
        random.shuffle(selected)
    return selected[:settings.limit] if settings.limit > 0 else selected


def _load_order(work_dir: Path, available: list[Path], settings: MontageSettings, rep: Reporter) -> list[Path]:
    """Return the saved order, so that resuming keeps batch contents stable, or create one."""
    path = work_dir / ORDER_FILE
    saved = read_json(path)
    if saved:
        if saved.get("settings") != asdict(settings):
            raise JobError("Montage settings changed since this montage was started. "
                           "Use 'Start over' to rebuild it with the new settings.")
        order = [Path(p) for p in saved["items"]]
        known = set(order)
        selected = [p for p in available
                    if settings.min_score <= score_from_name(p) <= settings.max_score]
        new = sum(1 for p in selected if p.resolve() not in known)
        if new and not settings.limit:
            rep.log(f"NOTE: {new} newer file(s) are not in this montage. Use 'Start over' to include them.")
        return order
    order = [p.resolve() for p in _select(available, settings)]
    if not order:
        raise JobError("Nothing to build: no files match the chosen score range.")
    write_json(path, {"items": [str(p) for p in order], "settings": asdict(settings)})
    return order


def _write_playlist(montage_dir: Path) -> None:
    with open(montage_dir / PLAYLIST_FILE, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for batch in sorted(montage_dir.glob(BATCH_GLOB)):
            f.write(f"{batch.name}\n")


def _clear(montage_dir: Path, work_dir: Path, rep: Reporter) -> None:
    for batch in montage_dir.glob(BATCH_GLOB):
        batch.unlink()
    (montage_dir / PLAYLIST_FILE).unlink(missing_ok=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    rep.log("Started over: removed earlier montage batches.")


def _run_batches(order: list[Path], batch_size: int, montage_dir: Path, work_dir: Path, rep: Reporter,
                 encode: Callable[[list[Path], Path], bool]) -> None:
    """Encode each batch not yet finished, recording progress and refreshing the playlist."""
    state_path = work_dir / STATE_FILE
    completed = set(read_json(state_path).get("completed", []))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rep.log(f"{len(order)} file(s) in {len(batches)} batch(es).")
    built = skipped = 0
    failed = []
    for i, batch in enumerate(batches, 1):
        rep.check()
        name = f"Montage_batch_{i:03d}.mp4"
        output = montage_dir / name
        if name in completed and output.exists():
            skipped += 1
            continue
        completed.discard(name)
        output.unlink(missing_ok=True)
        present = [p for p in batch if p.exists()]
        if not present:
            rep.log(f"[{i}/{len(batches)}] skipped: all files in this batch are gone")
            continue
        rep.log(f"[{i}/{len(batches)}] {name} ({len(present)} file(s))")
        if encode(present, output):
            built += 1
            completed.add(name)
            write_json(state_path, {"completed": sorted(completed)})
            _write_playlist(montage_dir)
        else:
            failed.append(name)
    _write_playlist(montage_dir)
    rep.log(f"Done: {built} built, {skipped} already done, {len(failed)} failed. "
            f"Playlist: {montage_dir / PLAYLIST_FILE}")
    if failed:
        rep.log(f"Failed batches: {', '.join(failed)}. Run again to retry them; finished batches are kept.")


# ---- Photos ----------------------------------------------------------------------------------


def build_photo_montage(photos_dir: Path, montage_dir: Path, settings: PhotoMontageSettings, rep: Reporter,
                        restart: bool = False, broken_dir: Path | None = None,
                        stall_timeout: float = 300.0, max_retries: int = 10) -> None:
    """Build a slideshow. If broken_dir is set, photos that hang ffmpeg are moved there."""
    require_tools("ffmpeg")
    if not Path(photos_dir).is_dir():
        raise JobError(f"Photos folder not found: {photos_dir}. Run the photo scan first.")
    work_dir = montage_dir / "batches"
    if restart:
        _clear(montage_dir, work_dir, rep)
    work_dir.mkdir(parents=True, exist_ok=True)
    order = _load_order(work_dir, top_level_files(photos_dir, IMAGE_EXTS), settings, rep)

    def encode(batch: list[Path], output: Path) -> bool:
        for attempt in range(max_retries + 1):
            batch = [p for p in batch if p.exists()]
            if not batch:
                return False
            ok, culprit = _encode_photo_batch(batch, output, settings, work_dir, stall_timeout, rep)
            if ok:
                return True
            if culprit is None:
                return False
            if broken_dir is None or attempt == max_retries:
                rep.log(f"  batch failed, likely photo: {culprit}")
                return False
            target = set_aside(culprit, broken_dir)
            rep.log(f"  set aside {culprit.name} -> {target}, retrying the batch")
        return False

    _run_batches(order, settings.batch_size, montage_dir, work_dir, rep, encode)


def _encode_photo_batch(photos: Sequence[Path], output: Path, settings: PhotoMontageSettings, work_dir: Path,
                        stall_timeout: float, rep: Reporter) -> tuple[bool, Path | None]:
    """Encode one slideshow. Returns (ok, photo being read when it failed).

    Some JPEGs make ffmpeg hang rather than fail, so this watches its progress output and stops it
    when nothing moves. ffmpeg's verbose log names each file as it opens it, which identifies the
    photo responsible.
    """
    list_path = work_dir / "_photos.txt"
    write_concat_list(list_path, photos, settings.duration)
    tmp = partial_path(output)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "verbose", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", _video_filter(settings.width, settings.height), "-r", str(settings.fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-progress", "pipe:1", "-stats_period", "1", "-nostats", str(tmp),
    ]
    lookup = {os.path.normcase(os.path.normpath(p)): p for p in photos}
    total_us = max(1, len(photos) * settings.duration) * 1_000_000
    current: list[Path | None] = [None]
    last_progress = [time.monotonic()]
    stalled = threading.Event()
    cancelled = threading.Event()
    finished = threading.Event()
    tail: list[str] = []

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace", bufsize=1, creationflags=_NO_WINDOW)

    def read_stderr() -> None:
        opening = re.compile(r"Opening '(.+)' for reading")
        for line in proc.stderr:
            match = opening.search(line)
            if match:
                found = lookup.get(os.path.normcase(os.path.normpath(match.group(1))))
                if found is not None:
                    current[0] = found
            tail.append(line.rstrip())
            del tail[:-20]

    def watchdog() -> None:
        while not finished.wait(0.5):
            if rep.should_stop():
                cancelled.set()
            elif time.monotonic() - last_progress[0] > stall_timeout:
                stalled.set()
            else:
                continue
            proc.kill()
            return

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()
    threading.Thread(target=watchdog, daemon=True).start()

    try:
        best_us = 0
        out_time = re.compile(r"out_time_us=(\d+)")
        for line in proc.stdout:
            match = out_time.search(line)
            if match and int(match.group(1)) > best_us:
                best_us = int(match.group(1))
                last_progress[0] = time.monotonic()
                rep.progress(int(min(best_us, total_us) / 1000), int(total_us / 1000), output.name)
        proc.wait()
        finished.set()
        stderr_thread.join(timeout=5)

        if cancelled.is_set():
            raise Cancelled()
        if stalled.is_set() or proc.returncode != 0:
            why = f"no progress for {stall_timeout:.0f}s" if stalled.is_set() else "; ".join(tail[-3:])[:300]
            rep.log(f"  ffmpeg failed: {why}")
            return False, current[0]
        tmp.replace(output)
        return True, None
    finally:
        finished.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        tmp.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)


# ---- Video clips -----------------------------------------------------------------------------


def build_video_montage(clips_dir: Path, montage_dir: Path, settings: VideoMontageSettings, rep: Reporter,
                        restart: bool = False, broken_dir: Path | None = None,
                        clip_timeout: float = 300.0) -> None:
    """Join clips in two stages. If broken_dir is set, clips that fail are moved there.

    Clips come from different cameras, and ffmpeg's concat demuxer needs identical stream formats.
    So each clip is first re-encoded to a common format, which also pins any failure on one clip,
    and the results are then joined without re-encoding.
    """
    require_tools("ffmpeg", "ffprobe")
    if not Path(clips_dir).is_dir():
        raise JobError(f"Clips folder not found: {clips_dir}. Run the video scan first.")
    work_dir = montage_dir / "batches"
    if restart:
        _clear(montage_dir, work_dir, rep)
    norm_dir = work_dir / "norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    order = _load_order(work_dir, top_level_files(clips_dir, VIDEO_EXTS), settings, rep)

    def encode(batch: list[Path], output: Path) -> bool:
        normalized = []
        for i, clip in enumerate(batch, 1):
            rep.check()
            rep.progress(i - 1, len(batch), f"{output.name} | {clip.name}")
            norm = norm_dir / f"{clip.stem}__norm.mp4"
            if norm.exists() and norm.stat().st_size > 0:
                normalized.append(norm)
                continue
            error = _normalize_clip(clip, norm, settings, clip_timeout, rep)
            if error is None:
                normalized.append(norm)
                continue
            rep.log(f"  skipped {clip.name}: {error[:120]}")
            if broken_dir is not None:
                set_aside(clip, broken_dir)
        if not normalized:
            return False
        list_path = work_dir / "_clips.txt"
        write_concat_list(list_path, normalized)
        try:
            result = run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy",
                                 "-movflags", "+faststart"], output, rep, 120 + 2 * len(normalized))
        finally:
            list_path.unlink(missing_ok=True)
        if not result.ok:
            rep.log(f"  join failed: {result.error[:200]}")
            return False
        for norm in normalized:
            norm.unlink(missing_ok=True)
        return True

    _run_batches(order, settings.batch_size, montage_dir, work_dir, rep, encode)


def has_audio(clip: Path, rep: Reporter) -> bool | None:
    """Return whether the clip has audio, or None if ffprobe can't read it."""
    result = run_tool(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                       "-of", "json", str(clip)], rep, timeout=30)
    try:
        data = json.loads(result.stdout)
        if float(data["format"]["duration"]) <= 0:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def _normalize_clip(clip: Path, output: Path, settings: VideoMontageSettings, timeout: float,
                    rep: Reporter) -> str | None:
    """Re-encode a clip to the common format. Returns an error message, or None on success."""
    audio = has_audio(clip, rep)
    if audio is None:
        return "unreadable"
    args = ["-i", str(clip)]
    if audio:
        args += ["-map", "0:v:0", "-map", "0:a:0", "-af", "aresample=async=1:first_pts=0"]
    else:
        # A silent track keeps the audio in sync once clips are joined.
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                 "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    args += [
        "-vf", f"fps={settings.fps}," + _video_filter(settings.width, settings.height),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-video_track_timescale", "90000", "-movflags", "+faststart",
    ]
    result = run_ffmpeg(args, output, rep, timeout)
    return None if result.ok else result.error
