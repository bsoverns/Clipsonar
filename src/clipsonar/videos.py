"""Scan video folders for a person and extract a short clip around each appearance."""

from __future__ import annotations

import csv
import gc
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .common import (
    VIDEO_EXTS,
    JobError,
    Reporter,
    VideoLayout,
    format_hms,
    iter_files,
    read_json,
    require_tools,
    run_ffmpeg,
    set_aside,
    valid_sources,
    warn_if_settings_changed,
    write_json,
)
from .faces import DEFAULT_THRESHOLD, FaceMatcher, reference_fingerprint

FIELDS = ["video", "timestamp_sec", "timestamp_hms", "score", "clip_file"]


@dataclass
class VideoScanConfig:
    references: Path
    sources: list[Path]
    destination: Path
    threshold: float = DEFAULT_THRESHOLD
    use_gpu: bool = False
    interval: float = 2.0
    pre: float = 5.0
    post: float = 10.0
    min_gap: float = 45.0
    clip_timeout: float = 180.0
    dry_run: bool = False
    restart: bool = False
    retry_failed: bool = False
    # Off by default: a failure can be caused by ffmpeg or the network rather than the file,
    # and this moves the user's original out of its folder.
    move_broken_sources: bool = False
    excluded_names: list[str] = field(default_factory=list)


def dedupe_hits(hits: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Merge hits closer than min_gap into one, keeping the first time and the best score."""
    kept: list[tuple[float, float]] = []
    for ts, score in sorted(hits):
        if not kept or ts - kept[-1][0] >= min_gap:
            kept.append((ts, score))
        elif score > kept[-1][1]:
            kept[-1] = (kept[-1][0], score)
    return kept


def clip_name(video: Path, ts: float, score: float) -> str:
    # The path hash keeps clips apart when two folders both hold an IMG_0001.mp4.
    tag = hashlib.sha1(str(video).encode("utf-8", "replace")).hexdigest()[:6]
    return f"{video.stem}__{tag}__{format_hms(ts)}__s{score:.2f}.mp4"


def scan_video(matcher, video: Path, cfg: VideoScanConfig, rep: Reporter) -> list[tuple[float, float]] | None:
    """Sample frames and return (seconds, score) hits, or None if the video can't be opened."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        step = max(1, round(fps * cfg.interval))
        hits = []
        for frame_index in range(0, total, step):
            rep.check()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            score = matcher.score(frame)
            if score is not None and score >= cfg.threshold:
                hits.append((frame_index / fps, score))
        return hits
    finally:
        cap.release()


def extract_clip(video: Path, ts: float, output: Path, cfg: VideoScanConfig, rep: Reporter):
    start = max(0.0, ts - cfg.pre)
    args = [
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{cfg.pre + cfg.post:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
    ]
    return run_ffmpeg(args, output, rep, cfg.clip_timeout)


def scan_videos(cfg: VideoScanConfig, rep: Reporter, matcher: FaceMatcher | None = None) -> None:
    if not Path(cfg.references).is_dir():
        raise JobError(f"Sample images folder not found: {cfg.references}")
    if not cfg.dry_run:
        require_tools("ffmpeg")
    sources = valid_sources(cfg.sources, rep)
    layout = VideoLayout.under(cfg.destination)
    layout.root.mkdir(parents=True, exist_ok=True)

    if cfg.restart:
        shutil.rmtree(layout.clips, ignore_errors=True)
        for path in (layout.hits, layout.state, layout.meta):
            path.unlink(missing_ok=True)
        rep.log("Started over: cleared earlier scan results and clips.")
    layout.clips.mkdir(exist_ok=True)

    # Clips must not overlap, so the gap between hits is at least one clip long.
    min_gap = max(cfg.min_gap, cfg.pre + cfg.post)
    warn_if_settings_changed(layout.meta, {
        "references": reference_fingerprint(Path(cfg.references)), "threshold": cfg.threshold,
        "interval": cfg.interval, "pre": cfg.pre, "post": cfg.post, "min_gap": min_gap,
    }, rep)

    rep.log("Listing videos (this can take a while on a network share)...")
    excluded = {name.lower() for name in cfg.excluded_names}
    videos = [v for v in iter_files(sources, VIDEO_EXTS, rep, exclude=[cfg.destination])
              if v.name.lower() not in excluded]
    rep.log(f"Found {len(videos)} video(s).")

    state = read_json(layout.state)
    scanned, failed = set(state.get("scanned", [])), set(state.get("failed", []))
    if cfg.retry_failed:
        scanned -= failed
        failed.clear()

    def save_state() -> None:
        write_json(layout.state, {"scanned": sorted(scanned), "failed": sorted(failed)})

    known_hits = _read_hits(layout.hits)
    write_header = not layout.hits.exists()
    pending = [v for v in videos if str(v) not in scanned]
    if pending:
        matcher = matcher or FaceMatcher.load(cfg.references, cfg.use_gpu, rep)
    new_hits = clips_made = 0

    with open(layout.hits, "a", newline="", encoding="utf-8", buffering=1) as hits_file, \
            open(layout.log, "a", encoding="utf-8", buffering=1) as log:
        writer = csv.DictWriter(hits_file, FIELDS)
        if write_header:
            writer.writeheader()
        log.write(f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        for i, video in enumerate(videos, 1):
            rep.check()
            key = str(video)
            rep.progress(i - 1, len(videos), f"{new_hits} new match(es), {clips_made} clip(s) | {video.name}")
            if key in failed:
                continue

            if key not in scanned:
                hits = scan_video(matcher, video, cfg, rep)
                if hits is None:
                    log.write(f"could not open {video}\n")
                    rep.log(f"  could not open {video}")
                    _mark_failed(video, failed, scanned, layout, cfg, rep, log)
                    save_state()
                    continue
                hits = dedupe_hits(hits, min_gap)
                new_hits += len(hits)
                known_hits[key] = {}
                for ts, score in hits:
                    name = clip_name(video, ts, score)
                    known_hits[key][name] = ts
                    writer.writerow({"video": key, "timestamp_sec": f"{ts:.2f}", "timestamp_hms": format_hms(ts),
                                     "score": f"{score:.3f}", "clip_file": name})

            if not cfg.dry_run:
                made, attempts = _extract_missing(video, known_hits.get(key, {}), layout, cfg, rep, log)
                clips_made += made
                if attempts and made == 0:
                    log.write(f"all {attempts} clip(s) failed for {video}\n")
                    rep.log(f"  all clips failed for {video}")
                    _mark_failed(video, failed, scanned, layout, cfg, rep, log)

            scanned.add(key)
            save_state()
            gc.collect()

    rep.progress(len(videos), len(videos), "")
    rep.log(f"Done: {new_hits} new match(es), {clips_made} clip(s) made, {len(failed)} video(s) failed. "
            f"Clips: {layout.clips}")
    if failed:
        rep.log(f"Failed videos are listed in {layout.state}. Use 'Retry failed videos' to try them again.")


def _extract_missing(video: Path, hits: dict[str, float], layout: VideoLayout, cfg: VideoScanConfig,
                     rep: Reporter, log) -> tuple[int, int]:
    made = attempts = 0
    for name, ts in hits.items():
        output = layout.clips / name
        if output.exists() and output.stat().st_size > 0:
            continue
        attempts += 1
        result = extract_clip(video, ts, output, cfg, rep)
        if result.ok:
            made += 1
            log.write(f"ok {name}\n")
        else:
            log.write(f"FAILED {name}: {result.error}\n")
            rep.log(f"  clip failed: {name}: {result.error[:120]}")
    return made, attempts


def _mark_failed(video: Path, failed: set[str], scanned: set[str], layout: VideoLayout, cfg: VideoScanConfig,
                 rep: Reporter, log) -> None:
    failed.add(str(video))
    scanned.add(str(video))
    if cfg.move_broken_sources:
        try:
            target = set_aside(video, layout.broken_sources)
            log.write(f"moved {video} to {target}\n")
            rep.log(f"  moved to {target}")
        except OSError as e:
            log.write(f"could not move {video}: {e}\n")


def _read_hits(path: Path) -> dict[str, dict[str, float]]:
    """Return {video path: {clip name: seconds}}. Clip names also drop rows repeated by a crash."""
    hits: dict[str, dict[str, float]] = {}
    if not path.exists():
        return hits
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hits.setdefault(row["video"], {})[row["clip_file"]] = float(row["timestamp_sec"])
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as e:
        raise JobError(f"Could not read {path}: {e}. Use 'Start over' to replace it.") from e
    return hits
