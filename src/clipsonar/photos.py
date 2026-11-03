"""Scan photo folders for a person and copy the matches. Source photos are only ever read."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from .common import (
    IMAGE_EXTS,
    JobError,
    PhotoLayout,
    Reporter,
    iter_files,
    unique_target,
    valid_sources,
    warn_if_settings_changed,
)
from .faces import DEFAULT_THRESHOLD, FaceMatcher, read_image, reference_fingerprint

FIELDS = ["source_file", "score", "matched", "copied_as"]


@dataclass
class PhotoScanConfig:
    references: Path
    sources: list[Path]
    destination: Path
    threshold: float = DEFAULT_THRESHOLD
    use_gpu: bool = False
    dry_run: bool = False
    restart: bool = False


def scan_photos(cfg: PhotoScanConfig, rep: Reporter, matcher: FaceMatcher | None = None) -> None:
    if not Path(cfg.references).is_dir():
        raise JobError(f"Sample images folder not found: {cfg.references}")
    sources = valid_sources(cfg.sources, rep)
    layout = PhotoLayout.under(cfg.destination)
    layout.root.mkdir(parents=True, exist_ok=True)

    if cfg.restart:
        shutil.rmtree(layout.matches, ignore_errors=True)
        layout.hits.unlink(missing_ok=True)
        layout.meta.unlink(missing_ok=True)
        rep.log("Started over: cleared earlier scan results and matches.")
    layout.matches.mkdir(exist_ok=True)

    warn_if_settings_changed(layout.meta, {"references": reference_fingerprint(Path(cfg.references))}, rep)

    rep.log("Listing photos (this can take a while on a network share)...")
    photos = iter_files(sources, IMAGE_EXTS, rep, exclude=[cfg.destination])
    rep.log(f"Found {len(photos)} photo(s).")

    prior = _read_rows(layout.hits)
    _reconcile(prior, layout, cfg, rep)
    scanned = {row["source_file"] for row in prior}
    remaining = [p for p in photos if str(p) not in scanned]
    done = len(photos) - len(remaining)
    matches = sum(1 for row in prior if row["matched"] == "yes")
    if done:
        rep.log(f"Resuming: {done} already scanned, {len(remaining)} to go.")

    if remaining:
        matcher = matcher or FaceMatcher.load(cfg.references, cfg.use_gpu, rep)
        # Line buffering keeps every result on disk, so a cancelled or crashed scan can resume.
        with open(layout.hits, "a", newline="", encoding="utf-8", buffering=1) as f:
            writer = csv.DictWriter(f, FIELDS)
            if not prior:
                writer.writeheader()
            for i, photo in enumerate(remaining, done + 1):
                rep.check()
                row = _scan_one(photo, matcher, layout, cfg, rep)
                writer.writerow(row)
                matches += row["matched"] == "yes"
                rep.progress(i, len(photos), f"{matches} match(es) | {photo.name}")

    verb = "found (preview only, nothing copied)" if cfg.dry_run else f"in {layout.matches}"
    rep.log(f"Done: {matches} match(es) {verb}. Log: {layout.hits}")


def _scan_one(photo: Path, matcher, layout: PhotoLayout, cfg: PhotoScanConfig, rep: Reporter) -> dict:
    row = {"source_file": str(photo), "score": "", "matched": "no", "copied_as": ""}
    image = read_image(photo)
    if image is None:
        rep.log(f"  could not read {photo}")
        return row
    score = matcher.score(image)
    if score is None:
        return row
    row["score"] = f"{score:.3f}"
    if score >= cfg.threshold:
        row["matched"] = "yes"
        if not cfg.dry_run:
            row["copied_as"] = _copy_match(photo, score, layout)
    return row


def _copy_match(photo: Path, score: float, layout: PhotoLayout) -> str:
    target = unique_target(layout.matches / f"{photo.stem}__s{score:.2f}{photo.suffix}")
    shutil.copy2(photo, target)
    return target.name


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if row.get("source_file")]
    except (OSError, ValueError, csv.Error) as e:
        raise JobError(f"Could not read {path}: {e}. Use 'Start over' to replace it.") from e


def _reconcile(rows: list[dict], layout: PhotoLayout, cfg: PhotoScanConfig, rep: Reporter) -> None:
    """Apply the current threshold to earlier scores, and copy matches a preview left uncopied.

    A row that already has a copied_as name is never copied again, so photos deleted from the
    matches folder during review stay deleted.
    """
    changed = False
    try:
        for row in rows:
            if not row.get("score"):
                continue
            score = float(row["score"])
            matched = "yes" if score >= cfg.threshold else "no"
            if matched != row["matched"]:
                row["matched"] = matched
                changed = True
            if matched == "yes" and not row.get("copied_as") and not cfg.dry_run:
                rep.check()
                source = Path(row["source_file"])
                if source.exists():
                    row["copied_as"] = _copy_match(source, score, layout)
                    changed = True
    finally:
        if changed:
            _write_rows(layout.hits, rows)


def _write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
