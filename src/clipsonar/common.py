"""Helpers shared by every job: file discovery, progress reporting, and running ffmpeg."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mts", ".m2ts",
              ".3gp", ".3g2", ".hevc", ".mpg", ".mpeg", ".flv"}

# Thumbnail and system folders that NAS devices and Windows keep inside media shares.
SKIPPED_DIRS = {"@eadir", "#recycle", "$recycle.bin", "system volume information", ".thumbnails"}

_SCORE_RE = re.compile(r"__s(\d+\.\d+)")
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class JobError(Exception):
    """A problem the user can fix, such as a missing folder. The message is shown as-is."""


class Cancelled(Exception):
    """The user asked the job to stop. Progress so far is kept and a rerun resumes."""


class Reporter:
    """Receives job output. The CLI and the desktop UI each provide a subclass."""

    def log(self, message: str) -> None:
        pass

    def progress(self, done: int, total: int, label: str = "") -> None:
        pass

    def should_stop(self) -> bool:
        return False

    def check(self) -> None:
        if self.should_stop():
            raise Cancelled()


@dataclass(frozen=True)
class PhotoLayout:
    """Where photo results live inside the destination folder."""

    root: Path

    @classmethod
    def under(cls, destination: Path) -> PhotoLayout:
        return cls(Path(destination) / "photos")

    @property
    def hits(self) -> Path:
        return self.root / "hits.csv"

    @property
    def meta(self) -> Path:
        return self.root / "scan_meta.json"

    @property
    def matches(self) -> Path:
        return self.root / "matches"

    @property
    def broken(self) -> Path:
        return self.root / "broken_photos"

    @property
    def montages(self) -> Path:
        return self.root / "montages"


@dataclass(frozen=True)
class VideoLayout:
    """Where video results live inside the destination folder."""

    root: Path

    @classmethod
    def under(cls, destination: Path) -> VideoLayout:
        return cls(Path(destination) / "videos")

    @property
    def hits(self) -> Path:
        return self.root / "hits.csv"

    @property
    def meta(self) -> Path:
        return self.root / "scan_meta.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def log(self) -> Path:
        return self.root / "extract.log"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def broken_sources(self) -> Path:
        return self.root / "broken_videos"

    @property
    def broken_clips(self) -> Path:
        return self.root / "broken_clips"

    @property
    def montages(self) -> Path:
        return self.root / "montages"


def score_from_name(path: Path) -> float:
    """Read the match score from a name like 'IMG_0042__s0.62.jpg'. Returns 0.0 if there is none."""
    match = _SCORE_RE.search(path.stem)
    return float(match.group(1)) if match else 0.0


def unique_target(target: Path) -> Path:
    """Return target, or target with _1, _2, ... appended if the name is taken."""
    if not target.exists():
        return target
    n = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def set_aside(path: Path, folder: Path) -> Path:
    """Move a file into folder without overwriting anything. Files are never deleted."""
    folder.mkdir(parents=True, exist_ok=True)
    target = unique_target(folder / path.name)
    shutil.move(str(path), str(target))
    return target


def format_hms(seconds: float) -> str:
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def write_json(path: Path, data: dict) -> None:
    """Write through a temporary file so a crash can't leave a half-written file behind."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> dict:
    """Return the file's contents, or an empty dict if it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def warn_if_settings_changed(path: Path, settings: dict, rep: Reporter) -> None:
    """Warn when a scan resumes with different settings, then record the current ones."""
    previous = read_json(path)
    if previous and previous != settings:
        changed = sorted(k for k in settings if previous.get(k) != settings[k])
        rep.log(f"WARNING: settings changed since the last scan ({', '.join(changed)}). "
                "Earlier results keep the old settings. Use 'Start over' to redo them.")
    write_json(path, settings)


def iter_files(sources: Iterable[Path], extensions: set[str], rep: Reporter,
               exclude: Iterable[Path] = ()) -> list[Path]:
    """List matching files under every source, skipping excluded folders and system folders."""
    excluded = {_normalized(p) for p in exclude}
    found: set[Path] = set()
    for source in sources:
        for path in _walk(Path(source), extensions, excluded, rep):
            found.add(path)
    return sorted(found)


def _normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _walk(root: Path, extensions: set[str], excluded: set[str], rep: Reporter) -> Iterator[Path]:
    for folder, subdirs, names in os.walk(root):
        rep.check()
        subdirs[:] = [d for d in subdirs
                      if d.lower() not in SKIPPED_DIRS
                      and _normalized(Path(folder) / d) not in excluded]
        for name in names:
            path = Path(folder) / name
            if path.suffix.lower() in extensions and not path.stem.endswith(".part"):
                yield path


def top_level_files(folder: Path, extensions: set[str]) -> list[Path]:
    """List matching files directly in folder. Subfolders such as low_score/ are left alone."""
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in extensions and not p.stem.endswith(".part"))


def valid_sources(sources: Sequence[Path], rep: Reporter) -> list[Path]:
    """Keep the source folders that exist. Raises JobError if none do."""
    usable = [Path(s) for s in sources if Path(s).is_dir()]
    for source in sources:
        if not Path(source).is_dir():
            rep.log(f"WARNING: folder not found, skipping: {source}")
    if not usable:
        raise JobError("None of the folders to search could be found.")
    return usable


def require_tools(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise JobError(f"{' and '.join(missing)} not found. Install FFmpeg with 'winget install ffmpeg', "
                       "then restart Clipsonar.")


@dataclass
class ToolResult:
    ok: bool
    stdout: str = ""
    error: str = ""


def run_tool(cmd: Sequence[str], rep: Reporter, timeout: float | None = None,
             cwd: Path | None = None) -> ToolResult:
    """Run a command, stopping it if the user cancels or the timeout passes."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd, creationflags=_NO_WINDOW)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if rep.should_stop():
                proc.kill()
                proc.communicate()
                raise Cancelled() from None
            if deadline is not None and time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                return ToolResult(False, error=f"timed out after {timeout:.0f}s")
    return ToolResult(proc.returncode == 0, stdout.decode(errors="replace"),
                      stderr.decode(errors="replace")[:400])


def partial_path(target: Path) -> Path:
    """Temporary name for a file being written. File discovery ignores these."""
    return target.with_name(f"{target.stem}.part{target.suffix}")


def run_ffmpeg(args: Sequence[str], output: Path, rep: Reporter, timeout: float | None = None,
               cwd: Path | None = None) -> ToolResult:
    """Run ffmpeg so that output only appears once it is complete."""
    tmp = partial_path(output)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args, str(tmp)]
    try:
        result = run_tool(cmd, rep, timeout, cwd)
        if result.ok and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(output)
            return result
        return ToolResult(False, error=result.error or "ffmpeg produced no output")
    finally:
        tmp.unlink(missing_ok=True)


def write_concat_list(path: Path, files: Sequence[Path], duration: float | None = None) -> None:
    """Write an input list for ffmpeg's concat demuxer."""
    def line(file: Path) -> str:
        return "file '" + str(file.resolve()).replace("'", "'\\''") + "'\n"

    with open(path, "w", encoding="utf-8") as f:
        for file in files:
            f.write(line(file))
            if duration is not None:
                f.write(f"duration {duration}\n")
        if duration is not None:
            # The concat demuxer ignores the last duration unless the final file is repeated.
            f.write(line(files[-1]))
