"""Fail if personal media or scan results have been committed. Run by CI on every push."""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic", ".raw",
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg", ".flv",
    ".m3u", ".csv", ".log", ".db", ".onnx",
}
BLOCKED_NAMES = {"state.json", "scan_meta.json", "order.json", "batch_state.json", "settings.local.json"}
ALLOWED_FILES = {"docs/screenshot.png"}  # A UI screenshot with placeholder paths and no people.


def main() -> int:
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    problems = []
    for name in tracked:
        path = PurePosixPath(name)
        if name not in ALLOWED_FILES and (path.suffix.lower() in BLOCKED_EXTENSIONS or path.name in BLOCKED_NAMES):
            problems.append(name)
    for name in problems:
        print(f"Not allowed in this repository: {name}")
    print(f"Checked {len(tracked)} tracked files: {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
