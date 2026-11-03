"""Command-line interface. Each command does the same work as its button in the desktop UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .common import Cancelled, JobError, PhotoLayout, Reporter, VideoLayout
from .faces import DEFAULT_THRESHOLD


class ConsoleReporter(Reporter):
    def __init__(self) -> None:
        self._line_open = False

    def log(self, message: str) -> None:
        if self._line_open:
            print()
            self._line_open = False
        print(message, flush=True)

    def progress(self, done: int, total: int, label: str = "") -> None:
        percent = 100 * done // total if total else 0
        print(f"\r{percent:3d}% {label[:70]:<70}", end="", flush=True)
        self._line_open = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipsonar", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    defaults = argparse.ArgumentDefaultsHelpFormatter

    def add_scan(name: str, help_text: str) -> argparse.ArgumentParser:
        p = commands.add_parser(name, help=help_text, formatter_class=defaults)
        p.add_argument("--samples", required=True, type=Path, help="Folder of sample images of the person.")
        p.add_argument("--sources", required=True, nargs="+", type=Path, help="Folders to search, recursively.")
        p.add_argument("--out", required=True, type=Path, help="Destination folder.")
        p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="Match score needed. Higher is stricter.")
        p.add_argument("--gpu", action="store_true", help="Use an NVIDIA GPU if available.")
        p.add_argument("--dry-run", action="store_true", help="Scan and log only, without creating copies or clips.")
        p.add_argument("--restart", action="store_true", help="Delete earlier results for this scan first.")
        return p

    add_scan("photos", "Find photos of the person and copy them to the destination.")

    videos = add_scan("videos", "Find the person in videos and extract clips to the destination.")
    videos.add_argument("--interval", type=float, default=2.0, help="Seconds between sampled frames.")
    videos.add_argument("--pre", type=float, default=5.0, help="Clip seconds before a match.")
    videos.add_argument("--post", type=float, default=10.0, help="Clip seconds after a match.")
    videos.add_argument("--min-gap", type=float, default=45.0, help="Minimum seconds between clip matches.")
    videos.add_argument("--clip-timeout", type=float, default=180.0, help="Seconds allowed per clip extraction.")
    videos.add_argument("--retry-failed", action="store_true", help="Try videos that failed earlier again.")
    videos.add_argument("--move-broken-sources", action="store_true",
                        help="Move ORIGINAL videos that fail into <out>/videos/broken_videos.")
    videos.add_argument("--exclude", nargs="*", default=[], help="File names to skip.")

    def add_montage(name: str, help_text: str, batch_size: int) -> argparse.ArgumentParser:
        p = commands.add_parser(name, help=help_text, formatter_class=defaults)
        p.add_argument("--out", required=True, type=Path, help="Destination folder used for the scan.")
        p.add_argument("--batch-size", type=int, default=batch_size, help="Files per MP4.")
        p.add_argument("--no-shuffle", action="store_true", help="Keep file name order.")
        p.add_argument("--min-score", type=float, default=0.0)
        p.add_argument("--max-score", type=float, default=1.0)
        p.add_argument("--limit", type=int, default=0, help="Maximum number of files (0 = all).")
        p.add_argument("--restart", action="store_true", help="Delete earlier batches first.")
        p.add_argument("--set-aside-broken", action="store_true",
                       help="Move files that fail out of the destination's matches or clips folder.")
        return p

    photo_montage = add_montage("photo-montage", "Build a slideshow and playlist from matched photos.", 500)
    photo_montage.add_argument("--duration", type=float, default=1.0, help="Seconds per photo.")
    add_montage("video-montage", "Build a montage and playlist from extracted clips.", 50)

    review = commands.add_parser("review", help="Build a video that labels each matched photo, for checking.",
                                 formatter_class=defaults)
    review.add_argument("--out", required=True, type=Path, help="Destination folder used for the scan.")
    review.add_argument("--duration", type=float, default=0.5, help="Seconds per photo.")
    review.add_argument("--min-score", type=float, default=0.0)
    review.add_argument("--max-score", type=float, default=1.0)
    review.add_argument("--shuffle", action="store_true")

    low = commands.add_parser("filter", help="Move low-scoring matched photos into a subfolder.",
                              formatter_class=defaults)
    low.add_argument("--out", required=True, type=Path, help="Destination folder used for the scan.")
    low.add_argument("--min-score", type=float, default=0.50, help="Photos below this score are moved.")
    low.add_argument("--dry-run", action="store_true")

    broken = commands.add_parser("check-photos", help="Set aside matched photos that ffmpeg can't decode.",
                                 formatter_class=defaults)
    broken.add_argument("--out", required=True, type=Path, help="Destination folder used for the scan.")
    broken.add_argument("--timeout", type=float, default=5.0, help="Seconds allowed per photo.")
    broken.add_argument("--dry-run", action="store_true")
    return parser


def _run(args: argparse.Namespace, rep: Reporter) -> None:
    # Imported here so that --help stays fast.
    from .montage import PhotoMontageSettings, VideoMontageSettings, build_photo_montage, build_video_montage
    from .photos import PhotoScanConfig, scan_photos
    from .review import build_review_video, filter_low_scores, set_aside_broken_photos
    from .videos import VideoScanConfig, scan_videos

    command = args.command
    if command == "photos":
        scan_photos(PhotoScanConfig(args.samples, args.sources, args.out, args.threshold, args.gpu,
                                    args.dry_run, args.restart), rep)
    elif command == "videos":
        scan_videos(VideoScanConfig(args.samples, args.sources, args.out, args.threshold, args.gpu, args.interval,
                                    args.pre, args.post, args.min_gap, args.clip_timeout, args.dry_run,
                                    args.restart, args.retry_failed, args.move_broken_sources, args.exclude), rep)
    elif command == "photo-montage":
        layout = PhotoLayout.under(args.out)
        settings = PhotoMontageSettings(args.batch_size, not args.no_shuffle, args.min_score, args.max_score,
                                        args.limit, duration=args.duration)
        build_photo_montage(layout.matches, layout.montages, settings, rep, args.restart,
                            layout.broken if args.set_aside_broken else None)
    elif command == "video-montage":
        layout = VideoLayout.under(args.out)
        settings = VideoMontageSettings(args.batch_size, not args.no_shuffle, args.min_score, args.max_score,
                                        args.limit)
        build_video_montage(layout.clips, layout.montages, settings, rep, args.restart,
                            layout.broken_clips if args.set_aside_broken else None)
    elif command == "review":
        layout = PhotoLayout.under(args.out)
        build_review_video(layout.matches, layout.montages / "PhotoReview.mp4", rep, layout.hits, args.duration,
                           args.min_score, args.max_score, args.shuffle)
    elif command == "filter":
        filter_low_scores(PhotoLayout.under(args.out).matches, args.min_score, rep, dry_run=args.dry_run)
    elif command == "check-photos":
        layout = PhotoLayout.under(args.out)
        set_aside_broken_photos(layout.matches, rep, layout.broken, args.timeout, args.dry_run)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rep = ConsoleReporter()
    try:
        _run(args, rep)
    except JobError as e:
        rep.log(f"ERROR: {e}")
        return 1
    except (Cancelled, KeyboardInterrupt):
        rep.log("Stopped. Run the same command again to resume.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
