"""Desktop UI: choose the folders, set the options, and run each job with live progress."""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .common import Cancelled, JobError, PhotoLayout, Reporter, VideoLayout
from .faces import DEFAULT_THRESHOLD

ABOUT_TEXT = (
    f"<b>Clipsonar {__version__}</b><br>Finds a person across photo and video folders. "
    "Everything runs on this computer, and nothing is uploaded.<br><br>"
    "Clipsonar's code is under the MIT License.<br><br>"
    "<b>The face model is not.</b> The InsightFace <i>buffalo_l</i> model, downloaded on first use, is licensed "
    "for non-commercial research only. Contact InsightFace for any other use.<br><br>"
    "Only use Clipsonar on media you have the right to process. Face data is regulated in some places. "
    "See THIRD_PARTY_NOTICES.md for all licenses."
)


class QtReporter(QObject, Reporter):
    logged = Signal(str)
    progressed = Signal(int, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._stop = threading.Event()

    def log(self, message: str) -> None:
        self.logged.emit(message)

    def progress(self, done: int, total: int, label: str = "") -> None:
        self.progressed.emit(done, total, label)

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()


class JobThread(QThread):
    """Runs one job away from the UI thread. Emits (status, message) when it ends."""

    ended = Signal(str, str)

    def __init__(self, job: Callable[[Reporter], None], reporter: QtReporter) -> None:
        super().__init__()
        self._job = job
        self._reporter = reporter

    def run(self) -> None:
        try:
            self._job(self._reporter)
            self.ended.emit("done", "")
        except Cancelled:
            self.ended.emit("stopped", "")
        except JobError as e:
            self.ended.emit("error", str(e))
        except Exception:
            self.ended.emit("error", traceback.format_exc())


class FolderField(QWidget):
    """A path box with a Browse button."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self.edit = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit)
        row.addWidget(browse)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self._title, self.edit.text())
        if folder:
            self.edit.setText(os.path.normpath(folder))

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None


def _spin(value: int, low: int, high: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    return box


def _dspin(value: float, low: float, high: float, step: float, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(2)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.setWindowTitle("Clipsonar")
        self.resize(900, 860)
        self._settings = settings
        self._thread: JobThread | None = None
        self._reporter: QtReporter | None = None
        self._job_buttons: list[QPushButton] = []
        self._saved_controls: dict[str, QWidget] = {}

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_folders())
        layout.addWidget(self._build_matching())
        tabs = QTabWidget()
        tabs.addTab(self._build_photos_tab(), "Photos")
        tabs.addTab(self._build_videos_tab(), "Videos")
        layout.addWidget(tabs)
        layout.addLayout(self._build_status())
        self._log = QPlainTextEdit(readOnly=True)
        self._log.setMaximumBlockCount(5000)
        self._log.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log, stretch=1)
        self.setCentralWidget(root)
        self._load_settings()

    # ---- Layout ------------------------------------------------------------------------------

    def _build_folders(self) -> QGroupBox:
        group = QGroupBox("Folders")
        form = QFormLayout(group)
        self._samples = FolderField("Folder of sample images")
        self._samples.edit.setPlaceholderText("Clear photos of the person to find")
        form.addRow("Sample images:", self._samples)

        self._sources = QListWidget()
        self._sources.setMaximumHeight(72)
        add = QPushButton("Add...")
        add.clicked.connect(self._add_source)
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: self._sources.takeItem(self._sources.currentRow()))
        buttons = QVBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        row = QHBoxLayout()
        row.addWidget(self._sources)
        row.addLayout(buttons)
        form.addRow("Search in:", row)

        self._destination = FolderField("Destination folder")
        self._destination.edit.setPlaceholderText("Copies, clips, montages and playlists go here")
        form.addRow("Destination:", self._destination)
        return group

    def _build_matching(self) -> QGroupBox:
        group = QGroupBox("Matching")
        row = QHBoxLayout(group)
        self._threshold = self._saved("threshold", _dspin(DEFAULT_THRESHOLD, 0.1, 0.95, 0.01))
        self._threshold.setToolTip("Lower finds more but makes more mistakes. 0.40 is a good start.")
        self._gpu = self._saved("gpu", QCheckBox("Use NVIDIA GPU if available"))
        row.addWidget(QLabel("Match score needed:"))
        row.addWidget(self._threshold)
        row.addSpacing(16)
        row.addWidget(self._gpu)
        row.addStretch()
        about = QPushButton("About and licenses")
        about.clicked.connect(lambda: QMessageBox.about(self, "About Clipsonar", ABOUT_TEXT))
        row.addWidget(about)
        return group

    def _build_photos_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scan = QGroupBox("1. Find photos")
        scan_row = QHBoxLayout(scan)
        self._photo_preview = self._saved("photo_preview", QCheckBox("Preview only (don't copy)"))
        self._photo_restart = QCheckBox("Start over")
        scan_row.addWidget(self._photo_preview)
        scan_row.addWidget(self._photo_restart)
        scan_row.addStretch()
        scan_row.addWidget(self._job_button("Find photos", self._run_photo_scan))
        layout.addWidget(scan)

        review = QGroupBox("2. Review matches (optional)")
        review_row = QHBoxLayout(review)
        self._filter_score = self._saved("filter_score", _dspin(0.50, 0.1, 0.95, 0.01))
        review_row.addWidget(self._job_button("Build review video", self._run_review))
        review_row.addWidget(self._job_button("Set aside broken photos", self._run_check_photos))
        review_row.addStretch()
        review_row.addWidget(QLabel("Move matches below:"))
        review_row.addWidget(self._filter_score)
        review_row.addWidget(self._job_button("Move", self._run_filter))
        layout.addWidget(review)

        montage = QGroupBox("3. Build slideshow and playlist")
        grid = QGridLayout(montage)
        self._photo_seconds = self._saved("photo_seconds", _dspin(1.0, 0.1, 30.0, 0.25, " s"))
        self._photo_batch = self._saved("photo_batch", _spin(500, 1, 100000))
        self._photo_min = self._saved("photo_min", _dspin(0.0, 0.0, 1.0, 0.01))
        self._photo_shuffle = self._saved("photo_shuffle", QCheckBox("Shuffle"))
        self._photo_shuffle.setChecked(True)
        self._photo_montage_restart = QCheckBox("Start over")
        grid.addWidget(QLabel("Seconds per photo:"), 0, 0)
        grid.addWidget(self._photo_seconds, 0, 1)
        grid.addWidget(QLabel("Photos per file:"), 0, 2)
        grid.addWidget(self._photo_batch, 0, 3)
        grid.addWidget(QLabel("Lowest score:"), 0, 4)
        grid.addWidget(self._photo_min, 0, 5)
        grid.addWidget(self._photo_shuffle, 1, 0)
        grid.addWidget(self._photo_montage_restart, 1, 1)
        grid.addWidget(self._job_button("Build slideshow", self._run_photo_montage), 1, 5)
        layout.addWidget(montage)
        layout.addStretch()
        return tab

    def _build_videos_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scan = QGroupBox("1. Find clips")
        grid = QGridLayout(scan)
        self._interval = self._saved("interval", _dspin(2.0, 0.1, 60.0, 0.5, " s"))
        self._pre = self._saved("pre", _dspin(5.0, 0.0, 600.0, 1.0, " s"))
        self._post = self._saved("post", _dspin(10.0, 0.5, 600.0, 1.0, " s"))
        self._gap = self._saved("gap", _dspin(45.0, 0.0, 3600.0, 5.0, " s"))
        self._video_preview = self._saved("video_preview", QCheckBox("Preview only (no clips)"))
        self._video_restart = QCheckBox("Start over")
        self._retry_failed = QCheckBox("Retry failed videos")
        self._move_broken = QCheckBox("Move failed ORIGINAL videos out of their folders")
        self._move_broken.setToolTip("Off by default. A failure can come from ffmpeg or the network, "
                                     "not the file itself.")
        timing = [("Check a frame every:", self._interval), ("Clip starts before:", self._pre),
                  ("Clip ends after:", self._post), ("Gap between matches:", self._gap)]
        for column, (label, box) in enumerate(timing):
            grid.addWidget(QLabel(label), 0, column * 2)
            grid.addWidget(box, 0, column * 2 + 1)
        grid.addWidget(self._video_preview, 1, 0, 1, 2)
        grid.addWidget(self._video_restart, 1, 2)
        grid.addWidget(self._retry_failed, 1, 3, 1, 2)
        grid.addWidget(self._move_broken, 2, 0, 1, 5)
        grid.addWidget(self._job_button("Find clips", self._run_video_scan), 2, 7)
        layout.addWidget(scan)

        montage = QGroupBox("2. Build montage and playlist")
        row = QHBoxLayout(montage)
        self._video_batch = self._saved("video_batch", _spin(50, 1, 100000))
        self._video_min = self._saved("video_min", _dspin(0.0, 0.0, 1.0, 0.01))
        self._video_shuffle = self._saved("video_shuffle", QCheckBox("Shuffle"))
        self._video_shuffle.setChecked(True)
        self._video_montage_restart = QCheckBox("Start over")
        row.addWidget(QLabel("Clips per file:"))
        row.addWidget(self._video_batch)
        row.addWidget(QLabel("Lowest score:"))
        row.addWidget(self._video_min)
        row.addWidget(self._video_shuffle)
        row.addWidget(self._video_montage_restart)
        row.addStretch()
        row.addWidget(self._job_button("Build montage", self._run_video_montage))
        layout.addWidget(montage)
        layout.addStretch()
        return tab

    def _build_status(self) -> QVBoxLayout:
        self._progress = QProgressBar()
        self._status = QLabel("Ready")
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._cancel = QPushButton("Stop")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self._stop_job)
        open_destination = QPushButton("Open destination")
        open_destination.clicked.connect(self._open_destination)
        top = QHBoxLayout()
        top.addWidget(self._progress, stretch=1)
        top.addWidget(self._cancel)
        top.addWidget(open_destination)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self._status)
        return layout

    def _job_button(self, text: str, handler: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(handler)
        self._job_buttons.append(button)
        return button

    def _saved(self, key: str, widget):
        self._saved_controls[key] = widget
        return widget

    # ---- Settings ----------------------------------------------------------------------------

    def _load_settings(self) -> None:
        s = self._settings
        self._samples.edit.setText(s.value("samples", "", str))
        self._destination.edit.setText(s.value("destination", "", str))
        self._sources.addItems([p for p in s.value("sources", "", str).split("|") if p])
        for key, widget in self._saved_controls.items():
            if isinstance(widget, QCheckBox):
                widget.setChecked(s.value(key, widget.isChecked(), bool))
            elif isinstance(widget, QSpinBox):
                widget.setValue(s.value(key, widget.value(), int))
            else:
                widget.setValue(s.value(key, widget.value(), float))

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("samples", self._samples.edit.text())
        s.setValue("destination", self._destination.edit.text())
        s.setValue("sources", "|".join(self._source_paths_text()))
        for key, widget in self._saved_controls.items():
            s.setValue(key, widget.isChecked() if isinstance(widget, QCheckBox) else widget.value())

    def _source_paths_text(self) -> list[str]:
        return [self._sources.item(i).text() for i in range(self._sources.count())]

    # ---- Inputs ------------------------------------------------------------------------------

    def _add_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder to search")
        if folder and os.path.normpath(folder) not in self._source_paths_text():
            self._sources.addItem(os.path.normpath(folder))

    def _require_destination(self) -> Path | None:
        destination = self._destination.path()
        if destination is None:
            QMessageBox.warning(self, "Clipsonar", "Choose a destination folder first.")
        return destination

    def _scan_inputs(self) -> tuple[Path, list[Path], Path] | None:
        samples = self._samples.path()
        sources = [Path(p) for p in self._source_paths_text()]
        if samples is None:
            QMessageBox.warning(self, "Clipsonar", "Choose a folder of sample images first.")
            return None
        if not sources:
            QMessageBox.warning(self, "Clipsonar", "Add at least one folder to search.")
            return None
        destination = self._require_destination()
        return (samples, sources, destination) if destination else None

    def _confirm_restart(self, what: str, folder: Path) -> bool:
        answer = QMessageBox.question(self, "Start over?", f"This deletes {what} in:\n{folder}\n\nContinue?")
        return answer == QMessageBox.StandardButton.Yes

    # ---- Jobs --------------------------------------------------------------------------------

    def _run_photo_scan(self) -> None:
        from .photos import PhotoScanConfig, scan_photos

        inputs = self._scan_inputs()
        if inputs is None:
            return
        samples, sources, destination = inputs
        restart = self._photo_restart.isChecked()
        if restart and not self._confirm_restart("earlier scan results and copied matches",
                                                 PhotoLayout.under(destination).root):
            return
        cfg = PhotoScanConfig(samples, sources, destination, self._threshold.value(), self._gpu.isChecked(),
                              self._photo_preview.isChecked(), restart)
        self._photo_restart.setChecked(False)
        self._start("Finding photos", lambda rep: scan_photos(cfg, rep))

    def _run_video_scan(self) -> None:
        from .videos import VideoScanConfig, scan_videos

        inputs = self._scan_inputs()
        if inputs is None:
            return
        samples, sources, destination = inputs
        restart = self._video_restart.isChecked()
        if restart and not self._confirm_restart("earlier scan results and extracted clips",
                                                 VideoLayout.under(destination).root):
            return
        if self._move_broken.isChecked():
            answer = QMessageBox.question(self, "Move original videos?",
                                          "Videos that fail will be moved out of their original folders.\n\nContinue?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        cfg = VideoScanConfig(samples, sources, destination, self._threshold.value(), self._gpu.isChecked(),
                              self._interval.value(), self._pre.value(), self._post.value(), self._gap.value(),
                              dry_run=self._video_preview.isChecked(), restart=restart,
                              retry_failed=self._retry_failed.isChecked(),
                              move_broken_sources=self._move_broken.isChecked())
        self._video_restart.setChecked(False)
        self._retry_failed.setChecked(False)
        self._start("Finding clips", lambda rep: scan_videos(cfg, rep))

    def _run_photo_montage(self) -> None:
        from .montage import PhotoMontageSettings, build_photo_montage

        destination = self._require_destination()
        if destination is None:
            return
        layout = PhotoLayout.under(destination)
        restart = self._photo_montage_restart.isChecked()
        if restart and not self._confirm_restart("earlier slideshow files", layout.montages):
            return
        settings = PhotoMontageSettings(self._photo_batch.value(), self._photo_shuffle.isChecked(),
                                        self._photo_min.value(), duration=self._photo_seconds.value())
        self._photo_montage_restart.setChecked(False)
        # This job only reads the matches folder, which holds copies, so broken photos can be set aside.
        self._start("Building slideshow", lambda rep: build_photo_montage(
            layout.matches, layout.montages, settings, rep, restart, layout.broken))

    def _run_video_montage(self) -> None:
        from .montage import VideoMontageSettings, build_video_montage

        destination = self._require_destination()
        if destination is None:
            return
        layout = VideoLayout.under(destination)
        restart = self._video_montage_restart.isChecked()
        if restart and not self._confirm_restart("earlier montage files", layout.montages):
            return
        settings = VideoMontageSettings(self._video_batch.value(), self._video_shuffle.isChecked(),
                                        self._video_min.value())
        self._video_montage_restart.setChecked(False)
        self._start("Building montage", lambda rep: build_video_montage(
            layout.clips, layout.montages, settings, rep, restart, layout.broken_clips))

    def _run_review(self) -> None:
        from .review import build_review_video

        destination = self._require_destination()
        if destination is not None:
            layout = PhotoLayout.under(destination)
            self._start("Building review video", lambda rep: build_review_video(
                layout.matches, layout.montages / "PhotoReview.mp4", rep, layout.hits))

    def _run_check_photos(self) -> None:
        from .review import set_aside_broken_photos

        destination = self._require_destination()
        if destination is not None:
            layout = PhotoLayout.under(destination)
            self._start("Checking photos", lambda rep: set_aside_broken_photos(layout.matches, rep, layout.broken))

    def _run_filter(self) -> None:
        from .review import filter_low_scores

        destination = self._require_destination()
        if destination is not None:
            score = self._filter_score.value()
            self._start("Moving low scores", lambda rep: filter_low_scores(
                PhotoLayout.under(destination).matches, score, rep))

    def _start(self, title: str, job: Callable[[Reporter], None]) -> None:
        self._save_settings()
        self._reporter = QtReporter()
        self._reporter.logged.connect(self._log.appendPlainText)
        self._reporter.progressed.connect(self._on_progress)
        self._thread = JobThread(job, self._reporter)
        self._thread.ended.connect(self._on_ended)
        self._set_running(True)
        self._progress.setRange(0, 0)
        self._status.setText(f"{title}...")
        self._log.appendPlainText(f"\n--- {title} ---")
        self._thread.start()

    def _stop_job(self) -> None:
        if self._reporter is not None:
            self._reporter.stop()
            self._cancel.setEnabled(False)
            self._status.setText("Stopping...")

    def _set_running(self, running: bool) -> None:
        for button in self._job_buttons:
            button.setEnabled(not running)
        self._cancel.setEnabled(running)

    def _on_progress(self, done: int, total: int, label: str) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)
        if label and self._cancel.isEnabled():
            self._status.setText(label)

    def _on_ended(self, status: str, message: str) -> None:
        self._set_running(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(100 if status == "done" else 0)
        self._thread.wait()
        if status == "error":
            self._status.setText("Failed")
            self._log.appendPlainText(f"ERROR: {message}")
            QMessageBox.critical(self, "Clipsonar", message)
        elif status == "stopped":
            self._status.setText("Stopped. Run the same job again to resume.")
            self._log.appendPlainText("Stopped. Run the same job again to resume.")
        else:
            self._status.setText("Done")

    def _open_destination(self) -> None:
        destination = self._require_destination()
        if destination is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            answer = QMessageBox.question(self, "Clipsonar", "A job is running. Stop it and quit? It can resume later.")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_job()
            self._thread.wait()
        self._save_settings()
        event.accept()


def _self_test(samples: str | None) -> None:
    """Confirm that a packaged build can load its dependencies, and optionally match faces."""
    import cv2  # noqa: F401
    import insightface.app  # noqa: F401
    import onnxruntime  # noqa: F401

    if samples:
        from .faces import FaceMatcher

        FaceMatcher.load(Path(samples), False, Reporter())


def main() -> int:
    multiprocessing.freeze_support()
    # A windowed build has no console, and libraries that write progress bars expect one.
    for stream in ("stdout", "stderr"):
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w"))
    if "--self-test" in sys.argv:
        # Report through the exit code: an uncaught error in a windowed build opens a dialog, which would hang CI.
        try:
            _self_test(sys.argv[2] if len(sys.argv) > 2 else None)
        except Exception:
            return 1
        return 0
    app = QApplication(sys.argv)
    window = MainWindow(QSettings("Clipsonar", "Clipsonar"))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
