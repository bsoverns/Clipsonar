import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from clipsonar.gui import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_settings_survive_a_restart(app, settings):
    window = MainWindow(settings)
    window._samples.edit.setText("C:/samples")
    window._destination.edit.setText("C:/dest")
    window._sources.addItems(["C:/one", "C:/two"])
    window._threshold.setValue(0.55)
    window._photo_shuffle.setChecked(False)
    window._save_settings()

    reopened = MainWindow(settings)
    assert reopened._samples.edit.text() == "C:/samples"
    assert reopened._destination.edit.text() == "C:/dest"
    assert reopened._source_paths_text() == ["C:/one", "C:/two"]
    assert reopened._threshold.value() == 0.55
    assert reopened._photo_shuffle.isChecked() is False


def test_moving_originals_is_never_a_saved_default(app, settings):
    window = MainWindow(settings)
    assert window._move_broken.isChecked() is False
    assert window._move_broken not in window._saved_controls.values()


def test_job_runs_and_reports_back(app, settings, tmp_path):
    window = MainWindow(settings)
    results = []
    window._log.textChanged.connect(lambda: results.append(window._log.toPlainText()))

    def job(rep):
        rep.log("hello from the job")
        rep.progress(1, 2, "halfway")

    window._start("Test job", job)
    assert window._cancel.isEnabled()
    thread = window._thread
    while not thread.isFinished():
        app.processEvents()
    app.processEvents()
    app.processEvents()
    assert "hello from the job" in window._log.toPlainText()
    assert window._status.text() == "Done"
    assert all(button.isEnabled() for button in window._job_buttons)
