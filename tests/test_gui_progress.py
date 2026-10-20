"""GUI state updates from the structured RVC progress stream."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.gui.main import StageWorker, Studio
from app.core.project import Project


class GuiProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_progress_eta_warning_and_stop_button(self):
        window = Studio()
        try:
            payload = {"total": 14, "current": 5, "completed": 4,
                       "percent": 28.6, "elapsed": 154, "eta": 385,
                       "gpu": "cuda:0 (GPU 0)", "status": "RVC 음색 변환 중",
                       "warning": "[WARNING] RVC 변환 지연 감지"}
            window._show_progress("RVC_PROGRESS " + json.dumps(payload, ensure_ascii=False))
            self.assertEqual(window.convert_progress.value(), 286)
            self.assertIn("5/14", window.convert_status.text())
            self.assertIn("06:25", window.convert_status.text())
            self.assertIn("[WARNING]", window.log.toPlainText())
            worker = StageWorker(None, "convert")
            window.worker = worker
            window._active_stage = "convert"
            window.cancel_work()
            self.assertTrue(worker.cancel_event.is_set())
            self.assertFalse(window.cancel_button.isEnabled())
        finally:
            window._active_stage = None
            window.close()

    def test_full_workflow_repeats_mix_after_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            project = Project(path=path, conversion_source="recording",
                              stages={"import": {"status": "완료"},
                                      "separate": {"status": "완료"},
                                      "mix": {"status": "완료"}})
            project.save()
            for name in ("source.wav", "vocals.wav", "final.mp3"):
                project.artifact(name).touch()
            called = []
            worker = StageWorker(path, "all")
            with patch("app.core.stage_process.run_stage_process",
                       side_effect=lambda _path, stage, *_args: called.append(stage)):
                worker.run()
            self.assertEqual(called, ["convert", "mix"])


if __name__ == "__main__":
    unittest.main()
