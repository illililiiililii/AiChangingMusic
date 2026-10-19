"""Fast checks of project persistence, score timing and audio export."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.pipeline import run_stage
from app.core.project import Project
from app.lyric_generation.alignment import align
from app.music_analysis.syllables import estimate_syllables
from app.mixing.render import render_mix


class IntegrationTest(unittest.TestCase):
    def test_project_import_alignment_and_mix(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rate = 44100
            time = np.arange(rate * 2) / rate
            vocal = (.1 * np.sin(2 * np.pi * 220 * time)).astype("float32")
            source = folder / "input.wav"
            sf.write(source, np.stack([vocal, vocal], axis=1), rate)
            project = Project(name="검증", source=str(source), conversion_index_rate=.4,
                              conversion_pitch_semitones=-12, conversion_protect=.2,
                              path=folder / "project" / "project.json")
            project.save()
            self.assertIn("source.wav", run_stage(project.path, "import"))
            loaded = Project.load(project.path)
            self.assertEqual(loaded.name, "검증")
            self.assertEqual(loaded.conversion_index_rate, .4)
            self.assertEqual(loaded.conversion_pitch_semitones, -12)
            self.assertEqual(loaded.conversion_protect, .2)
            self.assertTrue(loaded.artifact("source.wav").is_file())
            notes = [{"start": 0.1, "end": 0.5, "midi": 60},
                     {"start": 0.6, "end": 1.2, "midi": 64}]
            score = align("가나", notes)
            self.assertEqual([e["midi"] for e in score["syllables"]], [60, 64])
            self.assertEqual([e["start"] for e in score["syllables"]], [0.1, 0.6])
            self.assertEqual(len(align("가나", notes[:1])["syllables"]), 2)
            self.assertEqual([event["syllable"] for event in align("12BPM", notes * 3)["syllables"][:3]],
                             ["십", "이", "비"])
            self.assertTrue(align("가", notes * 10)["partial"])
            estimated = estimate_syllables(
                {"segments": [{"words": [{"start": .1, "end": .5, "text": " 가나다"}]}]}, notes)
            self.assertEqual([unit["syllable"] for unit in estimated], ["가", "나", "다"])
            self.assertEqual(estimated[0]["note_index"], 0)
            corrected = estimate_syllables(
                {"segments": [{"text": "틀린 인식", "words": [
                    {"start": .1, "end": .5, "text": " 틀린"},
                    {"start": .6, "end": 1.2, "text": " 인식"}]}]}, notes, "바른 가사")
            self.assertEqual("".join(unit["syllable"] for unit in corrected), "바른가사")
            self.assertTrue(all(unit["method"] == "edited lyrics over ASR intervals" for unit in corrected))
            stem = folder / "stem.wav"
            sf.write(stem, vocal[:rate], rate)
            target = folder / "mix.wav"
            result = render_mix(stem, source, target, reverb=.1)
            data, sample_rate = sf.read(target)
            self.assertEqual(sample_rate, rate)
            self.assertEqual(len(data), 2 * rate)
            self.assertLessEqual(result["peak_after_guard"], .97 + 1e-6)

    def test_gui_opens_offscreen(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont, QFontDatabase
        from app.gui.main import Studio
        app = QApplication.instance() or QApplication([])
        font = Path("C:/Windows/Fonts/malgun.ttf")
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
            app.setFont(QFont("Malgun Gothic", 10))
        window = Studio()
        self.assertEqual(window.tabs.count(), 5)
        self.assertIn("원곡", window.tabs.tabText(0))
        window.close()


if __name__ == "__main__":
    unittest.main()
