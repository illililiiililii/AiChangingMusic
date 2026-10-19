"""Fast checks of project persistence, score timing and audio export."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from app.core.pipeline import run_stage
from app.core.project import Project
from app.lyric_generation.alignment import align
from app.music_analysis.syllables import estimate_syllables
from app.mixing.render import render_mix
from scripts.render_singing import phrase_inputs
from scripts.render_korean_singing import score_lines


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
            # Changed line lengths must not push the next line across a rest.
            line_notes = [
                {"start": .0, "end": .4, "midi": 60},
                {"start": .4, "end": .8, "midi": 62},
                {"start": .8, "end": 1.2, "midi": 64},
                {"start": 3.0, "end": 3.4, "midi": 65},
                {"start": 3.4, "end": 3.8, "midi": 67},
            ]
            changed = align("가나다라\n마", line_notes, "가나다\n라마")
            self.assertEqual(len(changed["syllables"]), 5)
            self.assertEqual(changed["line_note_ranges"], [(0, 3), (3, 5)])
            self.assertEqual(changed["syllables"][4]["start"], 3.0)
            self.assertEqual(changed["syllables"][3]["end"], 1.2)
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

    def test_singing_score_preserves_short_rests(self):
        phrase = [{"syllable": "가", "start": 1.0, "end": 1.2, "midi": 60},
                  {"syllable": "나", "start": 1.3, "end": 1.5, "midi": 62}]
        durations, lyrics, notes = phrase_inputs(phrase)
        self.assertEqual(lyrics, ["가", "<X>", "나"])
        self.assertEqual(notes, [72, 0, 74])
        self.assertAlmostEqual(sum(durations), .5)
        self.assertAlmostEqual(durations[1], .1)

    def test_korean_renderer_uses_edited_lyrics(self):
        events = [{"syllable": char, "start": index * .3, "end": (index + 1) * .3,
                   "midi": 60} for index, char in enumerate("바른가사")]
        lines = score_lines("바른 가사", events)
        self.assertEqual(lines[0][0], ["바른", "가사"])
        with self.assertRaisesRegex(ValueError, "가사와 저장된"):
            score_lines("틀린 가사", events)

    def test_existing_project_keeps_legacy_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text('{"name": "기존 프로젝트", "conversion_pitch_semitones": -12}', encoding="utf-8")
            loaded = Project.load(path)
            self.assertEqual(loaded.singing_engine, "diffsinger")
            self.assertEqual(loaded.conversion_pitch_semitones, -12)

    def test_new_singing_requires_fresh_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            project = Project(path=folder / "project.json", parody_lyrics="가",
                              stages={"sing": {"status": "완료"},
                                      "convert": {"status": "완료"}, "mix": {"status": "완료"}})
            project.save()
            (folder / "score.json").write_text('{"syllables": [{"syllable": "가", "start": 0, "end": 1, "midi": 60}]}',
                                                  encoding="utf-8")
            (folder / "singing.wav").touch()
            with patch("app.core.pipeline._run") as run_mock:
                run_stage(project.path, "sing")
            self.assertIn("render_korean_singing.py", " ".join(run_mock.call_args.args[0]))
            loaded = Project.load(project.path)
            self.assertNotIn("convert", loaded.stages)
            self.assertNotIn("mix", loaded.stages)
            model, index = folder / "voice.pth", folder / "voice.index"
            model.touch()
            index.touch()
            loaded.voice_consent = True
            loaded.voice_model = str(model)
            loaded.voice_index = str(index)
            loaded.stages.pop("sing")
            loaded.save()
            with self.assertRaisesRegex(ValueError, "노래 합성을 다시 실행"):
                run_stage(project.path, "convert")

    def test_recorded_guide_is_used_for_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            guide = folder / "recorded.wav"
            sf.write(guide, np.zeros(4410, dtype="float32"), 44100)
            model, index, rvc = (folder / name for name in ("model.pth", "model.index", "python.exe"))
            for path in (model, index, rvc):
                path.touch()
            project = Project(path=folder / "project.json", voice_consent=True,
                              voice_model=str(model), voice_index=str(index),
                              conversion_source="recording", guide_vocal=str(guide))
            project.save()
            with patch("app.core.pipeline.RVC_PYTHON", rvc), \
                 patch("app.core.pipeline.decode", return_value=folder / "guide_vocal.wav") as decode_mock, \
                 patch("app.core.pipeline._run") as run_mock, \
                 patch("app.core.pipeline._gpu_with_free_memory", return_value=""), \
                 patch("app.core.pipeline.ffmpeg", return_value=folder / "ffmpeg.exe"):
                run_stage(project.path, "convert")
            self.assertEqual(decode_mock.call_args.args, (guide, folder / "guide_vocal.wav"))
            self.assertEqual(decode_mock.call_args.kwargs["channels"], 1)
            command = run_mock.call_args.args[0]
            self.assertEqual(command[command.index("--input") + 1], str(folder / "guide_vocal.wav"))

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
        self.assertEqual(window.conversion_source.count(), 2)
        window.conversion_source.setCurrentIndex(window.conversion_source.findData("recording"))
        window.guide_vocal_edit.setText("C:/guide.wav")
        window._sync()
        self.assertEqual(window.project.conversion_source, "recording")
        self.assertEqual(window.project.guide_vocal, "C:/guide.wav")
        window.project.singing_engine = "diffsinger"
        window.project.conversion_source = "synth"
        window.project.conversion_pitch_semitones = -12
        window._refresh()
        window.singing_engine.setCurrentIndex(window.singing_engine.findData("korean_tts"))
        window._sync()
        self.assertEqual(window.voice_pitch.value(), 0)
        window.close()


if __name__ == "__main__":
    unittest.main()
