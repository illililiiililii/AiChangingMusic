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
from app.core.stage_process import run_stage_process
from app.audio_processing.io import waveform
from app.lyric_generation.alignment import align
from app.music_analysis.syllables import estimate_syllables
from app.music_analysis.midi_io import read_midi, write_midi, monophonic_draft
from app.mixing.render import render_mix
from scripts.render_singing import phrase_inputs
from scripts.render_korean_singing import SAMPLE_RATE, score_lines, trim_tts_silence


class IntegrationTest(unittest.TestCase):
    def test_stage_process_failure_keeps_parent_alive_and_marks_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Project(path=Path(temporary) / "project.json")
            project.save()
            with self.assertRaisesRegex(RuntimeError, "하위 프로세스 종료"):
                run_stage_process(project.path, "analyze")
            self.assertEqual(Project.load(project.path).stages["analyze"]["status"], "실패")

    def test_waveform_reads_bounded_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            sf.write(path, np.ones((44100, 2), dtype="float32") * .25, 44100)
            values, duration = waveform(path, points=100)
            self.assertEqual(len(values), 100)
            self.assertAlmostEqual(duration, 1.)
            self.assertAlmostEqual(float(values.max()), .25, places=2)

    def test_midi_round_trip_uses_selected_bpm_and_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "melody.mid"
            notes = [{"start": .125, "end": .375, "midi": 60},
                     {"start": .5, "end": .85, "midi": 64}]
            write_midi(path, notes, 180, ["가", "나"])
            loaded, bpm = read_midi(path)
            self.assertAlmostEqual(bpm, 180, places=2)
            self.assertEqual([note["midi"] for note in loaded], [60, 64])
            self.assertAlmostEqual(loaded[0]["start"], .125, delta=.002)
            self.assertAlmostEqual(loaded[1]["end"], .85, delta=.002)
            draft = monophonic_draft([
                {"start": .0, "end": .4, "midi": 60, "velocity": 90},
                {"start": .1, "end": .3, "midi": 72, "velocity": 40},
                {"start": .5, "end": .9, "midi": 64, "velocity": 80}])
            self.assertEqual([note["midi"] for note in draft], [60, 64])

    def test_basic_pitch_keeps_edited_notes_as_separate_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root = folder / "app"
            executable = root / ".venv-basicpitch" / "Scripts" / "basic-pitch.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            project = Project(path=folder / "song" / "project.json")
            project.analysis["notes"] = [{"start": 1., "end": 1.5, "midi": 60}]
            project.save()
            project.artifact("vocals.wav").touch()

            def make_midi(command, **_):
                output = Path(command[1])
                write_midi(output / "candidate.mid",
                           [{"start": 2., "end": 2.5, "midi": 64}], 120)

            with patch("app.core.pipeline.ROOT", root), patch("app.core.pipeline._run", side_effect=make_midi):
                run_stage(project.path, "basic_pitch")
            loaded = Project.load(project.path)
            self.assertEqual(loaded.analysis["notes"][0]["midi"], 60)
            self.assertEqual(loaded.analysis["basic_pitch_notes"][0]["midi"], 64)

    def test_korean_tts_padding_is_trimmed_before_time_stretch(self):
        tone = .1 * np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE // 5) / SAMPLE_RATE)
        padded = np.concatenate((np.zeros(SAMPLE_RATE // 3), tone, np.zeros(SAMPLE_RATE // 3)))
        trimmed = trim_tts_silence(padded)
        self.assertLess(len(trimmed), len(padded) // 2)
        self.assertGreater(np.max(np.abs(trimmed[:SAMPLE_RATE // 20])), .05)

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
        self.assertEqual(window.conversion_source.count(), 3)
        window.bpm_override.setValue(180)
        window.beat_offset.setValue(18.9)
        window._sync()
        self.assertEqual(window.project.bpm_override, 180)
        self.assertEqual(window.project.beat_offset_seconds, 18.9)
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

    def test_piano_roll_mouse_edits_notes(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
        from app.gui.piano_roll import PianoRollDialog
        app = QApplication.instance() or QApplication([])
        dialog = PianoRollDialog([{"start": 1., "end": 1.4, "midi": 60}], 180)
        dialog.show()
        app.processEvents()
        view = dialog.view
        empty = view.mapFromScene(view.time_x(3), view.pitch_y(64) + 5)
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=empty)
        self.assertEqual(len(dialog.notes()), 2)
        first = view.items_list[0]
        point = view.mapFromScene(first.scenePos().x() + 10, first.scenePos().y() + 5)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(view.viewport(), point + QPoint(30, -18))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=point + QPoint(30, -18))
        self.assertGreater(dialog.notes()[0]["start"], 1.)
        self.assertEqual(dialog.notes()[0]["midi"], 61)
        dialog.undo()
        self.assertEqual(dialog.notes()[0]["midi"], 60)
        dialog.close()

    def test_piano_roll_save_updates_project_notes(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        from app.gui.main import Studio
        from app.gui.piano_roll import PianoRollDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            project = Project(path=path)
            project.analysis["notes"] = [{"start": 1., "end": 1.5, "midi": 60}]
            project.save()
            window = Studio()
            window.load_project(path)
            with patch.object(PianoRollDialog, "exec", return_value=QDialog.DialogCode.Accepted):
                window.open_piano_roll()
            self.assertEqual(Project.load(path).analysis["notes"][0]["midi"], 60)
            self.assertEqual(Project.load(path).analysis["notes"][0]["start"], 1.)
            window.close()


if __name__ == "__main__":
    unittest.main()
