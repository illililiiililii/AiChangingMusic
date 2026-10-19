"""Desktop editor and stage runner. Long audio work runs outside the UI thread."""

from __future__ import annotations

import sys
import subprocess
import traceback
import time
import re
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot, Qt
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.audio_processing.io import waveform
from app.core.project import Project, STAGES
from app.core.activity_log import LOG_DIR
from app.core.settings import ROOT, recent_projects, remember


LABELS = {"all": "전체 생성", "import": "원곡 가져오기", "separate": "보컬·반주 분리", "transcribe": "원곡 가사 인식",
          "analyze": "박자·멜로디 분석", "generate": "로컬 AI 개사", "lyrics": "가사 정렬", "sing": "노래 합성",
          "train": "목소리 학습", "sing_segment": "선택 구간 노래 재생성",
          "convert": "음색 변환", "mix": "믹싱·출력"}


class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(118)
        self.values = None
        self.seconds = 0.0

    def set_file(self, path: Path | None):
        self.values, self.seconds = (None, 0.0)
        if path and path.is_file():
            try:
                self.values, self.seconds = waveform(path)
            except Exception:
                pass
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#152030"))
        if self.values is None or len(self.values) == 0:
            painter.setPen(QColor("#91a4ba"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "오디오를 선택하면 파형이 표시됩니다")
            return
        painter.setPen(QPen(QColor("#54d2c5"), 1))
        middle = self.height() / 2
        step = max(1, int(len(self.values) / max(self.width(), 1)))
        for x in range(self.width()):
            region = self.values[x * step:(x + 1) * step]
            if len(region) == 0:
                break
            height = float(max(region)) * (middle - 8)
            painter.drawLine(x, int(middle - height), x, int(middle + height))
        painter.setPen(QColor("#d2e0eb"))
        painter.drawText(8, 18, f"{self.seconds:.1f}초")


class StageWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    progress = Signal(str)
    stage_progress = Signal(int, int)

    def __init__(self, path: Path, stage: str):
        super().__init__()
        self.path, self.stage = path, stage

    def run(self):
        try:
            from app.core.pipeline import run_stage
            if self.stage == "all":
                project = Project.load(self.path)
                stages = ["import", "separate"]
                if project.conversion_source == "synth":
                    stages.extend(["transcribe", "analyze"])
                    if not project.parody_lyrics.strip() and project.theme.strip():
                        stages.append("generate")
                    stages.append("lyrics")
                if not project.voice_model and project.voice_source:
                    stages.append("train")
                if project.conversion_source == "synth":
                    stages.append("sing")
                stages.extend(["convert", "mix"])
                expected = {"import": "source.wav", "separate": "vocals.wav", "transcribe": "transcription.json",
                            "analyze": "analysis.json", "lyrics": "score.json", "sing": "singing.wav",
                            "convert": "converted.wav", "mix": "final.mp3"}
                stages = [stage for stage in stages if
                          project.stages.get(stage, {}).get("status") != "완료" or
                          (stage in expected and not project.artifact(expected[stage]).is_file())]
                for index, stage in enumerate(stages, 1):
                    self.progress.emit(f"[{index}/{len(stages)}] {LABELS[stage]}")
                    run_stage(self.path, stage, self.progress.emit)
                    self.stage_progress.emit(index, len(stages))
                result = "전체 처리 완료: final.wav, final.mp3"
            else:
                if self.stage in ("separate", "transcribe", "analyze"):
                    project = Project.load(self.path)
                    if (project.stages.get("import", {}).get("status") != "완료"
                            or not project.artifact("source.wav").is_file()):
                        self.progress.emit("선행 작업: 원곡 가져오기")
                        run_stage(self.path, "import", self.progress.emit)
                if self.stage in ("transcribe", "analyze"):
                    project = Project.load(self.path)
                    if (project.stages.get("separate", {}).get("status") != "완료"
                            or not project.artifact("vocals.wav").is_file()):
                        self.progress.emit("선행 작업: 보컬·반주 분리")
                        run_stage(self.path, "separate", self.progress.emit)
                self.progress.emit(f"{LABELS[self.stage]} 실행 중")
                result = run_stage(self.path, self.stage, self.progress.emit)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = Project()
        self.thread = None
        self.worker = None
        self._active_stage = None
        self._train_started = None
        self._train_phase = ""
        self._train_timer = QTimer(self)
        self._train_timer.timeout.connect(self._update_train_status)
        self.setWindowTitle("AI Parody Music Studio")
        self.resize(1130, 820)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self._build()
        self._refresh()

    def _build(self):
        menu = self.menuBar().addMenu("프로젝트")
        for title, callback in (("새 프로젝트", self.new_project), ("열기", self.open_project),
                                ("저장", self.save_project)):
            action = QAction(title, self)
            action.triggered.connect(callback)
            menu.addAction(action)
        self.recent_menu = menu.addMenu("최근 프로젝트")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._source_tab(), "1 원곡")
        self.tabs.addTab(self._lyrics_tab(), "2 가사")
        self.tabs.addTab(self._voice_tab(), "3 목소리")
        self.tabs.addTab(self._render_tab(), "4 생성·출력")
        self.tabs.addTab(self._models_tab(), "5 모델·환경")
        self.statusBar().showMessage("프로젝트를 만들거나 여세요")
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#101923; color:#e4ecf4; font-size:13px; }
            QLineEdit, QPlainTextEdit, QComboBox { background:#1b2a39; border:1px solid #40536a;
                border-radius:5px; padding:6px; selection-background-color:#277c84; }
            QPushButton { background:#245a68; border:1px solid #3d8692; border-radius:5px; padding:8px 12px; }
            QPushButton:hover { background:#327d89; } QPushButton:disabled { color:#8294a2; background:#263540; }
            QGroupBox { border:1px solid #34495d; margin-top:10px; padding:12px; border-radius:5px; }
            QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
            QTabBar::tab { padding:10px 19px; background:#1a2a3b; }
            QTabBar::tab:selected { background:#245a68; }
        """)

    def _source_tab(self):
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        row = QHBoxLayout()
        self.source_path = QLineEdit()
        self.source_path.setReadOnly(True)
        row.addWidget(self.source_path)
        choose = QPushButton("음악 파일 선택")
        choose.clicked.connect(self.choose_source)
        row.addWidget(choose)
        layout.addLayout(row)
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel("원곡 언어"))
        self.source_language = QComboBox()
        self.source_language.addItem("한국어", "ko")
        self.source_language.addItem("English", "en")
        self.source_language.addItem("자동 감지", "auto")
        language_row.addWidget(self.source_language)
        language_row.addStretch()
        layout.addLayout(language_row)
        self.wave = Waveform()
        layout.addWidget(self.wave)
        play = QPushButton("원곡 재생 / 일시정지")
        play.clicked.connect(self.play_source)
        layout.addWidget(play)
        buttons = QGridLayout()
        self.stage_buttons = {}
        for i, stage in enumerate(("import", "separate", "transcribe", "analyze")):
            button = QPushButton(LABELS[stage])
            button.clicked.connect(lambda checked=False, s=stage: self.run(s))
            buttons.addWidget(button, i // 2, i % 2)
            self.stage_buttons[stage] = button
        layout.addLayout(buttons)
        self.bpm_label = QLabel("BPM: 분석 전")
        layout.addWidget(self.bpm_label)
        self.structure_label = QLabel("구간: 분석 전")
        layout.addWidget(self.structure_label)
        layout.addWidget(QLabel("원곡 가사 인식 결과 (직접 수정 가능)"))
        self.original_text = QPlainTextEdit()
        self.original_text.setMinimumHeight(100)
        layout.addWidget(self.original_text)
        layout.addWidget(QLabel("인식 음절 시각 추정 (Whisper 단어 구간 균등 분할, 검토 필요)"))
        self.original_syllables_table = QTableWidget(0, 4)
        self.original_syllables_table.setHorizontalHeaderLabels(["음절", "시작", "끝", "대응 음표"])
        self.original_syllables_table.setMinimumHeight(120)
        layout.addWidget(self.original_syllables_table)
        layout.addWidget(QLabel("음표 편집: 시작(초) · 끝(초) · MIDI 음정"))
        self.notes_table = QTableWidget(0, 3)
        self.notes_table.setHorizontalHeaderLabels(["시작", "끝", "MIDI"])
        self.notes_table.setMinimumHeight(150)
        layout.addWidget(self.notes_table)
        note_row = QHBoxLayout()
        add_note = QPushButton("음표 추가")
        add_note.clicked.connect(lambda: self.notes_table.insertRow(self.notes_table.rowCount()))
        note_row.addWidget(add_note)
        remove_note = QPushButton("선택 음표 삭제")
        remove_note.clicked.connect(lambda: self.notes_table.removeRow(self.notes_table.currentRow())
                                    if self.notes_table.currentRow() >= 0 else None)
        note_row.addWidget(remove_note)
        save_notes = QPushButton("음표 저장")
        save_notes.clicked.connect(self.save_notes)
        note_row.addWidget(save_notes)
        layout.addLayout(note_row)
        previews = QHBoxLayout()
        for title, name in (("보컬 듣기", "vocals.wav"), ("반주 듣기", "instrumental.wav")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, n=name: self.play_artifact(n))
            previews.addWidget(button)
        layout.addLayout(previews)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _lyrics_tab(self):
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        self.theme = QLineEdit()
        self.theme.setPlaceholderText("개사 주제 (로컬 Ollama 사용 시)")
        layout.addWidget(self.theme)
        self.style = QComboBox()
        self.style.addItems(["유머", "진지함", "풍자"])
        layout.addWidget(self.style)
        self.llm_model = QLineEdit()
        self.llm_model.setPlaceholderText("Ollama 모델명 (예: qwen3.5:4b)")
        layout.addWidget(self.llm_model)
        self.parody_text = QPlainTextEdit()
        self.parody_text.setPlaceholderText("새 가사를 입력하세요. 한 음절씩 음표에 정렬됩니다.")
        layout.addWidget(self.parody_text)
        row = QHBoxLayout()
        generate = QPushButton("로컬 AI 개사")
        generate.clicked.connect(self.generate_lyrics)
        row.addWidget(generate)
        align = QPushButton("음절·음표 정렬 및 평가")
        align.clicked.connect(lambda: self.run("lyrics"))
        row.addWidget(align)
        layout.addLayout(row)
        self.fit_label = QLabel("적합도: 분석 전")
        layout.addWidget(self.fit_label)
        layout.addWidget(QLabel("음절 정렬 편집: 음절 · 시작(초) · 끝(초) · MIDI"))
        self.alignment_table = QTableWidget(0, 4)
        self.alignment_table.setHorizontalHeaderLabels(["음절", "시작", "끝", "MIDI"])
        layout.addWidget(self.alignment_table)
        save_alignment = QPushButton("음절 타이밍·음정 저장")
        save_alignment.clicked.connect(self.save_alignment)
        layout.addWidget(save_alignment)
        return page

    def _voice_tab(self):
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        self.voice_consent = QCheckBox("이 음성의 AI 학습 및 노래 변환에 대한 명시적 허락을 받았습니다")
        layout.addWidget(self.voice_consent)
        for attr, title, pattern in (("voice_model", "RVC 모델", "*.pth"), ("voice_index", "RVC 인덱스", "*.index")):
            row = QHBoxLayout()
            edit = QLineEdit()
            setattr(self, attr + "_edit", edit)
            row.addWidget(edit)
            button = QPushButton(title + " 선택")
            button.clicked.connect(lambda checked=False, e=edit, p=pattern: self.choose_file(e, p))
            row.addWidget(button)
            layout.addLayout(row)
        layout.addWidget(QLabel("말하기 음성 학습은 아래 관리 화면에서 실행합니다. 기존 프로필도 선택할 수 있습니다."))
        self.voice_files = QLineEdit()
        self.voice_files.setPlaceholderText("말하기 WAV 파일이 들어있는 폴더")
        layout.addWidget(self.voice_files)
        row = QHBoxLayout()
        choose = QPushButton("음성 폴더 선택")
        choose.clicked.connect(self.choose_voice_folder)
        row.addWidget(choose)
        choose_file = QPushButton("음성 파일 선택")
        choose_file.clicked.connect(self.choose_voice_file)
        row.addWidget(choose_file)
        self.train_button = QPushButton("RVC 프로필 학습")
        self.train_button.clicked.connect(self.train_voice)
        row.addWidget(self.train_button)
        layout.addLayout(row)
        self.train_status = QLabel("학습 대기 중")
        layout.addWidget(self.train_status)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)
        index_row = QHBoxLayout()
        index_row.addWidget(QLabel("음색 반영 강도 (잡음이 심하면 낮추세요)"))
        self.index_rate = QDoubleSpinBox()
        self.index_rate.setRange(0, 1)
        self.index_rate.setDecimals(2)
        self.index_rate.setSingleStep(0.05)
        index_row.addWidget(self.index_rate)
        layout.addLayout(index_row)
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("변환 보컬 음높이 (반음, 낮은 목소리는 -12부터 시험)"))
        self.voice_pitch = QSpinBox()
        self.voice_pitch.setRange(-24, 24)
        pitch_row.addWidget(self.voice_pitch)
        layout.addLayout(pitch_row)
        protect_row = QHBoxLayout()
        protect_row.addWidget(QLabel("자음·숨소리 보호 (낮을수록 강함)"))
        self.voice_protect = QDoubleSpinBox()
        self.voice_protect.setRange(0, 0.5)
        self.voice_protect.setDecimals(2)
        self.voice_protect.setSingleStep(0.05)
        protect_row.addWidget(self.voice_protect)
        layout.addLayout(protect_row)
        layout.addWidget(QLabel("음색 변환에 사용할 노래 보컬"))
        self.conversion_source = QComboBox()
        self.conversion_source.addItem("앱이 합성한 보컬 (singing.wav)", "synth")
        self.conversion_source.addItem("직접 녹음한 보컬 파일", "recording")
        layout.addWidget(self.conversion_source)
        recording_row = QHBoxLayout()
        self.guide_vocal_edit = QLineEdit()
        self.guide_vocal_edit.setPlaceholderText("반주에 맞춰 부른 보컬만 있는 WAV/MP3/FLAC/OGG")
        recording_row.addWidget(self.guide_vocal_edit)
        choose_recording = QPushButton("녹음 파일 선택")
        choose_recording.clicked.connect(self.choose_recording)
        recording_row.addWidget(choose_recording)
        preview_recording = QPushButton("녹음 미리듣기")
        preview_recording.clicked.connect(lambda: self.play(self.guide_vocal_edit.text().strip()))
        recording_row.addWidget(preview_recording)
        layout.addLayout(recording_row)
        layout.addWidget(QLabel("녹음 파일은 원곡 반주와 시작 시각을 맞추고 반주 소리는 빼 주세요. 부른 음정을 유지하려면 음높이 0."))
        layout.addStretch()
        return page

    def _render_tab(self):
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("노래 합성 방식"))
        self.singing_engine = QComboBox()
        self.singing_engine.addItem("한국어 발음 우선 (박자에 맞춘 말소리)", "korean_tts")
        self.singing_engine.addItem("기존 DiffSingerKR (음표 중심)", "diffsinger")
        engine_row.addWidget(self.singing_engine)
        layout.addLayout(engine_row)
        grid = QGridLayout()
        self.stage_status = {}
        for i, stage in enumerate(STAGES):
            label = QLabel()
            grid.addWidget(label, i, 0)
            self.stage_status[stage] = label
            if stage in ("sing", "convert", "mix"):
                button = QPushButton(LABELS[stage])
                button.clicked.connect(lambda checked=False, s=stage: self.run(s))
                grid.addWidget(button, i, 1)
                self.stage_buttons[stage] = button
        layout.addLayout(grid)
        all_button = QPushButton("원곡부터 최종 출력까지 전체 실행")
        all_button.clicked.connect(lambda: self.run("all"))
        layout.addWidget(all_button)
        form = QFormLayout()
        self.vocal_gain = QSlider(Qt.Orientation.Horizontal)
        self.vocal_gain.setRange(0, 200)
        self.instrumental_gain = QSlider(Qt.Orientation.Horizontal)
        self.instrumental_gain.setRange(0, 200)
        self.reverb = QSlider(Qt.Orientation.Horizontal)
        self.reverb.setRange(0, 40)
        form.addRow("보컬 볼륨 (%)", self.vocal_gain)
        form.addRow("반주 볼륨 (%)", self.instrumental_gain)
        form.addRow("보컬 리버브 (%)", self.reverb)
        layout.addLayout(form)
        row = QHBoxLayout()
        for title, name in (("합성 보컬", "singing.wav"), ("변환 보컬", "converted.wav"),
                            ("최종 미리듣기", "final.wav")):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, n=name: self.play_artifact(n))
            row.addWidget(button)
        layout.addLayout(row)
        segment = QHBoxLayout()
        segment.addWidget(QLabel("구간 미리듣기"))
        self.segment_start = QDoubleSpinBox()
        self.segment_end = QDoubleSpinBox()
        for box in (self.segment_start, self.segment_end):
            box.setRange(0, 3600)
            box.setDecimals(2)
            box.setSuffix(" 초")
        self.segment_end.setValue(10)
        segment.addWidget(self.segment_start)
        segment.addWidget(self.segment_end)
        preview = QPushButton("변환 보컬 구간 재생")
        preview.clicked.connect(self.play_segment)
        segment.addWidget(preview)
        regenerate = QPushButton("선택 구간 노래 재생성")
        regenerate.clicked.connect(lambda: self.run("sing_segment"))
        segment.addWidget(regenerate)
        layout.addLayout(segment)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        open_logs = QPushButton("로그 폴더 열기")
        open_logs.clicked.connect(self.open_logs)
        layout.addWidget(open_logs)
        return page

    def open_logs(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_DIR)))

    def _models_tab(self):
        page, layout = QWidget(), QVBoxLayout()
        page.setLayout(layout)
        layout.addWidget(QLabel("모델 파일은 이 PC에만 저장됩니다. 최초 설치는 setup.bat을 실행하세요."))
        self.model_status = QPlainTextEdit()
        self.model_status.setReadOnly(True)
        layout.addWidget(self.model_status)
        refresh = QPushButton("모델·GPU 상태 다시 확인")
        refresh.clicked.connect(self.refresh_models)
        layout.addWidget(refresh)
        guide = QPushButton("설치 안내 열기")
        guide.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT / "INSTALL.md"))))
        layout.addWidget(guide)
        return page

    def refresh_models(self):
        checks = {
            "DiffSingerKR": ROOT / "external" / "DiffSingerKR-space" / "Checkpoint" / "S_200000.pt",
            "RVC": ROOT / "external" / "RVC" / "infer" / "cli.py",
            "RVC 음성 특징 모델": ROOT / "external" / "RVC" / "assets" / "hubert_base" / "pytorch_model.bin",
            "Demucs": ROOT / "models" / "torch" / "hub" / "checkpoints" / "955717e8-8726e21a.th",
            "Whisper": ROOT / "models" / "faster-whisper-base" / "model.bin",
            "FFmpeg": ROOT / "external" / "ffmpeg" / "ffmpeg.exe",
        }
        lines = [f"{name}: {'설치됨' if path.is_file() else '없음'}\n  {path}" for name, path in checks.items()]
        try:
            nvidia_smi = Path("C:/Windows/System32/nvidia-smi.exe")
            gpu = subprocess.run([str(nvidia_smi) if nvidia_smi.is_file() else "nvidia-smi",
                                  "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            lines.append("GPU:\n" + (gpu.stdout.strip() if gpu.returncode == 0 else "감지 실패"))
        except (OSError, subprocess.TimeoutExpired):
            lines.append("GPU: NVIDIA 도구를 찾지 못했습니다")
        self.model_status.setPlainText("\n\n".join(lines))

    def _refresh(self):
        self.setWindowTitle(f"AI Parody Music Studio — {self.project.name}")
        self.source_path.setText(self.project.source)
        self.source_language.setCurrentIndex(max(0, self.source_language.findData(self.project.source_language)))
        self.original_text.setPlainText(self.project.original_lyrics)
        self.parody_text.setPlainText(self.project.parody_lyrics)
        self.theme.setText(self.project.theme)
        self.style.setCurrentText(self.project.lyric_style)
        self.llm_model.setText(self.project.llm_model)
        self.voice_model_edit.setText(self.project.voice_model)
        self.voice_index_edit.setText(self.project.voice_index)
        self.voice_files.setText(self.project.voice_source)
        self.voice_consent.setChecked(self.project.voice_consent)
        self.index_rate.setValue(self.project.conversion_index_rate)
        self.voice_pitch.setValue(self.project.conversion_pitch_semitones)
        self.voice_protect.setValue(self.project.conversion_protect)
        self.singing_engine.setCurrentIndex(max(0, self.singing_engine.findData(self.project.singing_engine)))
        self.conversion_source.setCurrentIndex(max(0, self.conversion_source.findData(self.project.conversion_source)))
        self.guide_vocal_edit.setText(self.project.guide_vocal)
        self.vocal_gain.setValue(round(self.project.vocal_gain * 100))
        self.instrumental_gain.setValue(round(self.project.instrumental_gain * 100))
        self.reverb.setValue(round(self.project.reverb * 100))
        self.segment_start.setValue(self.project.segment_start)
        self.segment_end.setValue(self.project.segment_end)
        self.bpm_label.setText(f"BPM: {self.project.analysis.get('bpm', '분석 전')}")
        sections = self.project.analysis.get("sections", [])
        self.structure_label.setText("구간: " + (" · ".join(
            f"{item['start']:.1f}-{item['end']:.1f}s {item['label']}" for item in sections[:5])
            if sections else "분석 전"))
        self.fit_label.setText(f"적합도: {self.project.analysis.get('lyric_fit', '분석 전')}")
        notes = self.project.analysis.get("notes", [])
        self.notes_table.setRowCount(len(notes))
        for row, note in enumerate(notes):
            for column, key in enumerate(("start", "end", "midi")):
                self.notes_table.setItem(row, column, QTableWidgetItem(str(note[key])))
        original_units = self.project.analysis.get("original_syllables", [])
        self.original_syllables_table.setRowCount(len(original_units))
        for row, unit in enumerate(original_units):
            values = (unit["syllable"], unit["start"], unit["end"],
                      "" if unit["note_index"] is None else unit["note_index"] + 1)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.original_syllables_table.setItem(row, column, item)
        events = self.project.analysis.get("alignment", [])
        self.alignment_table.setRowCount(len(events))
        for row, event in enumerate(events):
            for column, key in enumerate(("syllable", "start", "end", "midi")):
                self.alignment_table.setItem(row, column, QTableWidgetItem(str(event[key])))
        for stage, label in self.stage_status.items():
            state = self.project.stages.get(stage, {})
            label.setText(f"{LABELS[stage]}: {state.get('status', '대기')} {state.get('message', '')}")
        self.recent_menu.clear()
        for path in recent_projects():
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda checked=False, p=path: self.load_project(Path(p)))
        normalized_source = self.project.artifact("source.wav") if self.project.path else None
        if (normalized_source and normalized_source.is_file()
                and self.project.stages.get("import", {}).get("status") == "완료"):
            self.wave.set_file(normalized_source)
        else:
            self.wave.set_file(Path(self.project.source) if self.project.source else None)
        self.refresh_models()

    def _sync(self):
        if self.singing_engine.currentData() != self.project.singing_engine:
            for stage in ("sing", "sing_segment", "convert", "mix"):
                self.project.stages.pop(stage, None)
        if self.original_text.toPlainText() != self.project.original_lyrics:
            self.project.analysis.pop("original_syllables", None)
            self.project.analysis.pop("syllable_timing_method", None)
            self.project.stages.pop("analyze", None)
        if self.source_language.currentData() != self.project.source_language:
            for stage in ("transcribe", "generate", "lyrics", "sing", "sing_segment", "convert", "mix"):
                self.project.stages.pop(stage, None)
        if self.parody_text.toPlainText() != self.project.parody_lyrics:
            for stage in ("lyrics", "sing", "sing_segment", "convert", "mix"):
                self.project.stages.pop(stage, None)
            self.project.analysis.pop("alignment", None)
            self.project.analysis.pop("lyric_fit", None)
        if (self.voice_model_edit.text().strip() != self.project.voice_model or
                self.voice_index_edit.text().strip() != self.project.voice_index or
                self.index_rate.value() != self.project.conversion_index_rate or
                self.voice_pitch.value() != self.project.conversion_pitch_semitones or
                self.voice_protect.value() != self.project.conversion_protect or
                self.conversion_source.currentData() != self.project.conversion_source or
                self.guide_vocal_edit.text().strip() != self.project.guide_vocal):
            for stage in ("convert", "mix"):
                self.project.stages.pop(stage, None)
        if (self.vocal_gain.value() / 100 != self.project.vocal_gain or
                self.instrumental_gain.value() / 100 != self.project.instrumental_gain or
                self.reverb.value() / 100 != self.project.reverb):
            self.project.stages.pop("mix", None)
        self.project.original_lyrics = self.original_text.toPlainText()
        self.project.source_language = self.source_language.currentData()
        self.project.parody_lyrics = self.parody_text.toPlainText()
        self.project.theme = self.theme.text()
        self.project.lyric_style = self.style.currentText()
        self.project.llm_model = self.llm_model.text().strip()
        self.project.voice_model = self.voice_model_edit.text().strip()
        self.project.voice_index = self.voice_index_edit.text().strip()
        self.project.voice_source = self.voice_files.text().strip()
        self.project.voice_consent = self.voice_consent.isChecked()
        self.project.conversion_index_rate = self.index_rate.value()
        self.project.conversion_pitch_semitones = self.voice_pitch.value()
        self.project.conversion_protect = self.voice_protect.value()
        self.project.conversion_source = self.conversion_source.currentData()
        self.project.guide_vocal = self.guide_vocal_edit.text().strip()
        self.project.singing_engine = self.singing_engine.currentData()
        self.project.vocal_gain = self.vocal_gain.value() / 100
        self.project.instrumental_gain = self.instrumental_gain.value() / 100
        self.project.reverb = self.reverb.value() / 100
        self.project.segment_start = self.segment_start.value()
        self.project.segment_end = self.segment_end.value()

    def new_project(self):
        directory = QFileDialog.getExistingDirectory(self, "프로젝트 저장 폴더 선택", str(ROOT / "outputs"))
        if not directory:
            return
        project_path = Path(directory) / "project.json"
        if project_path.is_file():
            self.load_project(project_path)
            return
        self.project = Project(name=Path(directory).name, path=project_path)
        self.project.save()
        remember(self.project.path)
        self._refresh()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", str(ROOT / "outputs"), "Project (*.json)")
        if path:
            self.load_project(Path(path))

    def load_project(self, path: Path):
        try:
            self.project = Project.load(path)
            remember(path)
            self._refresh()
        except Exception as exc:
            QMessageBox.critical(self, "열기 실패", str(exc))

    def save_project(self):
        if self.project.path is None:
            self.new_project()
            if self.project.path is None:
                return
        self._sync()
        self.project.save()
        remember(self.project.path)
        self._refresh()
        self.statusBar().showMessage("저장했습니다", 4000)

    def choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "원곡 선택", "", "Audio (*.mp3 *.wav *.flac *.ogg)")
        if path:
            if path != self.project.source:
                self.project.stages.clear()
                self.project.analysis.clear()
                self.project.original_lyrics = ""
                self.original_text.clear()
            self.project.source = path
            self.source_path.setText(path)
            self.wave.set_file(Path(path))
            if self.project.path:
                self.save_project()

    def save_notes(self):
        if self.project.path is None:
            self.new_project()
            if self.project.path is None:
                return
        notes = []
        try:
            for row in range(self.notes_table.rowCount()):
                start = float(self.notes_table.item(row, 0).text())
                end = float(self.notes_table.item(row, 1).text())
                midi = int(self.notes_table.item(row, 2).text())
                if start < 0 or end <= start or not 24 <= midi <= 108:
                    raise ValueError(f"{row + 1}행의 값이 올바르지 않습니다")
                notes.append({"start": start, "end": end, "midi": midi})
            notes.sort(key=lambda note: note["start"])
            if any(notes[i]["end"] > notes[i + 1]["start"] for i in range(len(notes) - 1)):
                raise ValueError("음표가 서로 겹칩니다")
        except (AttributeError, ValueError) as exc:
            QMessageBox.warning(self, "음표 오류", str(exc))
            return
        self._sync()
        self.project.analysis["notes"] = notes
        self.project.analysis.pop("alignment", None)
        self.project.analysis.pop("lyric_fit", None)
        for stage in ("lyrics", "sing", "sing_segment", "convert", "mix"):
            self.project.stages.pop(stage, None)
        self.project.save()
        self._refresh()
        self.statusBar().showMessage(f"음표 {len(notes)}개 저장", 4000)

    def save_alignment(self):
        if self.project.path is None:
            self.statusBar().showMessage("프로젝트를 먼저 만드세요", 4000)
            return
        events = []
        try:
            for row in range(self.alignment_table.rowCount()):
                unit = self.alignment_table.item(row, 0).text().strip()
                start = float(self.alignment_table.item(row, 1).text())
                end = float(self.alignment_table.item(row, 2).text())
                midi = int(self.alignment_table.item(row, 3).text())
                if len(unit) != 1 or not "가" <= unit <= "힣" or start < 0 or end <= start or not 24 <= midi <= 108:
                    raise ValueError(f"{row + 1}행의 음절·시간·음정을 확인하세요")
                events.append({"syllable": unit, "start": start, "end": end, "midi": midi})
            if not events or any(events[i]["end"] > events[i + 1]["start"] for i in range(len(events) - 1)):
                raise ValueError("음절이 없거나 시간 구간이 겹칩니다")
        except (AttributeError, ValueError) as exc:
            QMessageBox.warning(self, "정렬 오류", str(exc))
            return
        from app.core.pipeline import _json
        self._sync()
        score_file = self.project.artifact("score.json")
        result = {"score": 100, "syllables": events,
                  "note_count": len(self.project.analysis.get("notes", [])), "edited_by_user": True}
        _json(score_file, result)
        self.project.analysis["alignment"] = events
        self.project.analysis["lyric_fit"] = "수동 편집"
        for stage in ("sing", "sing_segment", "convert", "mix"):
            self.project.stages.pop(stage, None)
        self.project.save()
        self._refresh()
        self.statusBar().showMessage(f"음절 {len(events)}개 저장", 4000)

    def choose_file(self, edit: QLineEdit, pattern: str):
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", str(ROOT / "external" / "RVC" / "assets"), f"Model ({pattern})")
        if path:
            edit.setText(path)

    def choose_voice_folder(self):
        path = QFileDialog.getExistingDirectory(self, "허락받은 말하기 음성 폴더")
        if path:
            self.voice_files.setText(path)

    def choose_voice_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "허락받은 말하기 음성 파일", "",
                                               "Audio (*.mp3 *.wav *.flac *.ogg *.m4a)")
        if path:
            self.voice_files.setText(path)

    def choose_recording(self):
        path, _ = QFileDialog.getOpenFileName(self, "직접 부른 보컬 파일", "",
                                              "Audio (*.wav *.mp3 *.flac *.ogg)")
        if path:
            self.guide_vocal_edit.setText(path)
            self.conversion_source.setCurrentIndex(self.conversion_source.findData("recording"))

    def train_voice(self):
        if not self.voice_consent.isChecked():
            QMessageBox.warning(self, "음성 권한", "명시적 사용 허락을 확인해 주세요.")
            return
        self.run("train")

    def generate_lyrics(self):
        self.run("generate")

    def play(self, value: str):
        path = Path(value) if value else None
        if not path or not path.is_file():
            self.statusBar().showMessage("재생할 파일이 없습니다", 4000)
            return
        if self.player.source().toLocalFile() == str(path) and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.player.play()

    def play_source(self):
        if not self.project.source:
            self.statusBar().showMessage("원곡 파일을 선택하세요", 4000)
            return
        source = Path(self.project.source)
        if self.project.path and self.project.stages.get("import", {}).get("status") == "완료":
            normalized = self.project.artifact("source.wav")
            if normalized.is_file():
                self.play(str(normalized))
                return
        if source.suffix.lower() == ".mp3" and source.is_file():
            with source.open("rb") as audio_file:
                header = audio_file.read(12)
            if header[4:8] == b"ftyp":
                self.statusBar().showMessage("이 파일은 MP3가 아닌 MP4입니다. '원곡 가져오기' 후 재생하세요", 7000)
                return
        self.play(str(source))

    def play_artifact(self, name: str):
        if self.project.path:
            self.play(str(self.project.artifact(name)))

    def play_segment(self):
        if not self.project.path or self.segment_end.value() <= self.segment_start.value():
            self.statusBar().showMessage("올바른 구간을 선택하세요", 4000)
            return
        target = self.project.artifact("converted.wav")
        if not target.is_file():
            self.statusBar().showMessage("변환 보컬이 없습니다", 4000)
            return
        self.player.setSource(QUrl.fromLocalFile(str(target)))
        self.player.setPosition(round(self.segment_start.value() * 1000))
        self.player.play()
        QTimer.singleShot(round((self.segment_end.value() - self.segment_start.value()) * 1000), self.player.pause)

    def run(self, stage: str):
        if self.thread and self.thread.isRunning():
            self.statusBar().showMessage("진행 중인 작업이 끝날 때까지 기다려 주세요", 4000)
            return
        if self.project.path is None:
            self.new_project()
            if self.project.path is None:
                return
        if stage in ("import", "separate", "transcribe", "analyze", "all") and not self.project.source:
            self.choose_source()
            if not self.project.source:
                self.statusBar().showMessage("원곡 파일을 선택하면 작업을 시작합니다", 5000)
                return
        self._sync()
        self.project.save()
        self.progress.setRange(0, 0)
        self.log.appendPlainText(f"▶ {LABELS[stage]}")
        if stage == "train":
            self._train_started = time.monotonic()
            self._train_phase = "준비 중"
            self.train_button.setEnabled(False)
            self.train_button.setText("학습 중…")
            self.train_progress.setRange(0, 0)
            self._train_timer.start(1000)
            self._update_train_status()
        self.thread = QThread(self)
        self._active_stage = stage
        self.worker = StageWorker(self.project.path, stage)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._show_progress)
        self.worker.stage_progress.connect(self._stage_tick)
        self.worker.finished.connect(self._done_result)
        self.worker.failed.connect(self._failed_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(str)
    def _done_result(self, result: str):
        if self._active_stage == "train":
            self._finish_train_status(True)
        self._done(self._active_stage, result)

    @Slot(str)
    def _failed_result(self, error: str):
        if self._active_stage == "train":
            self._finish_train_status(False)
        self._failed(self._active_stage, error)

    @Slot(str)
    def _show_progress(self, message: str):
        self.statusBar().showMessage(message)
        self.log.appendPlainText(message)
        if self._active_stage == "train":
            if message.startswith("=== "):
                self._train_phase = message.strip("= ")
                self.train_progress.setRange(0, 0)
            else:
                match = re.search(r"학습 에포크:\s*(\d+)\s*/\s*(\d+)", message)
                if match:
                    current, total = map(int, match.groups())
                    self._train_phase = f"RVC 학습 {current}/{total} 에포크"
                    self.train_progress.setRange(0, total)
                    self.train_progress.setValue(current)
            self._update_train_status()

    @Slot()
    def _update_train_status(self):
        if self._train_started is not None:
            elapsed = int(time.monotonic() - self._train_started)
            self.train_status.setText(f"{self._train_phase} · 경과 {elapsed // 60}분 {elapsed % 60}초")

    def _finish_train_status(self, success: bool):
        self._train_timer.stop()
        self._train_started = None
        self.train_button.setEnabled(True)
        self.train_button.setText("RVC 프로필 학습")
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(100 if success else 0)
        self.train_status.setText("학습 완료" if success else "학습 실패 — logs 폴더 확인")

    @Slot()
    def _thread_finished(self):
        self.thread = None
        self.worker = None
        self._active_stage = None

    def _stage_tick(self, current: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    def _done(self, stage: str, result: str):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.log.appendPlainText(f"✓ {result}")
        self.project = Project.load(self.project.path)
        self._refresh()
        self.statusBar().showMessage(f"{LABELS[stage]} 완료", 5000)

    def _failed(self, stage: str, error: str):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log.appendPlainText(error)
        self.project = Project.load(self.project.path)
        self._refresh()
        self.statusBar().showMessage(f"{LABELS[stage]} 실패 — logs/app.log 확인", 10000)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AI Parody Music Studio")
    font_file = Path("C:/Windows/Fonts/malgun.ttf")
    if font_file.is_file():
        QFontDatabase.addApplicationFont(str(font_file))
    app.setFont(QFont("Malgun Gothic", 10))
    window = Studio()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        window.load_project(Path(sys.argv[1]))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
