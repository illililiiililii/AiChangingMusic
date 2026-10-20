"""A mouse-editable piano roll for the monophonic source melody."""

from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import (QComboBox, QDialog, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout)
from app.music_analysis.midi_io import monophonic_draft


LOW, HIGH = 24, 108
ROW = 18
TOP = 30


class NoteItem(QGraphicsRectItem):
    def __init__(self, view: "PianoRollView", note: dict, index: int):
        super().__init__()
        self.view = view
        self.note = dict(note)
        self.index = index
        self.setBrush(QBrush(QColor("#53c4d0")))
        self.setPen(QPen(QColor("#d0faff"), 1))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(2)
        self.update_geometry()

    def update_geometry(self):
        start, end = self.note["start"], self.note["end"]
        self.setPos(self.view.time_x(start), self.view.pitch_y(self.note["midi"]))
        self.setRect(QRectF(0, 0, max(5, self.view.time_x(end - start)), ROW - 2))
        self.setToolTip(f"#{self.index + 1} {self.note.get('syllable', '')}  MIDI {self.note['midi']}  {start:.3f}–{end:.3f}초")

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.note.get("syllable") and self.rect().width() > 16:
            painter.setPen(QColor("#10202a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.note["syllable"])

    def mousePressEvent(self, event):
        self.view.select_note(self)
        self._original = dict(self.note)
        self._anchor = event.scenePos()
        self._resize = (bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                        or (self.rect().width() >= 20 and
                            event.pos().x() >= self.rect().width() - 9))
        event.accept()

    def mouseMoveEvent(self, event):
        dx = (event.scenePos().x() - self._anchor.x()) / self.view.pixels_per_second
        if self._resize:
            end = self.view.snap(self._original["end"] + dx)
            self.note["end"] = max(self.note["start"] + self.view.minimum_duration(), end)
        else:
            duration = self._original["end"] - self._original["start"]
            start = max(0., self.view.snap(self._original["start"] + dx))
            dy = round((event.scenePos().y() - self._anchor.y()) / ROW)
            self.note.update(start=start, end=start + duration,
                             midi=max(LOW, min(HIGH, self._original["midi"] - dy)))
        self.update_geometry()
        self.view.on_selection_changed()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.note["start"] = round(self.note["start"], 3)
        self.note["end"] = round(self.note["end"], 3)
        if self.note != self._original:
            self.view.changed()
        event.accept()


class PianoRollView(QGraphicsView):
    def __init__(self, dialog: "PianoRollDialog"):
        super().__init__()
        self.dialog = dialog
        self.scene_data = QGraphicsScene(self)
        self.setScene(self.scene_data)
        from PySide6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#14202c"))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.items_list: list[NoteItem] = []
        self.selected: NoteItem | None = None
        self.pixels_per_beat = 64
        self.bpm = 120.
        self.beat_offset = 0.
        self.beats_per_bar = 4
        self.grid_division = 4

    @property
    def pixels_per_second(self) -> float:
        return self.pixels_per_beat * self.bpm / 60.

    def time_x(self, seconds: float) -> float:
        return seconds * self.pixels_per_second

    def pitch_y(self, midi: int) -> float:
        return TOP + (HIGH - midi) * ROW

    def snap(self, seconds: float) -> float:
        if self.grid_division == 0:
            return seconds
        step = 60. / self.bpm / self.grid_division
        return self.beat_offset + round((seconds - self.beat_offset) / step) * step

    def minimum_duration(self) -> float:
        return 60. / self.bpm / self.grid_division if self.grid_division else .05

    def drawForeground(self, painter, rect):
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        left, top = visible.left(), visible.top()
        painter.fillRect(QRectF(left, top, visible.width(), TOP), QColor("#233646"))
        painter.fillRect(QRectF(left, top + TOP, 53, visible.height() - TOP), QColor("#263949"))
        painter.setPen(QColor("#e3edf2"))
        first = math.floor((left - self.time_x(self.beat_offset)) / self.pixels_per_beat)
        last = math.ceil((visible.right() - self.time_x(self.beat_offset)) / self.pixels_per_beat)
        for beat in range(first, last + 1):
            if beat % self.beats_per_bar:
                continue
            x = self.time_x(self.beat_offset) + beat * self.pixels_per_beat
            painter.drawText(QRectF(x + 3, top + 3, 100, TOP - 4),
                             f"{beat // self.beats_per_bar + 1} | "
                             f"{self.beat_offset + beat * 60 / self.bpm:.1f}s")
        for midi in range(LOW, HIGH + 1):
            y = self.pitch_y(midi)
            if visible.top() + TOP <= y <= visible.bottom() and midi % 12 == 0:
                painter.drawText(QRectF(left + 4, y, 45, ROW), f"C{midi // 12 - 1}")

    def redraw(self, notes: list[dict], *, preserve_scroll: bool = False):
        first_time = min((note["start"] for note in notes), default=0.)
        first_pitch = notes[0]["midi"] if notes else 72
        scroll_x = self.horizontalScrollBar().value() if preserve_scroll else max(0, self.time_x(first_time) - 90)
        scroll_y = self.verticalScrollBar().value() if preserve_scroll else self.pitch_y(first_pitch) - 200
        self.scene_data.clear()
        self.items_list = []
        self.selected = None
        ending = max((note["end"] for note in notes), default=30.) + 8.
        width = max(1500., self.time_x(ending))
        height = TOP + (HIGH - LOW + 1) * ROW
        self.scene_data.setSceneRect(0, 0, width, height)
        for midi in range(LOW, HIGH + 1):
            y = self.pitch_y(midi)
            if midi % 12 in (1, 3, 6, 8, 10):
                self.scene_data.addRect(0, y, width, ROW, QPen(Qt.PenStyle.NoPen),
                                        QBrush(QColor("#1b2938")))
            self.scene_data.addLine(0, y, width, y, QPen(QColor("#304353"), .5))
            if midi % 12 == 0:
                label = self.scene_data.addText(f"C{midi // 12 - 1}")
                label.setDefaultTextColor(QColor("#98adbb"))
                label.setPos(3, y - 2)
        beat_width = self.pixels_per_beat
        first_beat = math.floor(-self.time_x(self.beat_offset) / beat_width)
        for beat in range(first_beat, int(width / beat_width) + 2):
            x = self.time_x(self.beat_offset) + beat * beat_width
            if x < 0 or x > width:
                continue
            strong = beat % self.beats_per_bar == 0
            self.scene_data.addLine(x, TOP, x, height,
                                    QPen(QColor("#6b8795" if strong else "#344c5d"), 1))
            if strong:
                label = self.scene_data.addText(
                    f"{beat // self.beats_per_bar + 1} | {self.beat_offset + beat * 60 / self.bpm:.1f}s")
                label.setDefaultTextColor(QColor("#d4e2ea"))
                label.setPos(x + 3, 0)
        for index, note in enumerate(notes):
            item = NoteItem(self, note, index)
            self.scene_data.addItem(item)
            self.items_list.append(item)
        self.color_overlaps()
        self.horizontalScrollBar().setValue(scroll_x)
        self.verticalScrollBar().setValue(int(scroll_y))
        self.on_selection_changed()

    def color_overlaps(self):
        for item in self.items_list:
            item.setBrush(QBrush(QColor("#53c4d0")))
        ordered = sorted(self.items_list, key=lambda item: item.note["start"])
        for previous, current in zip(ordered, ordered[1:]):
            if previous.note["end"] > current.note["start"]:
                previous.setBrush(QBrush(QColor("#e99b5f")))
                current.setBrush(QBrush(QColor("#e99b5f")))

    def notes(self) -> list[dict]:
        return [dict(item.note) for item in self.items_list]

    def select_note(self, item: NoteItem):
        self.selected = item
        for other in self.items_list:
            other.setSelected(other is item)
        self.on_selection_changed()

    def on_selection_changed(self):
        item = self.selected
        self.dialog.selection_label.setText(
            f"선택 음표: MIDI {item.note['midi']} · {item.note['start']:.3f}–{item.note['end']:.3f}초"
            if item else "빈 곳 두 번 클릭: 추가 · 드래그: 시각/음높이 · 오른쪽 끝 또는 Alt+드래그: 길이")

    def changed(self):
        self.color_overlaps()
        self.dialog.remember()

    def mouseDoubleClickEvent(self, event):
        if not self.dialog.allow_note_count_changes:
            return super().mouseDoubleClickEvent(event)
        point = event.position().toPoint()
        if isinstance(self.itemAt(point), NoteItem):
            return super().mouseDoubleClickEvent(event)
        position = self.mapToScene(point)
        pitch = max(LOW, min(HIGH, HIGH - int((position.y() - TOP) // ROW)))
        start = max(0., self.snap(position.x() / self.pixels_per_second))
        duration = 60. / self.bpm
        note = {"start": round(start, 3), "end": round(start + duration, 3), "midi": pitch}
        item = NoteItem(self, note, len(self.items_list))
        self.scene_data.addItem(item)
        self.items_list.append(item)
        self.select_note(item)
        self.changed()
        event.accept()

    def keyPressEvent(self, event):
        if (self.dialog.allow_note_count_changes and
                event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected):
            self.scene_data.removeItem(self.selected)
            self.items_list.remove(self.selected)
            self.selected = None
            self.on_selection_changed()
            self.changed()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.dialog.undo()
        else:
            super().keyPressEvent(event)


class PianoRollDialog(QDialog):
    def __init__(self, notes: list[dict], bpm: float, parent=None,
                 beat_offset: float = 0., beats_per_bar: int = 4,
                 allow_note_count_changes: bool = True):
        super().__init__(parent)
        self.allow_note_count_changes = allow_note_count_changes
        self.setWindowTitle("멜로디 피아노롤 — 음표를 직접 고치세요")
        self.resize(1250, 780)
        self.selection_label = QLabel()
        self.view = PianoRollView(self)
        self.view.bpm = bpm
        self.view.beat_offset = beat_offset
        self.view.beats_per_bar = beats_per_bar
        self.history = [deepcopy(notes)]
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel(f"BPM {bpm:g} · {beats_per_bar}/4 · 첫 박 {beat_offset:g}초"))
        controls.addWidget(QLabel("스냅"))
        self.grid = QComboBox()
        for title, division in (("끔", 0), ("1박", 1), ("1/2박", 2),
                                ("1/4박", 4), ("1/8박", 8)):
            self.grid.addItem(title, division)
        self.grid.setCurrentIndex(3)
        self.grid.currentIndexChanged.connect(self._grid_changed)
        controls.addWidget(self.grid)
        controls.addWidget(QLabel("가로 확대"))
        zoom = QSlider(Qt.Orientation.Horizontal)
        zoom.setRange(28, 150)
        zoom.setValue(64)
        zoom.setMaximumWidth(180)
        zoom.valueChanged.connect(self._zoom_changed)
        controls.addWidget(zoom)
        undo = QPushButton("실행 취소")
        undo.clicked.connect(self.undo)
        controls.addWidget(undo)
        delete = QPushButton("선택 음표 삭제")
        delete.clicked.connect(self.delete_selected)
        delete.setEnabled(allow_note_count_changes)
        controls.addWidget(delete)
        simplify = QPushButton("겹친 음표 초안 정리")
        simplify.clicked.connect(self.simplify)
        simplify.setEnabled(allow_note_count_changes)
        controls.addWidget(simplify)
        controls.addStretch()
        layout.addLayout(controls)
        layout.addWidget(self.view)
        layout.addWidget(self.selection_label)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("음표를 저장한 뒤 보컬과 믹싱을 다시 생성하세요."))
        bottom.addStretch()
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        save = QPushButton("이 음표 사용")
        save.clicked.connect(self.accept)
        bottom.addWidget(cancel)
        bottom.addWidget(save)
        layout.addLayout(bottom)
        self.view.redraw(notes)

    def _grid_changed(self):
        self.view.grid_division = self.grid.currentData()

    def _zoom_changed(self, value: int):
        notes = self.view.notes()
        self.view.pixels_per_beat = value
        self.view.redraw(notes, preserve_scroll=True)

    def remember(self):
        self.history.append(deepcopy(self.view.notes()))
        self.history = self.history[-60:]

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.view.redraw(deepcopy(self.history[-1]), preserve_scroll=True)

    def delete_selected(self):
        if self.view.selected:
            self.view.scene_data.removeItem(self.view.selected)
            self.view.items_list.remove(self.view.selected)
            self.view.selected = None
            self.view.on_selection_changed()
            self.remember()

    def simplify(self):
        self.view.redraw(monophonic_draft(self.view.notes()), preserve_scroll=True)
        self.remember()

    def notes(self) -> list[dict]:
        return sorted(self.view.notes(), key=lambda note: (note["start"], note["midi"]))
