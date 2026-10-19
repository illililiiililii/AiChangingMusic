"""Portable project metadata; all generated media remains local and ignored by Git."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


STAGES = ("import", "separate", "transcribe", "analyze", "generate", "lyrics", "train", "sing", "sing_segment", "convert", "mix")


@dataclass
class Project:
    name: str = "새 프로젝트"
    source: str = ""
    source_language: str = "ko"
    original_lyrics: str = ""
    parody_lyrics: str = ""
    theme: str = ""
    lyric_style: str = "유머"
    llm_model: str = "qwen3.5:4b"
    voice_model: str = ""
    voice_index: str = ""
    voice_source: str = ""
    voice_consent: bool = False
    conversion_index_rate: float = 0.25
    conversion_pitch_semitones: int = 0
    conversion_protect: float = 0.2
    conversion_source: str = "synth"
    guide_vocal: str = ""
    singing_engine: str = "korean_tts"
    vocal_gain: float = 1.0
    instrumental_gain: float = 0.8
    reverb: float = 0.08
    segment_start: float = 0.0
    segment_end: float = 10.0
    stages: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    path: Path | None = field(default=None, repr=False)

    @property
    def directory(self) -> Path:
        if self.path is None:
            raise ValueError("프로젝트 저장 위치를 먼저 선택하세요.")
        return self.path.parent

    def artifact(self, name: str) -> Path:
        return self.directory / name

    def save(self, path: Path | None = None) -> None:
        if path is not None:
            self.path = Path(path).resolve()
        if self.path is None:
            raise ValueError("프로젝트 경로가 없습니다.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: value for key, value in vars(self).items() if key != "path"}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    @classmethod
    def load(cls, path: Path) -> "Project":
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys() - {"path"}
        return cls(**{key: value for key, value in data.items() if key in allowed}, path=path)

    def mark(self, stage: str, status: str, message: str = "") -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        self.stages[stage] = {"status": status, "message": message}
        self.save()
