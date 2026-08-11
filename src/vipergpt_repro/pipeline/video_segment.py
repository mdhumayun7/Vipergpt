"""VideoSegment — the video API surface (NExT-QA).

Wraps a [T, 3, H, W] frame stack with temporal metadata and yields per-frame
ImagePatch objects. `select_answer` delegates to the text LLM for multiple-choice.
"""
from __future__ import annotations

from collections.abc import Iterator

from vipergpt_repro.pipeline.image_patch import ImagePatch


class VideoSegment:
    def __init__(self, video, start: int = 0, end: int | None = None, bus=None, parent_start: int = 0):
        self.video = video  # [T, 3, H, W]
        self.T = video.shape[0]
        self.start = start
        self.end = end if end is not None else self.T
        self._bus = bus
        self.num_frames = self.end - self.start

    def frame_iterator(self) -> Iterator[ImagePatch]:
        for i in range(self.start, self.end):
            yield ImagePatch(self.video[i], bus=self._bus)

    def frame_from_index(self, index: int) -> ImagePatch:
        return ImagePatch(self.video[min(max(index, 0), self.T - 1)], bus=self._bus)

    def trim(self, start: int | None = None, end: int | None = None) -> VideoSegment:
        start = self.start if start is None else max(self.start, start)
        end = self.end if end is None else min(self.end, end)
        return VideoSegment(self.video, start, end, bus=self._bus)

    def select_answer(self, info: str, question: str, options) -> int:
        return int(self._bus.call("select_answer", info, question, options))

    def __repr__(self) -> str:
        return f"VideoSegment(start={self.start}, end={self.end}, T={self.T})"
