from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class Turn:
    source: str
    translation: str
    at: float


class ContextWindow:
    def __init__(self, max_turns: int = 8, ttl_s: float = 300.0, clock: Callable[[], float] = time.monotonic):
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self._clock = clock
        self._turns: deque[Turn] = deque()
        self._lock = threading.Lock()

    def configure(self, max_turns: int, ttl_s: float) -> None:
        with self._lock:
            self.max_turns, self.ttl_s = max_turns, ttl_s
            self._prune()

    def _prune(self) -> None:
        cutoff = self._clock() - self.ttl_s
        while self._turns and self._turns[0].at < cutoff:
            self._turns.popleft()
        if len(self._turns) > self.max_turns:
            # Dropping half at once keeps the start of the prompt, and so the saved model state, for several lines.
            while len(self._turns) > max(1, self.max_turns // 2):
                self._turns.popleft()

    def turns(self) -> list[Turn]:
        with self._lock:
            self._prune()
            return list(self._turns)

    def add(self, source: str, translation: str, replace_last: bool = False) -> None:
        with self._lock:
            if replace_last and self._turns:
                self._turns.pop()
            self._turns.append(Turn(source, translation, self._clock()))
            self._prune()

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
