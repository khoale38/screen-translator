"""Translates many separate pieces of text at once, one llama.cpp sequence each.

A GPU step for several sequences costs little more than a step for one, so a screen of short pieces takes about as
long as its longest piece instead of the sum of them all.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import time

import numpy as np

OnStep = Callable[[int, list[int], bool], None]

# llama.cpp's fast matrix-vector kernels (CUDA, Vulkan, Metal) take at most 8 rows; a 9th switches every layer to
# the general matrix kernels, which are several times slower per step on NVIDIA cards.
MAX_ACTIVE = 8


class ParallelDecoder:
    """Greedy decoding of several prompts side by side in a context of its own.

    The instructions every prompt starts with are evaluated once and kept in a spare sequence: hybrid models such as
    Qwen3.5 cannot rewind to a shared prefix, but a whole sequence can be copied.
    """

    def __init__(self, llm, max_sequences: int = 12, n_ctx: int = 4096, n_batch: int = 1024) -> None:
        import llama_cpp as lib

        self._lib = lib
        self._llm = llm
        params = lib.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = min(512, n_batch)
        params.n_seq_max = max_sequences + 1
        params.kv_unified = True
        self._ctx = lib.llama_init_from_model(llm.model, params)
        if not self._ctx:
            raise RuntimeError("llama.cpp could not create a context for parallel translation")
        self._memory = lib.llama_get_memory(self._ctx)
        self._n_batch = n_batch
        self._n_vocab = llm.n_vocab()
        self.max_sequences = max_sequences
        self._spare = max_sequences
        self._cached: tuple[int, ...] = ()
        self.timing: dict[str, int] = {}

    def close(self) -> None:
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None

    def _decode(self, entries: list[tuple[int, int, int, bool]]) -> None:
        lib = self._lib
        for start in range(0, len(entries), self._n_batch):
            chunk = entries[start:start + self._n_batch]
            batch = lib.llama_batch_init(len(chunk), 0, 1)
            try:
                for i, (token, pos, seq, wants_logits) in enumerate(chunk):
                    batch.token[i], batch.pos[i], batch.logits[i] = token, pos, wants_logits
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = seq
                batch.n_tokens = len(chunk)
                if lib.llama_decode(self._ctx, batch) != 0:
                    raise RuntimeError("llama.cpp could not decode the batch")
            finally:
                lib.llama_batch_free(batch)

    def _next_token(self, index: int) -> int:
        logits = np.ctypeslib.as_array(self._lib.llama_get_logits_ith(self._ctx, index), shape=(self._n_vocab,))
        return int(np.argmax(logits))

    def _clear(self, seq: int) -> None:
        self._lib.llama_memory_seq_rm(self._memory, seq, -1, -1)

    def run(self, prompts: list[list[int]], instructions: int, budgets: list[int], stop: set[int],
            on_step: OnStep) -> list[list[int]]:
        """Decode every prompt; `instructions` leading tokens are shared by all and worth keeping for next time.

        `on_step(index, tokens_so_far, finished)` is called as each output grows; raising from it stops decoding.
        """
        lib = self._lib
        started = time.monotonic()
        count = len(prompts)
        if not 0 < count <= self.max_sequences:
            raise ValueError(f"between 1 and {self.max_sequences} prompts at a time")
        common = 0
        while all(len(p) > common + 1 for p in prompts) and len({p[common] for p in prompts}) == 1:
            common += 1
        instructions = min(instructions, common)
        key = tuple(prompts[0][:instructions])
        try:
            if key != self._cached:
                self._clear(self._spare)
                self._cached = ()
                self._decode([(token, pos, self._spare, False) for pos, token in enumerate(key)])
                self._cached = key
            if key:
                lib.llama_memory_seq_cp(self._memory, self._spare, 0, -1, -1)
            self._decode([(prompts[0][pos], pos, 0, False) for pos in range(instructions, common)])
            for seq in range(1, count):
                lib.llama_memory_seq_cp(self._memory, 0, seq, -1, -1)

            own = [(prompt[pos], pos, seq, pos == len(prompt) - 1)
                   for seq, prompt in enumerate(prompts) for pos in range(common, len(prompt))]
            if len(own) > self._n_batch:
                self._decode([entry for entry in own if not entry[3]])
                own = [entry for entry in own if entry[3]]
            self._decode(own)
            upcoming = {seq: self._next_token(row) for row, (_t, _p, seq, last) in enumerate(own) if last}
            listed = time.monotonic()
            self.timing = {"list": round((listed - started) * 1000), "steps": 0, "step": 0}
            outputs: list[list[int]] = [[] for _ in prompts]
            positions = [len(p) for p in prompts]
            waiting = deque(sorted(range(count), key=lambda seq: -budgets[seq]))
            active: list[int] = []
            while waiting or active:
                while waiting and len(active) < MAX_ACTIVE:
                    active.append(waiting.popleft())
                step = []
                for seq in active:
                    token = upcoming.pop(seq)
                    done = token in stop or len(outputs[seq]) >= budgets[seq]
                    if not done:
                        outputs[seq].append(token)
                    on_step(seq, outputs[seq], done)
                    if not done:
                        step.append((token, positions[seq], seq, True))
                        positions[seq] += 1
                active = [seq for _token, _pos, seq, _ in step]
                if step:
                    self._decode(step)
                    upcoming.update((seq, self._next_token(row)) for row, (_t, _p, seq, _l) in enumerate(step))
                    steps = self.timing["steps"] + 1
                    self.timing.update(steps=steps, step=round((time.monotonic() - listed) * 1000 / steps))
            return outputs
        finally:
            for seq in range(count):
                self._clear(seq)
