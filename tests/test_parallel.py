from types import SimpleNamespace

from screen_translator.parallel import MAX_ACTIVE, ParallelDecoder

STOP = 0


class ScriptedDecoder(ParallelDecoder):
    """The scheduling of ParallelDecoder, with each prompt's reply scripted instead of computed by a model."""

    def __init__(self, replies: dict[int, list[int]]) -> None:
        self._lib = SimpleNamespace(llama_memory_seq_cp=lambda *a: None, llama_memory_seq_rm=lambda *a: None)
        self._memory = None
        self.max_sequences = 12
        self._n_batch = 1024
        self._spare = 12
        self._cached = ()
        self.timing = {}
        self.replies = replies
        self.step_sizes: list[int] = []
        self._rows: list[tuple[int, int] | None] = []

    def _decode(self, entries):
        if entries and all(pos >= self.prompt_length for _token, pos, _seq, _logits in entries):
            self.step_sizes.append(len(entries))
        self._rows = [(seq, pos) if logits else None for _token, pos, seq, logits in entries]

    def _next_token(self, index):
        seq, pos = self._rows[index]
        reply = self.replies[seq]
        made = pos - self.prompt_length + 1
        return reply[made] if made < len(reply) else STOP

    def run(self, prompts, *args, **kwargs):
        self.prompt_length = len(prompts[0])
        return super().run(prompts, *args, **kwargs)


def test_at_most_eight_pieces_are_decoded_at_once_and_the_rest_start_as_slots_free_up():
    replies = {seq: [100 + seq] * (2 + seq) for seq in range(12)}
    decoder = ScriptedDecoder(replies)
    prompts = [[1, 2, 3, 50 + seq] for seq in range(12)]
    finished = []
    outputs = decoder.run(prompts, 3, [64] * 12, {STOP}, lambda seq, tokens, done: done and finished.append(seq))
    assert outputs == [replies[seq] for seq in range(12)]
    assert max(decoder.step_sizes) == MAX_ACTIVE
    assert sorted(finished) == list(range(12))
    assert decoder.timing["steps"] == len(decoder.step_sizes)


def test_the_longest_pieces_start_first():
    decoder = ScriptedDecoder({seq: [7] * 3 for seq in range(10)})
    started = []
    budgets = [5, 5, 5, 5, 5, 5, 5, 5, 90, 80]
    decoder.run([[1, 2, 3, seq] for seq in range(10)], 3, budgets, {STOP},
                lambda seq, tokens, done: seq not in started and started.append(seq))
    assert started[:2] == [8, 9]
