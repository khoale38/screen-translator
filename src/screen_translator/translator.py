from __future__ import annotations

import re
from typing import Callable, Protocol

from . import languages
from .context import Turn

_THINK = re.compile(r"<think>.*?</think>", re.S)


def build_messages(
    current: str,
    history: list[Turn],
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str] | None = None,
    notes: str = "",
    *,
    address=None,
    budget: str = "",
    aside: str = "",
) -> list[dict]:
    target = languages.name(target_lang)
    source = "whatever language it is in" if source_lang == "auto" else languages.name(source_lang)
    system = (
        f"You translate on-screen text (subtitles, game dialogue, UI, chat) from {source} into {target}, live.\n"
        f"Rules:\n"
        f"- Reply with only the {target} translation of the latest message. No notes, quotes, romanization or alternatives.\n"
        f"- Keep character names as names: write them in their usual Latin spelling (ユキ is Yuki, 김민수 is Kim Min-su) "
        f"and never translate their meaning.\n"
        f"- Earlier messages are the lines that were on screen just before, with your translations. Use them to keep "
        f"names, pronouns, gender, tone, terminology and each speaker's voice consistent.\n"
        f"- Translate what the line means, not word by word. Idioms, slang, game objectives and film dialogue must "
        f"become what a native {target} speaker would actually say in that situation (e.g. \"hit the books\" means "
        f"study hard, not strike books). Keep it as short and punchy as the original.\n"        f"- Sound like a native {target} speaker talking, not a textbook: casual lines stay casual, with the "
        f"particles, interjections, slang and swearing people really use; formal lines stay formal. Insults and "
        f"curses keep their force.\n"
        f"- Keep controller buttons, keys and on-screen labels such as L1, R2, X, Enter or Esc exactly as written, "
        f"and keep holding a button distinct from pressing or tapping it.\n"
        f"- Keep every number, time, price and stat abbreviation (HP, MP, Lv, ATK, EXP) exactly as written; never "
        f"change, add or drop a digit.\n"
        f"- Write complete, natural {target}. When the source leaves out a subject or object (common in Japanese, "
        f"Korean and Chinese), work out who or what is meant from the earlier lines and state it.\n"
        f"- If the latest message continues the previous one, translate it so it reads naturally after it.\n"
        f"- If the first line is only a speaker's name, reply as \"Name: translation\" on one line.\n"
        f"- If the text is already in {target}, repeat it unchanged.\n"
        f"- Choose pronouns and forms of address from who is talking to whom: their gender, age, rank, closeness "
        f"and mood, worked out from names, the earlier lines and the background notes. A line usually answers the "
        f"one before it, so its listener is usually the previous speaker. Once two characters address each other "
        f"one way, keep it unless their relationship changes."
    )
    if rules := languages.ADDRESS.get(target_lang):
        system += f"\n- {rules}"
    if glossary:
        terms = "\n".join(f"- {src} → {dst}" for src, dst in glossary.items() if src.strip())
        system += f"\n\nAlways translate these terms exactly as given:\n{terms}"
    if notes.strip():
        system += f"\n\nBackground on what is being shown:\n{notes.strip()}"

    # The plain facts ride along for translation models that take their own short prompt instead (see for_hy_mt).
    terms = address.terms(source_lang) if address else {}
    messages = [{"role": "system", "content": system, "target": target, "glossary": dict(glossary or {}),
                 "notes": notes.strip(), "address_terms": terms, "budget": budget,
                 "address_zh": (f"{address.speaker}对{address.listener}说话，翻译成越南语时，{address.speaker}自称"
                                f"“{address.me}”，称对方“{address.you}”。") if address else ""}]
    for turn in history:
        messages.append(_asking(turn.source, turn.note))
        messages.append({"role": "assistant", "content": turn.translation})
    note = " ".join(filter(None, (aside, address.note() if address else "", dub_note(budget))))
    messages.append(_asking(current, note))
    return messages


def dub_note(budget: str) -> str:
    """Speech that will be dubbed must fit the time its speaker took, as a simultaneous interpreter's would."""
    return (f"This line will be dubbed over the speaker, so it must be said in the time they took: at most "
            f"{budget}. If a faithful translation fits, give that; if not, condense like a simultaneous interpreter: "
            f"keep the meaning, names and tone, drop only filler, repetition and hedging.") if budget else ""


def _asking(text: str, note: str) -> dict:
    return {"role": "user", "content": with_note(text, note), "item": text} if note else {"role": "user", "content": text}


def with_note(text: str, note: str) -> str:
    """The line with how its speaker addresses the listener; history keeps the same text so saved state still fits."""
    return f"{text}\n\n(Translator's note: {note})" if note else text


_NUMBERED = re.compile(r"^\s*(\d+)\s*[.)、:：]\s*(.*?)\s*$")


def build_batch_messages(
    texts: list[str],
    history: list[Turn],
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str] | None = None,
    notes: str = "",
) -> list[dict]:
    """One request for many separate pieces of on-screen text; reading the prompt once is most of the speed-up."""
    numbered = "\n".join(f"{i}. {' / '.join(text.splitlines())}" for i, text in enumerate(texts, 1))
    messages = build_messages(numbered, history, source_lang, target_lang, glossary, notes)
    messages[0]["content"] += (
        "\n\nThe latest message is a numbered list of separate pieces of text seen on one screen. Translate each "
        "one on its own, using the others only as context. Reply with the same numbers, one line each, nothing else."
    )
    return messages


SCREEN_LETTERS = "ABCDEFGHIJKL"
SCREEN_NOTE = ("\n\nThe latest message lists the text seen on one screen, each item marked with a letter. Translate "
               "only the item it asks for, using the others as context, and reply with just that translation.")


def build_screen_messages(
    index: int,
    screen: list[str],
    history: list[Turn],
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str] | None = None,
    notes: str = "",
) -> list[dict]:
    """Item `index` of a screen's text, to be translated on its own with the whole screen as context.

    Every item of a screen gets the same prompt up to the closing "Translate item C.", so all of them share that
    prompt and are decoded side by side.
    """
    messages = build_messages("", history, source_lang, target_lang, glossary, notes)
    messages[0]["content"] += SCREEN_NOTE
    listing = "\n".join(f"{SCREEN_LETTERS[i]}. {' / '.join(text.splitlines())}" for i, text in enumerate(screen))
    messages[-1].update(content=f"{listing}\n\nTranslate item {SCREEN_LETTERS[index]}.", item=screen[index],
                        screen=screen)
    return messages


def for_hy_mt(messages: list[dict]) -> list[dict]:
    """The same request in Tencent HY-MT's own prompt format, which it follows far better than a system prompt.

    https://huggingface.co/tencent/HY-MT1.5-1.8B#prompts. The text is also listed above the instruction: on its
    own, a short line loses its subject ("ユキはまだ？" became "hasn't returned yet?").
    """
    facts, last = messages[0], messages[-1]
    text = last.get("item", last["content"])
    glossary = {**facts.get("address_terms", {}), **facts.get("glossary", {})}
    terms = "".join(f"{src} 翻译成 {dst}\n" for src, dst in glossary.items() if src.strip())
    parts = [f"参考下面的翻译：\n{terms}"] if terms else []
    background = "\n".join(filter(None, (facts.get("notes"), facts.get("address_zh"))))
    if background:
        parts.append(f"背景：{background}")
    turns = messages[1:-1]
    if turns:
        parts.append("前面的对话：\n" + "\n".join(f"{asked.get('item', asked['content'])}\n{answered['content']}"
                                               for asked, answered in zip(turns[::2], turns[1::2])))
    # Shown a whole screen, the 1.8B model translates the wrong item or all of them, so it sees only its own.
    parts.append("屏幕上的文字：\n" + text)
    brief = ""
    if facts.get("budget"):
        count, unit = facts["budget"].split()
        brief = f"译文用于配音，要像同声传译一样精简口语化，不超过{count}个{'字' if unit == 'characters' else '词'}，"
    prompt = "\n".join(parts) + (f"\n参考上面的信息，把下面的文本翻译成{facts['target']}，{brief}注意不需要翻译上文，"
                                  f"也不要额外解释：\n{text}")
    return [{"role": "user", "content": prompt}]


def strip_item_letter(text: str, index: int) -> str:
    """Small models sometimes repeat the item's letter ("C. Settings"); the translation is what follows it."""
    return re.sub(rf"^\s*{SCREEN_LETTERS[index]}\s*[.):]\s*", "", text)


def parse_numbered(raw: str, count: int, complete: bool = True) -> dict[int, str]:
    """Numbered lines of a batch reply; without `complete` the last line may still be streaming and is left out."""
    lines = raw.splitlines()
    if not complete and lines and not raw.endswith("\n"):
        lines = lines[:-1]
    found = {}
    for line in lines:
        match = _NUMBERED.match(line)
        if match and 1 <= int(match.group(1)) <= count and match.group(2):
            found[int(match.group(1))] = match.group(2)
    return found


def clean_output(raw: str) -> str:
    text = _THINK.sub("", raw)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def label_speaker(source: str, translation: str) -> str:
    """Models often answer a "Name\nline" source as two lines; subtitles read better as "Name: line"."""
    label, _, rest = source.partition("\n")
    if not rest or len(label) > 12:
        return translation
    head, _, body = translation.partition("\n")
    head = head.strip().rstrip(":：")
    if body.strip() and 0 < len(head) <= 24 and not head.endswith((".", "!", "?")):
        return f"{head}: {' '.join(body.split())}"
    return translation


def _stops(formatted) -> list[str]:
    # Some models' end token has no text (HY-MT); an empty stop string would end every reply before it starts.
    return [stop for stop in formatted.stop if stop]


class Translator(Protocol):
    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str: ...


class LlamaTranslator:
    """llama.cpp backed translator.

    Qwen3.5's hybrid layers cannot be trimmed back to a shared prompt start, so llama.cpp would re-read the whole
    prompt for every line. Instead the state after the instructions, and after each line's message, is saved and
    restored, so a new line only reads itself and the previous line's translation.
    """

    max_snapshots = 3
    max_turn_snapshots = 2
    min_shared_tokens = 64

    def __init__(self, model_path: str, gpu_layers: int = -1, n_ctx: int = 4096) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        from .devices import model_device

        self._llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=n_ctx, verbose=False)
        self.device = model_device(gpu_layers)
        token_text = lambda token: self._llm.detokenize([token], special=True).decode("utf-8", "ignore")
        self._format = Jinja2ChatFormatter(
            self._llm.metadata["tokenizer.chat_template"],
            eos_token=token_text(self._llm.token_eos()),
            bos_token=token_text(self._llm.token_bos()),
        )
        # Sampling picks a different pronoun from line to line; the likeliest word keeps characters consistent.
        self.temperature = 0.0
        self._hy_mt = self._llm.metadata.get("general.architecture", "").startswith("hunyuan")
        self._snapshots: dict[tuple[int, ...], object] = {}
        self._turn_snapshots: dict[tuple[int, ...], object] = {}
        self._parallel = None

    def close(self) -> None:
        self._snapshots.clear()
        self._turn_snapshots.clear()
        if self._parallel is not None:
            self._parallel.close()
        self._llm.close()

    @property
    def timing(self) -> dict[str, int]:
        return self._parallel.timing if self._parallel is not None else {}

    def translate_each(self, conversations: list[list[dict]], lengths: list[int],
                       on_piece: Callable[[int, str, bool], None] | None = None) -> list[str]:
        """Translate several pieces at once; `lengths` are the source lengths, which cap each output."""
        from .parallel import ParallelDecoder

        if self._parallel is None:
            self._parallel = ParallelDecoder(self._llm)
        if self._hy_mt:
            conversations = [for_hy_mt(messages) for messages in conversations]
        prompts = [self._tokens(messages)[1] for messages in conversations]
        instructions = self._shared(prompts[0], conversations[0][:1] + [{"role": "user", "content": "\u2063"}])
        stop = {self._llm.token_eos()}
        for marker in (b"<|im_end|>", b"<end_of_turn>", b"<|endoftext|>"):
            token = self._llm.tokenize(marker, add_bos=False, special=True)
            if len(token) == 1:
                stop.add(token[0])
        text = lambda tokens: clean_output(self._llm.detokenize(tokens).decode("utf-8", "ignore"))
        shown: dict[int, str] = {}

        def step(index: int, tokens: list[int], done: bool) -> None:
            current = text(tokens)
            if on_piece and (done or current != shown.get(index)):
                shown[index] = current
                on_piece(index, current, done)

        outputs = self._parallel.run(prompts, instructions, [24 + 3 * n for n in lengths], stop, step)
        return [text(tokens) for tokens in outputs]

    def _tokens(self, messages: list[dict]):
        # Reasoning models (Qwen3/3.5) honour this flag in their own template; others ignore it.
        formatted = self._format(messages=messages, enable_thinking=False)
        return formatted, self._llm.tokenize(formatted.prompt.encode("utf-8"), add_bos=False, special=True)

    def _shared(self, tokens: list[int], messages: list[dict]) -> int:
        """How many leading tokens of `tokens` the prompt for `messages` would start with too."""
        _, probe = self._tokens(messages)
        return next((i for i, (a, b) in enumerate(zip(tokens, probe)) if a != b), min(len(tokens), len(probe)))

    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str:
        budget = min(1024, 64 + 3 * len(messages[-1].get("item", messages[-1]["content"])))
        if self._hy_mt:
            # A plain transformer: llama.cpp reuses the shared start of the prompt by itself.
            messages = for_hy_mt(messages)
        formatted, tokens = self._tokens(messages)
        if not self._hy_mt:
            self._resume(tokens)
            self._save_turn(messages, tokens)
        raw = ""
        shown = ""
        for chunk in self._llm.create_completion(
            tokens, stream=True, temperature=self.temperature, top_p=0.9, max_tokens=budget, stop=_stops(formatted),
        ):
            raw += chunk["choices"][0]["text"]
            visible = clean_output(raw)
            if on_partial and visible != shown:
                shown = visible
                on_partial(visible)
        if not self._hy_mt:
            self._save_instructions(messages, tokens)
        return clean_output(raw)

    def answer(self, question: str, max_tokens: int = 120) -> str:
        """A one-off question with no history, for jobs other than translating."""
        _formatted, tokens = self._tokens([{"role": "user", "content": question}])
        if not self._holds(tokens):
            self._llm.reset()
        result = self._llm.create_completion(tokens, max_tokens=max_tokens, temperature=0.0)
        return clean_output(result["choices"][0]["text"])

    def warm_up(self, messages: list[dict]) -> None:
        if self._hy_mt:
            messages = for_hy_mt(messages)
        formatted, tokens = self._tokens(messages)
        for _chunk in self._llm.create_completion(tokens, max_tokens=4, temperature=0.0, stop=_stops(formatted)):
            pass
        if not self._hy_mt:
            self._save_instructions(messages, tokens)

    def _snapshot(self):
        # Llama.save_state also copies a 500 MB table of scores this app never reads; the sequence alone takes ms.
        import ctypes

        import llama_cpp as lib

        ctx = self._llm._ctx.ctx
        size = lib.llama_state_seq_get_size(ctx, 0)
        buffer = (ctypes.c_uint8 * size)()
        used = lib.llama_state_seq_get_data(ctx, buffer, size, 0)
        return (ctypes.c_uint8 * used).from_buffer_copy(buffer, 0) if used < size else buffer

    def _restore(self, key: tuple[int, ...], state) -> None:
        import llama_cpp as lib

        ctx = self._llm._ctx.ctx
        lib.llama_memory_seq_rm(lib.llama_get_memory(ctx), 0, -1, -1)
        if lib.llama_state_seq_set_data(ctx, state, len(state), 0) != len(state):
            self._llm.reset()
            return
        self._llm.input_ids[: len(key)] = key
        self._llm.n_tokens = len(key)
        self._llm._requires_eval = True

    def _holds(self, tokens: list[int] | tuple[int, ...]) -> bool:
        """Whether the model's memory is exactly a start of `tokens`, so reading on from it is safe."""
        n = self._llm.n_tokens
        return n <= len(tokens) and list(self._llm.input_ids[:n]) == list(tokens[:n])

    def _resume(self, tokens: list[int]) -> None:
        saved = [(store, key) for store in (self._snapshots, self._turn_snapshots) for key in store
                 if len(key) < len(tokens) and tuple(tokens[: len(key)]) == key]
        if not saved:
            return
        store, key = max(saved, key=lambda item: len(item[1]))
        if self._holds(tokens) and self._llm.n_tokens >= len(key):
            return
        store[key] = store.pop(key)
        self._restore(key, store[key])

    def _evaluate_to(self, key: tuple[int, ...]) -> None:
        if not self._holds(key):
            self._llm.reset()
        self._llm.eval(list(key[self._llm.n_tokens:]))

    def _save_turn(self, messages: list[dict], tokens: list[int]) -> None:
        """Save the state at the end of this line's message, where the next line's prompt will part from this one."""
        later = [messages + [{"role": "assistant", "content": filler}, {"role": "user", "content": filler}]
                 for filler in ("\u2063", "A")]
        shared = min(self._shared(tokens, probe) for probe in later)
        key = tuple(tokens[:shared])
        if shared < self.min_shared_tokens or shared >= len(tokens) or key in self._turn_snapshots:
            return
        self._evaluate_to(key)
        self._turn_snapshots[key] = self._snapshot()
        while len(self._turn_snapshots) > self.max_turn_snapshots:
            self._turn_snapshots.pop(next(iter(self._turn_snapshots)))

    def _save_instructions(self, messages: list[dict], tokens: list[int]) -> None:
        """Snapshot the prompt up to the first message, after this line so it is not delayed; once per instruction set."""
        shared = self._shared(tokens, messages[:1] + [{"role": "user", "content": "\u2063"}])
        key = tuple(tokens[:shared])
        if shared < self.min_shared_tokens or key in self._snapshots:
            return
        self._evaluate_to(key)
        self._snapshots[key] = self._snapshot()
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(next(iter(self._snapshots)))
