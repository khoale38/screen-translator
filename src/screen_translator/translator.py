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
) -> list[dict]:
    target = languages.name(target_lang)
    source = "whatever language it is in" if source_lang == "auto" else languages.name(source_lang)
    system = (
        f"You translate on-screen text (subtitles, game dialogue, UI, chat) from {source} into {target}, live.\n"
        f"Rules:\n"
        f"- Reply with only the {target} translation of the latest message. No notes, quotes, romanization or alternatives.\n"
        f"- Earlier messages are the lines that were on screen just before, with your translations. Use them to keep "
        f"names, pronouns, gender, tone, terminology and each speaker's voice consistent.\n"
        f"- Translate what the line means, not word by word. Idioms, slang, game objectives and film dialogue must "
        f"become what a native {target} speaker would actually say in that situation (e.g. \"hit the books\" means "
        f"study hard, not strike books). Keep it as short and punchy as the original.\n"
        f"- Keep controller buttons, keys and on-screen labels such as L1, R2, X, Enter or Esc exactly as written, "
        f"and keep holding a button distinct from pressing or tapping it.\n"
        f"- Write complete, natural {target}. When the source leaves out a subject or object (common in Japanese, "
        f"Korean and Chinese), work out who or what is meant from the earlier lines and state it.\n"
        f"- If the latest message continues the previous one, translate it so it reads naturally after it.\n"
        f"- If the first line is only a speaker's name, reply as \"Name: translation\" on one line.\n"
        f"- If the text is already in {target}, repeat it unchanged."
    )
    if glossary:
        terms = "\n".join(f"- {src} → {dst}" for src, dst in glossary.items() if src.strip())
        system += f"\n\nAlways translate these terms exactly as given:\n{terms}"
    if notes.strip():
        system += f"\n\nBackground on what is being shown:\n{notes.strip()}"

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": "user", "content": turn.source})
        messages.append({"role": "assistant", "content": turn.translation})
    messages.append({"role": "user", "content": current})
    return messages


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
    messages[-1]["content"] = f"{listing}\n\nTranslate item {SCREEN_LETTERS[index]}."
    return messages


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


class Translator(Protocol):
    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str: ...


class LlamaTranslator:
    """llama.cpp backed translator.

    Qwen3.5's hybrid layers cannot be trimmed back to a shared prompt start, so llama.cpp re-reads the whole prompt
    for every line (about 1 s). Instead the state after the instructions is saved once and restored per line.
    """

    max_snapshots = 3
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
        self.temperature = 0.2
        self._snapshots: dict[tuple[int, ...], object] = {}
        self._parallel = None

    def close(self) -> None:
        self._snapshots.clear()
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
        prompts = [self._tokens(messages)[1] for messages in conversations]
        _, probe = self._tokens(conversations[0][:1] + [{"role": "user", "content": "\u2063"}])
        instructions = next((i for i, (a, b) in enumerate(zip(prompts[0], probe)) if a != b), 0)
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

    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str:
        formatted, tokens = self._tokens(messages)
        self._resume(tokens)
        budget = min(1024, 64 + 3 * len(messages[-1]["content"]))
        raw = ""
        shown = ""
        for chunk in self._llm.create_completion(
            tokens, stream=True, temperature=self.temperature, top_p=0.9, max_tokens=budget, stop=formatted.stop,
        ):
            raw += chunk["choices"][0]["text"]
            visible = clean_output(raw)
            if on_partial and visible != shown:
                shown = visible
                on_partial(visible)
        self._save_instructions(messages, tokens)
        return clean_output(raw)

    def _resume(self, tokens: list[int]) -> None:
        matches = [key for key in self._snapshots if len(key) < len(tokens) and tuple(tokens[: len(key)]) == key]
        if matches:
            key = max(matches, key=len)
            self._snapshots[key] = self._snapshots.pop(key)
            self._llm.load_state(self._snapshots[key])

    def _save_instructions(self, messages: list[dict], tokens: list[int]) -> None:
        """Snapshot the prompt up to the first message, after this line so it is not delayed; once per instruction set."""
        _, probe = self._tokens(messages[:1] + [{"role": "user", "content": "\u2063"}])
        shared = next((i for i, (a, b) in enumerate(zip(tokens, probe)) if a != b), min(len(tokens), len(probe)))
        key = tuple(tokens[:shared])
        if shared < self.min_shared_tokens or key in self._snapshots:
            return
        self._llm.reset()
        self._llm.eval(list(key))
        self._snapshots[key] = self._llm.save_state()
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(next(iter(self._snapshots)))
