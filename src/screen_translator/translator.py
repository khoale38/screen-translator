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
    def __init__(self, model_path: str, gpu_layers: int = -1, n_ctx: int = 4096) -> None:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=n_ctx, verbose=False)
        token_text = lambda token: self._llm.detokenize([token], special=True).decode("utf-8", "ignore")
        self._format = Jinja2ChatFormatter(
            self._llm.metadata["tokenizer.chat_template"],
            eos_token=token_text(self._llm.token_eos()),
            bos_token=token_text(self._llm.token_bos()),
        )

    def close(self) -> None:
        self._llm.close()

    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str:
        # Reasoning models (Qwen3/3.5) honour this flag in their own template; others ignore it.
        formatted = self._format(messages=messages, enable_thinking=False)
        tokens = self._llm.tokenize(formatted.prompt.encode("utf-8"), add_bos=False, special=True)
        budget = min(512, 64 + 3 * len(messages[-1]["content"]))
        raw = ""
        shown = ""
        for chunk in self._llm.create_completion(
            tokens, stream=True, temperature=0.2, top_p=0.9, max_tokens=budget, stop=formatted.stop,
        ):
            raw += chunk["choices"][0]["text"]
            visible = clean_output(raw)
            if on_partial and visible != shown:
                shown = visible
                on_partial(visible)
        return clean_output(raw)
