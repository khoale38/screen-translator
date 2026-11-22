"""Who is talking to whom, and the Vietnamese pronouns that go with it (tao/mày, anh/em, tôi/quý khách...).

Small translation models default to the polite "tôi/bạn" because a single line rarely says how two characters are
related. Speakers come from "Name: line" subtitles, and the listener is usually whoever spoke just before. A language
model reads the conversation once per pair of characters, in the background, and answers plain facts (how they are
related, who is older, their genders, whether they talk rough); fixed rules turn those facts into pronouns, which
small models choose far more reliably than the pronouns themselves.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

from . import diagnostics

_LABEL = re.compile(r"^([^\s:：\d][^:：\n\d]{0,19}?)\s*[:：]\s*(\S.*)$", re.S)
_SENTENCE_END = (".", "!", "?", "。", "！", "？", "…", ",", "、")
SUPPORTED = {"vi"}
SELF_WORDS = {"en": ("I", "me", "I'm", "I'll", "I've", "I'd", "my", "myself"),
              "ja": ("俺", "僕", "私", "あたし", "オレ"), "ko": ("나", "저", "내"), "zh": ("我",)}
YOU_WORDS = {"en": ("you", "you're", "you'll", "you've", "your", "yourself"),
             "ja": ("お前", "てめぇ", "君", "あなた", "あんた", "貴様"), "ko": ("너", "당신", "네"), "zh": ("你", "您")}
# Cues clear enough to guess from at once, before the model has read the conversation.
ROUGH = re.compile(r"\b(damn|shit|fuck\w*|bitch|homie|bro|dude|fool|punk|ass(hole)?|hell|yo|dumbass|motherf\w*)\b|"
                   r"てめぇ|テメェ|貴様|ぶっ殺|うるせ|やがる|ふざけん|くそ|クソ|お前|새끼|씨발|개새|他妈|妈的|滚", re.I)
TO_ELDER = {"male": re.compile(r"\bbrother\b|\bbig bro\b|先輩|せんぱい|兄さん|兄ちゃん|お兄|兄貴|아저씨|형|오빠|선배|哥哥|师兄", re.I),
            "female": re.compile(r"\bsister\b|\bsis\b|姉さん|姉ちゃん|お姉|언니|누나|姐姐|师姐", re.I)}
SERVING = re.compile(r"\b(welcome|sir|ma'am|madam)\b|いらっしゃいませ|お客様|손님|어서 오세요|欢迎光临|客人", re.I)
RELATIONS = ("family", "couple", "school senior and junior", "friends", "rivals or enemies", "coworkers",
             "boss and employee", "staff and customer", "strangers")
PAIR_LINES = 10
RECHECK_AT = (4, 10)

QUESTION = """Background: {notes}
Recent lines:
{lines}

Answer questions about {a} and {b} with JSON only:
{{"relation": one of {relations},
 "older": "{a}" or "{b}" or "same age" or "unknown",
 "{a} gender": "male" or "female" or "unknown",
 "{b} gender": "male" or "female" or "unknown",
 "rough": true if they talk to each other crudely: swearing, insults, threats, gangster, thug or delinquent talk, or rude banter between close friends, else false}}"""


def _looks_like_name(label: str) -> bool:
    """One or two words, each capitalised unless written in a script without capitals ("Big Smoke", "リュウ")."""
    words = label.split()
    return 0 < len(words) <= 2 and all(word[0].isupper() or not word[0].isascii() for word in words)


def speaker_of(text: str) -> str | None:
    """The name in "Name: line" or a short name on its own first line, if the text has one."""
    text = text.strip()
    match = _LABEL.match(text)
    if match and _looks_like_name(match.group(1)):
        return match.group(1).strip()
    first, _, rest = text.partition("\n")
    first = first.strip()
    if rest.strip() and len(first) <= 12 and not first.endswith(_SENTENCE_END) and _looks_like_name(first):
        return first
    return None


def vietnamese(facts: dict, speaker: str, listener: str) -> tuple[str, str]:
    """(word for I, word for you) when `speaker` talks to `listener`."""
    relation = str(facts.get("relation", ""))
    older = facts.get("older", "")
    rough = facts.get("rough") is True
    gender = lambda who: facts.get(f"{who} gender", "unknown")
    elder = lambda who: "chị" if gender(who) == "female" else "anh"
    if "customer" in relation:
        return "tôi", "quý khách"
    if relation == "couple" and older not in (speaker, listener) and gender(speaker) != gender(listener) != "unknown":
        older = speaker if gender(speaker) == "male" else listener
    if relation in ("family", "couple", "school senior and junior") and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation == "friends" and not rough and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation in ("friends", "rivals or enemies") or rough:
        return ("tao", "mày") if rough or "rival" in relation else ("tớ", "cậu")
    if relation in ("coworkers", "boss and employee") and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    return "tôi", {"female": "chị", "male": "anh"}.get(gender(listener), "bạn")


@dataclass(frozen=True)
class Address:
    speaker: str
    listener: str
    me: str
    you: str

    def note(self) -> str:
        return (f'{self.speaker} is talking to {self.listener}: in Vietnamese {self.speaker} says "{self.me}" '
                f'for I/me and "{self.you}" for you.')

    def terms(self, source_lang: str) -> dict[str, str]:
        """Glossary entries for translation models that follow terms but not instructions (HY-MT)."""
        langs = [source_lang] if source_lang in SELF_WORDS else list(SELF_WORDS)
        terms = {word: self.me for lang in langs for word in SELF_WORDS[lang]}
        terms.update({word: self.you for lang in langs for word in YOU_WORDS[lang]})
        return terms


def guess_facts(line: str, speaker: str, listener: str) -> dict | None:
    """Facts plain from one line: calling the listener big brother or 先輩, serving a customer, swearing."""
    if SERVING.search(line):
        return {"relation": "staff and customer", "guess": True}
    for gender, cue in TO_ELDER.items():
        if cue.search(line):
            return {"relation": "school senior and junior", "older": listener, f"{listener} gender": gender,
                    "guess": True}
    if ROUGH.search(line):
        return {"relation": "friends", "rough": True, "guess": True}
    return None


def merged(judged: dict | None, cues: dict | None) -> dict | None:
    """The model's reading, except that a plain cue (先輩, "brother") settles who is older when it could not tell."""
    if not judged:
        return cues
    if cues and cues.get("older") and judged.get("older") not in cues["pair"]:
        return {**judged, "relation": cues["relation"], "older": cues["older"],
                **{key: value for key, value in cues.items() if key.endswith(" gender")}}
    return judged


def parse_facts(answer: str) -> dict | None:
    match = re.search(r"\{.*\}", answer, re.S)
    try:
        facts = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    return facts if isinstance(facts, dict) and facts.get("relation") else None


class Relationships:
    """Keeps the pronouns for each pair of speakers; `judge_factory` returns something with `answer(question)`."""

    def __init__(self, judge_factory: Callable[[], object], settings: Callable, emit: Callable[[dict], None],
                 clock: Callable[[], float] = time.monotonic, window_s: float = 120.0,
                 judge_key: Callable[[], object] = lambda: None) -> None:
        self.judge_factory = judge_factory
        self.judge_key = judge_key
        self._judge_made_for = None
        self.settings = settings
        self.emit = emit
        self.clock = clock
        self.window_s = window_s
        self._lock = threading.Lock()
        self._lines: deque[tuple[str | None, str, float]] = deque(maxlen=40)
        self._spoken: Counter[str] = Counter()
        self._facts: dict[frozenset, dict] = {}
        self._cues: dict[frozenset, dict] = {}
        self._asked: dict[frozenset, int] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._judge = None
        self._closing = threading.Event()
        self._unavailable = False

    def _listener(self, speaker: str) -> str | None:
        cutoff = self.clock() - self.window_s
        recent = [who for who, _text, at in reversed(self._lines) if who and who != speaker and at >= cutoff]
        return next((who for who in recent if self._spoken[who] >= 2), recent[0] if recent else None)

    def address(self, text: str) -> Address | None:
        if self.settings().target_lang not in SUPPORTED:
            return None
        speaker = speaker_of(text)
        if not speaker:
            facts = guess_facts(text, "the speaker", "the listener")
            return Address("the speaker", "the listener", *vietnamese(facts, "the speaker", "the listener")) \
                if facts else None
        with self._lock:
            listener = self._listener(speaker)
            pair = frozenset((speaker, listener))
            if listener:
                self._note_cues(pair)
            cues = self._cues.get(pair) if listener else None
            facts = merged(self._facts.get(pair), cues) if listener else None
        if facts is None:
            listener = listener or "the listener"
            facts = guess_facts(text, speaker, listener)
        return Address(speaker, listener, *vietnamese(facts, speaker, listener)) if facts else None

    def _note_cues(self, pair: frozenset) -> None:
        """Cues from every line either of the two has said so far, the latest ones winning."""
        for who, line, _at in self._lines:
            if who in pair:
                other = next(iter(pair - {who}))
                guess = guess_facts(line, who, other)
                if guess and (guess.get("older") or not self._cues.get(pair, {}).get("older")):
                    self._cues[pair] = {**guess, "pair": tuple(pair)}

    def heard(self, text: str) -> None:
        """A line that was shown and kept; a new pair of speakers, or one with more to go on, gets judged."""
        if self.settings().target_lang not in SUPPORTED:
            return
        if self._unavailable and self.judge_key() == self._judge_made_for:
            return
        self._unavailable = False
        speaker = speaker_of(text)
        with self._lock:
            listener = self._listener(speaker) if speaker else None
            self._lines.append((speaker, text.strip(), self.clock()))
            if speaker:
                self._spoken[speaker] += 1
            if not listener:
                return
            pair = frozenset((speaker, listener))
            self._note_cues(pair)
            said = sum(1 for who, _t, _a in self._lines if who in pair)
            asked = self._asked.get(pair)
            if asked is not None and not any(asked < at <= said for at in RECHECK_AT):
                return
            self._asked[pair] = said
            lines = [line for who, line, _a in self._lines if who in pair or who is None][-PAIR_LINES:]
        self._jobs.put((listener, speaker, lines))
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="relationships", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None or self._closing.is_set():
                return
            first, second, lines = job
            try:
                self._use_judge(self.judge_key())
                started = time.monotonic()
                answer = self._judge.answer(QUESTION.format(
                    notes=self.settings().notes or "none", lines="\n".join(lines), a=first, b=second,
                    relations=", ".join(f'"{r}"' for r in RELATIONS)))
            except FileNotFoundError:
                self._unavailable = True
                self._judge_made_for = self.judge_key()
                self.emit({"type": "relationships", "error": "Download Qwen3.5 4B to pick pronouns from who is "
                                                              "talking to whom."})
                continue
            except Exception as exc:
                diagnostics.log.warning("Relationship check failed: %s", exc)
                continue
            facts = parse_facts(answer)
            diagnostics.log.info("Relationship %s / %s: %s in %.1f s", first, second, facts or answer,
                                 time.monotonic() - started)
            if facts:
                with self._lock:
                    pair = frozenset((first, second))
                    self._facts[pair] = facts
                    facts = merged(facts, self._cues.get(pair))
                self.emit({"type": "relationships", "pair": [first, second],
                           "address": [" / ".join(vietnamese(facts, first, second)),
                                       " / ".join(vietnamese(facts, second, first))]})

    def _use_judge(self, key) -> None:
        if self._judge is not None and key == self._judge_made_for:
            return
        self._close_judge()
        self._judge = self.judge_factory()
        self._judge_made_for = key

    def _close_judge(self) -> None:
        close = getattr(self._judge, "close", None)
        if close:
            close()
        self._judge = None

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._spoken.clear()
            self._facts.clear()
            self._cues.clear()
            self._asked.clear()

    def close(self) -> None:
        """Stop judging; the model is freed only once it has finished any answer it is in the middle of."""
        self._closing.set()
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                return
            self._thread = None
        self._close_judge()
