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
from dataclasses import dataclass, field
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
# お前 and "hell" are left out: teachers, parents and couples say them all the time.
ROUGH = re.compile(r"\b(damn|shit|fuck\w*|bitch|homie|bro|dude|fool|punk|ass(hole)?|dumbass|motherf\w*)\b|"
                   r"てめぇ|テメェ|貴様|ぶっ殺|うるせ|やがる|ふざけん|くそ|クソ|새끼|씨발|개새|他妈|妈的|滚", re.I)
# English and Korean titles count only when they call the listener ("Brother, wait!", "오빠, 그 말..."), not when
# they mention someone ("오빠 편만 들어요"); Japanese and Chinese often use the title for "you" inside the sentence.
_VOCATIVE = r"(?:^\W*{0}(?![\w가-힣])|(?:[,!?.]\s*){0}(?=\s*(?:[,!?.~…]|$))|{0}(?=\s*[,!?~…]))"
# Recognized Korean speech has no spaces or commas: a title opening the line calls someone unless a particle follows.
_KOREAN_OPENING = r"|^\W*(?:{0})(?!가|는|를|도|의|한테|랑|이|에게|께|하고|님[이은을의])"
TO_ELDER = {
    "male": re.compile(_VOCATIVE.format(r"(?:(?:big )?bro(?:ther)?|아저씨|오빠|형(?:님)?)")
                       + _KOREAN_OPENING.format("오빠|아저씨") + r"|兄さん|兄ちゃん|お兄|兄貴|"
                       r"哥哥|师兄", re.I),
    "female": re.compile(_VOCATIVE.format(r"(?:(?:big )?sis(?:ter)?|언니|누나)") + _KOREAN_OPENING.format("언니|누나")
                         + r"|姉さん|姉ちゃん|お姉|姐姐|师姐", re.I),
    "": re.compile(r"先輩|せんぱい|선배(?:님)?|学长|学姐"),
}
SERVING = re.compile(r"^\W*welcome\b(?! back| home)|いらっしゃいませ|お客様|손님|어서 오세요|欢迎光临", re.I)
RELATIONS = ("parent and child", "grandparent and grandchild", "siblings or cousins", "couple", "teacher and student",
             "school senior and junior", "master and disciple", "ruler and subject", "friends", "rivals or enemies",
             "coworkers", "boss and employee", "staff and customer", "strangers")
# Titles in a name ("田中先生", "Princess Elena") or words calling someone ("Your Highness") settle a role outright;
# a 4B judge otherwise mixes up who outranks whom.
ROLES = {
    "teacher and student": re.compile(r"先生|せんせい|\bsensei\b|\bteacher\b|\bprofessor\b|老师|老師|선생님|교수님", re.I),
    "master and disciple": re.compile(r"师父|師父|師匠|师傅|\bmaster\b|sư phụ", re.I),
    "ruler and subject": re.compile(r"\b(princess|prince|king|queen|emperor|empress|majesty|highness|my lady|my lord)\b|"
                                    r"陛下|殿下|王子|公主|皇上|왕자|공주|폐하|전하", re.I),
    "grandparent and grandchild": re.compile(r"\b(grandma|grandpa|grandmother|grandfather|granny)\b|할머니|할아버지|"
                                             r"おばあ|おじい|祖母|祖父|奶奶|爷爷|外婆|外公", re.I),
    "parent and child": re.compile(r"\b(mom|mum|dad|mother|father)\b|엄마|아빠|어머니|아버지|お母|お父|母さん|父さん|妈妈|爸爸",
                                   re.I),
}
PRINCELY = re.compile(r"\b(prince|princess|highness)\b|殿下|王子|公主|왕자|공주|전하", re.I)
FEMALE_TITLES = re.compile(r"\b(princess|queen|empress|lady|grandma|grandmother|granny|mom|mum|mother)\b|공주|할머니|엄마|"
                           r"어머니|おばあ|お母|母さん|祖母|奶奶|外婆|妈妈|公主", re.I)
MALE_TITLES = re.compile(r"\b(prince|king|emperor|lord|grandpa|grandfather|dad|father)\b|왕자|할아버지|아빠|아버지|おじい|"
                         r"お父|父さん|祖父|爷爷|外公|爸爸|王子", re.I)
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
 "senior": who is the parent, grandparent, teacher, master, ruler, boss or shop staff: "{a}" or "{b}" or "neither",
 "rough": true if they talk to each other crudely: swearing, insults, threats, gangster, thug or delinquent talk, or rude banter between close friends, else false}}"""


def _looks_like_name(label: str) -> bool:
    """One or two words, each capitalised unless written in a script without capitals ("Big Smoke", "リュウ")."""
    words = label.split()
    return 0 < len(words) <= 2 and all(word[0].isupper() or not word[0].isascii() for word in words)


def body_of(text: str) -> str:
    """The line without its "Name:" label or name box, so a name such as "Sir Aldric" is not read as a cue."""
    match = _LABEL.match(text.strip())
    if match and _looks_like_name(match.group(1)):
        return match.group(2)
    name = speaker_of(text)
    if name:
        first, _, rest = text.strip().partition("\n")
        if first.strip() == name:
            return rest.strip()
    return text


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
    senior = facts.get("senior", "")
    rough = facts.get("rough") is True
    gender = lambda who: facts.get(f"{who} gender", "unknown")
    # With the elder's gender unknown, only the junior's "em" is certain; the model picks anh or chị.
    elder = lambda who: {"female": "chị", "male": "anh"}.get(gender(who), "")
    female = lambda who: gender(who) == "female"
    top = senior if senior in (speaker, listener) else older if older in (speaker, listener) else ""
    by_gender = lambda women, men: lambda who: {"female": women, "male": men}.get(gender(who), "")
    ranked = {"parent": by_gender("mẹ", "bố"), "grandparent": by_gender("bà", "ông"), "teacher": by_gender("cô", "thầy")}
    for role, title in ranked.items():
        if relation.startswith(role) and top:
            young = "con" if role == "parent" else "cháu" if role == "grandparent" else "em"
            return (title(speaker), young) if top == speaker else (young, title(listener))
    if relation.startswith("master") and top:
        return ("ta", "con") if top == speaker else ("đệ tử", "sư phụ")
    if relation.startswith("ruler") and top:
        return ("ta", "ngươi") if top == speaker else ("thần", facts.get("royal", "bệ hạ"))
    if "customer" in relation:
        if senior == listener:
            return "tôi", {"female": "chị", "male": "anh"}.get(gender(listener), "bạn")
        return "tôi", "quý khách"
    if relation == "couple" and older not in (speaker, listener) and gender(speaker) != gender(listener) != "unknown":
        older = speaker if gender(speaker) == "male" else listener
    if relation in ("family", "siblings or cousins", "couple", "school senior and junior") and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation == "friends" and not rough and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation == "friends" or rough:
        return ("tao", "mày") if rough else ("tớ", "cậu")
    if relation in ("coworkers", "boss and employee") and top:
        return (elder(speaker), "em") if top == speaker else ("em", elder(listener))
    return "tôi", {"female": "chị", "male": "anh"}.get(gender(listener), "bạn")


@dataclass(frozen=True)
class Address:
    speaker: str
    listener: str
    me: str
    you: str
    # From a single line rather than the judge: used for that line, not kept in the history for later ones.
    guessed: bool = field(default=False, compare=False)

    def note(self) -> str:
        words = [f'"{word}" for {role}' for word, role in ((self.me, "I/me"), (self.you, "you")) if word]
        return f'{self.speaker} is talking to {self.listener}: in Vietnamese {self.speaker} says {" and ".join(words)}.'


    def terms(self, source_lang: str) -> dict[str, str]:
        """Glossary entries for translation models that follow terms but not instructions (HY-MT)."""
        langs = [source_lang] if source_lang in SELF_WORDS else list(SELF_WORDS)
        terms = {word: self.me for lang in langs for word in SELF_WORDS[lang]} if self.me else {}
        if self.you:
            terms.update({word: self.you for lang in langs for word in YOU_WORDS[lang]})
        return terms


def guess_facts(line: str, speaker: str, listener: str) -> dict | None:
    """Facts plain from one line: calling the listener big brother or 先輩, serving a customer, swearing."""
    line = body_of(line)
    if SERVING.search(line):
        return {"relation": "staff and customer", "senior": speaker, "guess": True}
    for role, cue in ROLES.items():
        if calls_by(cue, line):
            facts = {"relation": role, "senior": listener, "older": listener, "guess": True}
            title = cue.search(line).group(0)
            gender = "female" if FEMALE_TITLES.search(title) else "male" if MALE_TITLES.search(title) else ""
            return {**facts, f"{listener} gender": gender} if gender else facts
    for gender, cue in TO_ELDER.items():
        found = cue.search(line)
        if found and not _MENTIONED.search(line[:found.start()]):
            facts = {"relation": "school senior and junior", "older": listener, "guess": True}
            return {**facts, f"{listener} gender": gender} if gender else facts
    if ROUGH.search(line):
        return {"relation": "friends", "rough": True, "guess": True}
    return None


def calls_by(cue: re.Pattern, line: str) -> bool:
    """Whether a title in the line calls the listener ("Mom, I'm home", "Your Highness!") rather than names someone."""
    for match in cue.finditer(line):
        before, after = line[:match.start()], line[match.end():]
        if re.match(r"(?:가|는|를|도|의|한테|랑|이|에게|께|하고)", after) or _MENTIONED.search(before):
            continue
        opening = not before.strip(" \"'「『(-—…")
        closing = not after.strip() or after.lstrip()[:1] in ",!?.、，！？~…。"
        # An English title must stand apart to call someone: "Mom, I'm home", not "Mom said so".
        if match.group(0).isascii() and opening and not closing:
            continue
        if opening or closing:
            return True
    return False


# Words before a title that make it a mention: "your mother", "like a mother", "手紙には", "우리 엄마".
_MENTIONED = re.compile(r"(?:\b(?:your|my|his|her|our|their|the|a|an|like|this|that|told|said|says)\s+|"
                        r"の|には|って|と|우리\s*|내\s*|네\s*)$", re.I)


def settled(facts: dict, a: str, b: str, said: list[tuple[str, str]]) -> dict:
    """The judge's reading, with roles taken from titles and forms of address wherever the text shows them.

    `said` holds (speaker, line) for the lines either of the two said to the other.
    """
    roles: dict[str, str] = {}
    facts = dict(facts)
    for name in (a, b):
        role = next((role for role, cue in ROLES.items() if cue.search(name)), None)
        if role:
            roles[name] = role
    for said_by, line in said:
        other = b if said_by == a else a if said_by == b else None
        if other and other not in roles:
            role = next((role for role, cue in ROLES.items() if calls_by(cue, body_of(line))), None)
            if role:
                roles[other] = role
                title = ROLES[role].search(body_of(line)).group(0)
                if FEMALE_TITLES.search(title):
                    facts.setdefault(f"{other} gender", "female")
                elif MALE_TITLES.search(title):
                    facts.setdefault(f"{other} gender", "male")
    facts = dict(facts)
    for name in (a, b):
        if FEMALE_TITLES.search(name):
            facts[f"{name} gender"] = "female"
        elif MALE_TITLES.search(name):
            facts[f"{name} gender"] = "male"
    enemies = str(facts.get("relation", "")).startswith("rivals")
    if len(roles) == 1 and not enemies:
        [(senior, role)] = roles.items()
        royal = "điện hạ" if any(PRINCELY.search(text) for text in [senior, *(line for _w, line in said)]) else "bệ hạ"
        return {**facts, "relation": role, "senior": senior, "royal": royal}
    relation = str(facts.get("relation", ""))
    if relation.startswith(("teacher", "master")):
        return {**facts, "relation": "school senior and junior", "senior": "neither"}
    if relation.startswith("ruler"):
        return {**facts, "relation": "boss and employee"}
    if relation.startswith(("parent", "grandparent")) and facts.get("senior") != facts.get("older"):
        return {**facts, "relation": "siblings or cousins", "senior": "neither"}
    return facts


def merged(judged: dict | None, cues: dict | None) -> dict | None:
    """The model's reading, except that a plain cue (先輩, "brother") settles who is older when it could not tell."""
    if not judged:
        return cues
    anchored = judged.get("senior") in (cues or {}).get("pair", ())
    if cues and cues.get("older") and judged.get("older") not in cues["pair"] and not anchored:
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


def _calls_name(line: str, name: str) -> bool:
    """A name opening the line or set off by punctuation calls that person; one inside the sentence only mentions them."""
    title = r"(?:(?:Uncle|Aunt|Mr|Mrs|Ms|Miss|Dr|Captain|Sergeant|Sir|Lady)\.?\s+)?"
    suffix = r"(?:さん|君|くん|ちゃん|様|先輩|先生|-sensei|-san|-kun|-chan|아|야|씨)?"
    return bool(re.search(rf"(?:^|[,、，!！?？]\s*){title}{re.escape(name)}{suffix}\s*(?:[,、，!！?？~…。.]|$)",
                          line.strip()))


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
        self._lines: deque[tuple[str | None, str, float, str | None]] = deque(maxlen=40)
        self._spoken: Counter[str] = Counter()
        self._facts: dict[frozenset, dict] = {}
        self._cues: dict[frozenset, dict] = {}
        self._asked: dict[frozenset, int] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._judge = None
        self._closing = threading.Event()
        self._unavailable = False

    def _listener(self, speaker: str, text: str = "") -> str | None:
        """Whoever the line calls by name, else the last other character who has said a few lines."""
        body = body_of(text)
        named = [who for who in self._spoken if who != speaker and _calls_name(body, who)]
        if named:
            return max(named, key=len)
        cutoff = self.clock() - self.window_s
        recent = [who for who, _text, at, _to in reversed(self._lines) if who and who != speaker and at >= cutoff]
        if not self._spoken[speaker]:
            # A newcomer's first line answers whoever just spoke.
            return recent[0] if recent else None
        return next((who for who in recent if self._spoken[who] >= 2), recent[0] if recent else None)

    def address(self, text: str) -> Address | None:
        if self.settings().target_lang not in SUPPORTED:
            return None
        speaker = speaker_of(text)
        if not speaker:
            facts = guess_facts(text, "the speaker", "the listener")
            return Address("the speaker", "the listener", *vietnamese(facts, "the speaker", "the listener"), True) \
                if facts else None
        with self._lock:
            listener = self._listener(speaker, text)
            pair = frozenset((speaker, listener))
            if listener:
                self._note_cues(pair)
            cues = self._cues.get(pair) if listener else None
            facts = merged(self._facts.get(pair), cues) if listener else None
            if listener and facts is None:
                by_title = settled({}, speaker, listener, self._said(pair) + [(speaker, text)])
                facts = by_title if by_title.get("relation") else None
        if facts is None:
            listener = listener or "the listener"
            facts = guess_facts(text, speaker, listener)
        return Address(speaker, listener, *vietnamese(facts, speaker, listener), bool(facts.get("guess"))) \
            if facts else None

    def _said(self, pair: frozenset) -> list[tuple[str, str]]:
        return [(who, line) for who, line, _a, to in self._lines if who in pair and to in pair and to != who]

    def _note_cues(self, pair: frozenset) -> None:
        """Cues from every line either of the two said to the other so far, the latest ones winning."""
        for who, line, _at, to in self._lines:
            if who in pair and (to is None or (to in pair and to != who)):
                other = next(iter(pair - {who}), None)
                if other is None:
                    continue
                guess = guess_facts(line, who, other)
                known = self._cues.get(pair, {})
                # One insult between siblings does not make them street rivals.
                if guess and guess.get("rough") and (known.get("older") or known.get("senior")):
                    continue
                if guess and (guess.get("older") or not known.get("older")):
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
            listener = self._listener(speaker, text) if speaker else None
            self._lines.append((speaker, text.strip(), self.clock(), listener))
            if speaker:
                self._spoken[speaker] += 1
            if not listener:
                return
            pair = frozenset((speaker, listener))
            self._note_cues(pair)
            said = sum(1 for who, _t, _a, _to in self._lines if who in pair)
            asked = self._asked.get(pair)
            if asked is not None and not any(asked < at <= said for at in RECHECK_AT):
                return
            self._asked[pair] = said
            lines = [line for who, line, _a, _to in self._lines if who in pair or who is None][-PAIR_LINES:]
            said = self._said(pair)
        self._jobs.put((listener, speaker, lines, said))
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="relationships", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None or self._closing.is_set():
                return
            first, second, lines, said = job
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
            if facts:
                facts = settled(facts, first, second, said)
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
