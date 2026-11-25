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
from .romanize import CHOICE_WORDS

_BRACKETED = re.compile(r"^[（(\[【〔]\s*([^()（）\[\]【】〔〕\n]{1,12}?)\s*[)）\]】〕]\s*(\S.*)$", re.S)
# Lines to a whole group get no pronoun pair: a host to her viewers, a teacher to the class.
GROUP = re.compile(r"\b(?:everyone|everybody|you guys|guys|folks|you all|y'all)\b|các bạn|여러분|皆さん|みなさん|みんな|大家|各位",
                   re.I)
_DISCOURSE = {"then", "okay", "ok", "so", "wait", "sorry", "listen", "well", "oh", "hey", "yes", "no", "please", "thanks",
              "and", "but", "now", "look", "come", "go", "hi", "bye"}
_VOCATIVE_TITLES = {"mom", "mum", "dad", "mother", "father", "grandma", "grandpa", "sis", "bro", "honey", "sir", "ma'am",
                    "dear", "babe", "kid"}
_LABEL = re.compile(r"^([^\s:：\d][^:：\n\d]{0,19}?)\s*[:：]\s*(\S.*)$", re.S)
_SENTENCE_END = (".", "!", "?", "。", "！", "？", "…", ",", "、")
SUPPORTED = {"vi"}
SELF_WORDS = {"en": ("I", "me", "I'm", "I'll", "I've", "I'd", "my", "myself"),
              "ja": ("俺", "僕", "私", "あたし", "オレ"), "ko": ("나", "저", "내"), "zh": ("我",)}
YOU_WORDS = {"en": ("you", "you're", "you'll", "you've", "your", "yourself"),
             "ja": ("お前", "てめぇ", "君", "あなた", "あんた", "貴様"), "ko": ("너", "당신", "네"), "zh": ("你", "您")}
# Cues clear enough to guess from at once, before the model has read the conversation.
# お前 and "hell" are left out: teachers, parents and couples say them all the time.
ROUGH = re.compile(r"\b(damn|shit|fuck\w*|bitch|fool|punk|ass(hole)?|dumbass|motherf\w*)\b|"
                   r"黙れ|닥쳐|따위가|闭嘴|閉嘴|てめぇ|てめえ|テメェ|貴様|ぶっ殺|うるせ|やがる|ふざけん|(?<![やおゆ])くそ(?!こ)|クソ|새끼|씨발|개새|他妈的|(?:^|[，,。！!？?\s])妈的|(?:^|[，,。！!？?\s]|给我)滚(?:开|出去|蛋)?(?=[！!。，,\s]|$)", re.I)
# English and Korean titles count only when they call the listener ("Brother, wait!", "오빠, 그 말..."), not when
# they mention someone ("오빠 편만 들어요"); Japanese and Chinese often use the title for "you" inside the sentence.
_VOCATIVE = r"(?:^\W*{0}(?![\w가-힣])|(?:[,!?.]\s*){0}(?=\s*(?:[,!?.~…]|$))|{0}(?=\s*[,!?~…]))"
# Recognized Korean speech has no spaces or commas: a title opening the line calls someone unless a particle follows.
_KOREAN_OPENING = r"|^\W*(?:{0})(?!가|는|를|도|의|한테|랑|이(?![거것제])|에게|께|하고|님[이은을의])"
# 형 starts many words (형사, 형제): glued to the line only before words that go on a call.
_KOREAN_GLUED_TITLE = r"|^\W*(?:{0})(?=나|저|제가|내가|우리|너|왜|빨리|어디|이거|지금|잠깐|같이|조심|걱정)"
TO_ELDER = {
    "male": re.compile(_VOCATIVE.format(r"(?:big bro|(?:big )?brother|hyung|oppa|gege|shixiong|aniki|onii-?(?:chan|san)|nii-?san|오빠|"
                                        r"사형|형(?:님)?)")
                       + _KOREAN_OPENING.format("오빠|사형") + _KOREAN_GLUED_TITLE.format("형") + r"|(?:오빠|형)(?=는\s*왜)|^\W*bro(?=\s*[,!])|兄さん|兄ちゃん|お兄|兄貴|大哥|二哥|大师兄|大師兄|"
                       r"^哥(?= |[，,！!？?]|你|我)|哥哥|大佬|师兄|師兄|皇兄|兄上|学长|學長", re.I),
    "female": re.compile(_VOCATIVE.format(r"(?:(?:big )?sis(?:ter)?|noona|unnie|jiejie|shijie|onee-?(?:chan|san)|nee-?san|언니|누나|사저)")
                         + _KOREAN_OPENING.format("언니|누나|사저") + r"|(?:언니|누나)(?=는\s*왜)|姉さん|姉ちゃん|お姉|姐姐|姊姊|家姐|师姐|師姐|学姐|學姊|^姐(?= |[，,！!？?]|你|我)",
                         re.I),
    "": re.compile(_VOCATIVE.format(r"(?:senpai|sunbae(?:nim)?)") + r"|先輩|せんぱい|선배(?:님)?|대리님|주임님|선임님", re.I),
}
SERVING = re.compile(r"^\W*welcome\b(?! back| home)|いらっしゃいませ|^\W*お客様[、，,]|^\W*손님\s*[,，]|어서 오세요|欢迎光临|주문|"
                     r"こちらへどうぞ",
                     re.I)
STAFF = re.compile(r"\b(?:waiter|waitress|clerk|cashier|bartender|barista)\b|服务员|服務員|店员|店員|ウェイター|ウェイトレス|"
                   r"웨이터|직원|점원", re.I)
RELATIONS = ("parent and child", "grandparent and grandchild", "siblings or cousins", "couple", "teacher and student",
             "school senior and junior", "master and disciple", "ruler and subject", "friends", "rivals or enemies",
             "coworkers", "boss and employee", "staff and customer", "strangers")
# Titles in a name ("田中先生", "Princess Elena") or words calling someone ("Your Highness") settle a role outright;
# a 4B judge otherwise mixes up who outranks whom.
ROLES = {
    "teacher and student": re.compile(r"先生|せんせい|\bsensei\b|\bteacher\b|\bprofessor\b|老师|老師|선생님|교수님|"
                                      r"^\W*(?:Miss|Mrs|Ms|Mr)\.? [A-Z][a-z]+(?=,.*\b(?:homework|book|class|test|exam|quiz|school|"
                                      r"assignment)\b)"),
    "master and disciple": re.compile(r"师父|師父|師匠|师傅(?=.*?(?:弟子|徒儿|练功|武功|剑法|教训|教诲|指点|收[^，,。]{0,4}徒弟|当师傅))|师尊|師尊|为师|為師|掌门|掌門|"
                                      r"(?<!young )\bmaster\b|\bshizun\b|\bshifu\b|sư phụ|사부님|사부|스승님",
                                      re.I),
    "ruler and subject": re.compile(r"\b(princess|prince|king|queen|emperor|empress|majesty|highness)\b|"
                                    r"陛下|殿下|王子|公主|皇上|太后|娘娘|姫様|마마|왕자|공주|폐하|전하", re.I),
    # Not "young lady" (a parent scolding), "Oh my Lord", or a waiter's bare 小姐/아가씨.
    "noble and servant": re.compile(r"\b(my lady|milady|m'lady|milord|young master)\b|(?<!oh )(?<!good )(?<!dear )"
                                    r"\bmy lord\b|お嬢様|お坊ちゃま|ご主人様|旦那様|[大二三]小姐|少爷|도련님", re.I),
    "boss and employee": re.compile(r"부장님|팀장님|과장님|차장님|실장님|본부장님|회장님|대표님|사장님|部長|課長|社長|经理|總監|总监|总裁|"
                                   r"董事长|[\u4e00-\u9fff]总(?=[，,！!？?])|\bboss\b", re.I),
    "aunt or uncle and niece or nephew": re.compile(r"\b(uncle|aunt|auntie|aunty|ahjussi|ahjumma|mister)\b|叔父|伯父|叔母|伯母|"
                                                    r"おじさん|おばさん|おじちゃん|おばちゃん|叔叔|阿姨|伯伯|大伯|舅舅|舅妈|舅媽|婶婶|嬸嬸|"
                                                    r"姑姑|姨妈|외삼촌|외숙모|큰아버지|작은아버지|큰어머니|작은어머니|숙모|삼촌|이모|고모|아저씨|"
                                                    r"아줌마", re.I),
    "grandparent and grandchild": re.compile(r"\b(grandma|grandpa|grandmother|grandfather|granny|obaa-?chan|ojii-?chan|"
                                             r"halmeoni|harabeoji)\b|할머니|할아버지|"
                                             r"おばあ|おじい|じいちゃん|ばあちゃん|じっちゃん|ばっちゃん|じいじ|ばあば|爺ちゃん|"
                                             r"婆ちゃん|祖母|祖父|奶奶|爷爷|爺爺|外婆|外公|阿嫲|嫲嫲|阿爺|婆婆|公公", re.I),
    "parent and child": re.compile(r"\b(mom|mum|dad|mother|father|mommy|mummy|daddy|mama|papa|o?tou-?san|o?kaa-?san|appa|eomma)\b|ママ|パパ|^[母父]$|엄마|아빠|어머니|아버지|어머님|아버님|장모님|장인어른|丈母娘|岳母|岳父|お母|お父|(?<![叔伯])[母父]さん|妈妈|爸爸|媽媽|媽咪|爹哋|老豆|阿媽|阿爸|"
                                   r"(?:^|(?<=喂，)|(?<=喂,)|(?<=喂 )|(?<=喂？)|(?<=喂\?))(?:妈|爸|媽|娘|爹)(?= |[，,！!？?]|女儿|儿子|孩儿|我|您|你)|父亲|母亲|爹爹|娘亲|父上|母上|母后|父皇|母妃", re.I),
}
YOUNGER = re.compile(_VOCATIVE.format(r"(?:(?:little|kid|baby) (?:sis(?:ter)?|bro(?:ther)?)|사제|사매)") + r"|동생아|"
                     r"(?:^|[，,。！？!?]\s*)(?:[^\W\d的我你他她们們]{1,2})?小?(?:三弟|二弟|师弟|师妹|師弟|師妹|小妹|弟弟|妹妹|細佬|细佬|阿妹)(?=[，,。！？!? ])",
                     re.I)
# A parent or grandparent calling the child: "儿子，…", "乖孙", "Son, …", "아들, …".
TO_CHILD = {
    "grandparent and grandchild": re.compile(r"乖孙|乖孫|孙女|孫女|손자|손녀|孫よ", re.I),
    # Not "young man" (anyone older says it) or "Son, Kane and Kulusevski" (a name list).
    "parent and child": re.compile(_VOCATIVE.format(r"(?:son(?!\s*,\s*(?!(?:Mom|Dad|Mum)\b)(?-i:[A-Z])\w+\s*(?:,|and\b)))")
                                   + r"|儿子|兒子|闺女|閨女|아들아(?!이)|아들(?!아이)|딸아(?!이)|息子よ|^傻孩子|^孩子(?=[，,！!？? ])(?![，,]\s*是)|^丫头(?=[，,！!？? ])|"
                                   + _VOCATIVE.format("child"), re.I),
}
# What one spouse or lover calls the other; 老公 is said to a husband, 老婆 and 마누라 to a wife, 여보 to either.
COUPLE = {
    "male": re.compile(r"老公|夫君|相公|旦那様?(?=[、，,！!？?])|"
                       r"^\W*あなた(?=[、，,]\s*.*(?:おかえり|ご飯|ごはん|お風呂|家|行ってらっしゃい|起きて|ただいま))", re.I),
    "female": re.compile(r"老婆|娘子|마누라|\bwifey\b", re.I),
    "": re.compile(_VOCATIVE.format(r"(?:honey|babe|baby|sweetheart|darling|sweetie)") + r"|亲爱的|親愛的|ダーリン|여보(?!세요|게)|자기야",
                   re.I),
}
# Recognized Korean speech has no spaces: these open or close a line only as a call ("자기야오늘…", "네마마").
_KOREAN_GLUED_CALL = re.compile(r"^\W*(마마|중전마마|자기야|여보(?!세요)|아들아(?!이)|딸아(?!이))|(마마|자기야|여보)\W*$")
# Words a speaker uses for themself that fix both sides in Vietnamese, and what the other side answers with.
SELF_WORDS_CUES = [
    (re.compile("臣妾"), ("thần thiếp", "hoàng thượng"), ("trẫm", "ái phi")),
    (re.compile("朕"), ("trẫm", ""), ("", "hoàng thượng")),
    (re.compile("奴婢"), ("nô tỳ", ""), ("", "")),
    (re.compile("奴才"), ("nô tài", ""), ("", "")),
    (re.compile("晚辈|晚輩"), ("vãn bối", "tiền bối"), ("", "")),
    (re.compile("在下"), ("tại hạ", ""), ("", "")),
    (re.compile("老夫"), ("lão phu", ""), ("", "")),
    (re.compile("本宫|本宮"), ("bổn cung", "ngươi"), ("", "nương nương")),
    (re.compile("哀家"), ("ai gia", "ngươi"), ("", "Thái hậu")),
    (re.compile("儿臣|兒臣"), ("nhi thần", ""), ("", "hoàng nhi")),
    (re.compile("为师|為師"), ("ta", "con"), ("đệ tử", "sư phụ")),
    (re.compile("小女子"), ("tiểu nữ", ""), ("", "cô nương")),
    (re.compile(r"(?:^|(?<=[，,。！？!?\s]))小弟(?![，,。！？!?们們])"), ("tiểu đệ", "huynh"), ("huynh", "đệ")),
    (re.compile(r"^\W*소인(?=[,\s이은])"), ("tiểu nhân", ""), ("", "")),
]
CALLED_WORDS = {"爱妃": ("trẫm", "ái phi"), "愛妃": ("trẫm", "ái phi"), "爱卿": ("trẫm", "ái khanh"), "公子": ("", "công tử"),
                "姑娘": ("", "cô nương"), "小子": ("ta", "nhóc"), "嫂子": ("em", "chị"), "형부": ("em", "anh"),
                "大长老": ("", "trưởng lão"), "长老": ("", "trưởng lão"), "長老": ("", "trưởng lão"), "师叔": ("", "sư thúc"),
                "師叔": ("", "sư thúc"), "师伯": ("", "sư bá"), "師伯": ("", "sư bá"), "徒儿": ("ta", "con"), "徒兒": ("ta", "con")}
# "Baby, don't cry. Mommy's here.": a parent comforting a child, not a lover.
# A scanlator's gloss ("T/N: Hyung = older brother") explains words; nobody says it to anyone.
_TRANSLATOR_NOTE = re.compile(r"^\s*(?:T/?N|TL ?note|TN|Note|N/A|注|註|※|작가의\s*말|역주)\s*[:：]", re.I)
_PARENT_SELF = re.compile(r"\b(mommy|mummy|daddy|mom|mum|dad|your (?:mother|father|mom|dad))\b|엄마가|아빠가|"
                          r"(?:妈妈|爸爸)(?=在|回来|来了|给你|带你|陪你|帮你|爱你|会)",
                          re.I)
# Vietnamese has its own word for each side of the family: 舅舅 is cậu, 伯伯 bác, 이모 dì.
KIN_WORDS = [("舅舅|外삼촌|외삼촌", "cậu"), ("舅妈|舅媽|외숙모", "mợ"), ("婶婶|嬸嬸|숙모|작은어머니", "thím"),
             ("伯伯|伯父|大伯|伯母|큰아버지|큰어머니|伯父さん", "bác"), ("이모|姨妈|姨媽", "dì"), ("姑姑|고모", "cô"),
             ("叔叔|叔父|삼촌|작은아버지", "chú")]
ROYAL_PARENTS = {"父皇": "phụ hoàng", "母后": "mẫu hậu", "母妃": "mẫu phi"}
PRINCELY = re.compile(r"\b(prince|princess|highness)\b|殿下|王子|公主|姫様|왕자|공주|전하", re.I)
FEMALE_TITLES = re.compile(r"\b(princess|queen|empress|lady|milady|grandma|grandmother|granny|mom|mum|mother|aunt|auntie|"
                           r"aunty|noona|unnie|ahjumma|mommy|mummy|mama|o?kaa-?san|eomma|obaa-?chan|halmeoni|jiejie|shijie|miss|mrs|ms|woman|"
                           r"wife|girlfriend)\b|^母$|ママ|おばちゃん|婆ちゃん|舅妈|舅媽|婶婶|嬸嬸|伯母|큰어머니|작은어머니|외숙모|숙모|学姐|學姊|姊姊|장모님|丈母娘|岳母|ばあちゃん|阿嫲|嫲嫲|婆婆|媽|家姐|아줌마|어머님|娘|母亲|母上|姫様|叔母|伯母|おばさん|阿姨|姑姑|이모|고모|사저|사매|师姐|師姐|师妹|師妹|공주|"
                           r"할머니|엄마|어머니|아가씨|おばあ|お母|母さん|お嬢様|祖母|奶奶|外婆|妈|公主|母后|母妃|小姐", re.I)
MALE_TITLES = re.compile(r"\b(prince|king|emperor|lord|milord|grandpa|grandfather|dad|father|uncle|hyung|oppa|ahjussi|daddy|papa|o?tou-?san|appa|ojii-?chan|harabeoji|gege|shixiong|aniki|mister|mr|man|husband|boyfriend|bro)\b|^父$|パパ|爺ちゃん|大伯|学长|學長|장인어른|岳父|爺爺|じいちゃん|阿爺|公公|老豆|爹哋|大佬|아저씨|아버님|爹|父亲|父上|兄上|叔父|伯父|"
                         r"おじさん|叔叔|伯伯|舅舅|삼촌|사형|사제|师兄|師兄|师弟|師弟|왕자|할아버지|아빠|아버지|"
                         r"도련님|おじい|お父|父さん|お坊ちゃま|祖父|爷爷|外公|爸|王子|父皇|少爷", re.I)
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
    """One to three words, each capitalised unless written in a script without capitals ("Big Smoke", "リュウ")."""
    words = re.sub(r"^\[[A-Za-z]+\]\s*", "", label).split()
    return 0 < len(words) <= 3 and len(label) <= 24 and all(word[0].isupper() or not word[0].isascii() for word in words)


def body_of(text: str) -> str:
    """The line without its "Name:" label or name box, so a name such as "Sir Aldric" is not read as a cue."""
    match = _LABEL.match(text.strip())
    if match and _looks_like_name(match.group(1)):
        return match.group(2)
    if _bracketed_name(text):
        return _BRACKETED.match(text.strip()).group(2)
    name = speaker_of(text)
    if name:
        first, _, rest = text.strip().partition("\n")
        if first.strip() == name:
            return rest.strip()
    return text


_STAGE_DIRECTION = re.compile(r"[A-Z]+(?:S|ING|ED)|MUSIC|SILENCE|BEEP|BANG|笑|小声|回想|独白|心の声|ため息|泣|叫|内心|旁白|"
                              r"叹气|哭|喊|ナレーション|モノローグ|電話|电话|拍手|音乐|音楽", re.I)


def _bracketed_name(text: str) -> str | None:
    """【ハルカ】, （ハルカ） or (MOM) before a line; a bracketed aside such as （まずい……） or (Sighs) is not a speaker."""
    match = _BRACKETED.match(text.strip())
    if not match:
        return None
    name, opener = match.group(1).strip(), text.strip()[0]
    if opener in "【〔":
        return None if _not_a_name(name) else name
    if re.search(r"[.…!?！？。、,]", name) or _STAGE_DIRECTION.fullmatch(name):
        return None
    if opener == "（" and re.fullmatch("[\u30a0-\u30ff\u4e00-\u9fff]{1,6}", name):
        return name
    return name if name.isupper() and name.isascii() else None


_HEADING = re.compile(
    r"(?i:objectives?|warning|(?:main |side |daily |active |current )?quests?|backlog|history|log|missions?|tips?|hint|notice|rewards?|goals?|tutorial|new|live|legend|"
    r"video|audio|graphics|display|controls|settings|options|inventory|map|status|chat|name|level|class|hd|4k|"
    r"meanwhile|later|today|tonight|tomorrow|yesterday|that night|the next day|(?:the )?next morning|(?:\w+ )?(?:years?|months?|days?|hours?) later)|"
    r"S\d+\s*E\d+|\d+\+|\d{1,2}:\d{2}|TV-\w+|KBS\d?|MBC|SBS|JTBC|tvN|CCTV\d*|NHK|HBO|BBC|CNN|TBS|그날\s?밤|다음\s?날|한편|"
    r"那天晚上|第二天|その夜|翌日|次の日|一方(?:その頃)?|その頃|[\d一二三四五六七八九十几幾数半]+(?:年|个月|個月|天|ヶ月|か月|日)[后後]|"
    r"\d+\s?(?:년|개월|일)\s?후|第[\d一二三四五六七八九十百]+[集话話回章]|제\s?\d+\s?[화회장]|"
    r"나|저|너|당신|私|俺|僕|我|你|응|어|음|헐|흥|네|예|嗯|哦|啊|呃|唉|哼|喂|噢|欸|诶|うん|ええ|はあ?|"
    r".*(?:クエスト|퀘스트|バックログ|チャット|历史记录|任务|任務|目標|目标|報酬|报酬|奖励|獎勵|보상|채팅|聊天|초대|ボーナス|実績|업적|成就|"
    r"設定|设置|설정|メニュー|메뉴|菜单|ステータス|状态|旁白|ナレーション|나레이션).*|이름|名前|姓名|레벨|レベル|等级|직업|職業|职业")
_MENU_WORDS = {"fight", "attack", "run", "flee", "escape", "items", "item", "magic", "skills", "skill", "defend", "guard",
               "talk", "continue", "new game", "load game", "quit", "攻击", "逃跑", "たたかう", "にげる", "공격", "도망"}
# A family word over a narrative line is that sentence's subject ("DAD / NEVER CAME HOME", "妈妈 / 总是说…").
_NARRATION = re.compile(r"^(?i:never|always|used to|once|was|were|had|came|went|left|died|said|told|would)\b|"
                        r"^(?:总是|總是|总说|一直|从来|從來|从没|曾经|曾經|常常|经常|再也|生前)|"
                        r"(?:었|았|였|셨|했|됐|갔|왔|졌|웠|렸|냈|켰|봤|줬)다[.。!…]*$")
_SOUND_TAG = re.compile(r"^\s*[\[(（【][^\]\)）】\n]{1,40}[\]\)）】]\s*\n")
_CALL_SIGN = re.compile(r"^([A-Z][A-Z0-9]*(?:[- ][A-Z0-9]+)*-\d{1,3})\s*:\s*(\S.*)$", re.S)
_QUOTED_LINE = re.compile(r"^([゠-ヿ一-鿿A-Za-z・]{1,10})[「『](.+?)[」』]?$", re.S)
def _korean_vocative(row: str) -> bool:
    """지수야 / 민준아: a Korean given name with the calling ending (야 after a vowel, 아 after a consonant)."""
    if not re.fullmatch("[가-힣]{3}", row) or row[-1] not in "아야":
        return False
    closed = (ord(row[1]) - 0xAC00) % 28 != 0
    return closed == (row[-1] == "아")


def _not_a_name(first: str) -> bool:
    return bool(_HEADING.fullmatch(first) or first.lower() in _MENU_WORDS or re.search("[，、；？！。]|[你我您她他]", first)
                or _korean_vocative(first))


def speaker_of(text: str) -> str | None:
    """The name in "Name: line" or a short name on its own first line, if the text has one."""
    text = _SOUND_TAG.sub("", text.strip()).strip()
    match = _LABEL.match(text) or _CALL_SIGN.match(text)
    if match and _looks_like_name(match.group(1)) and not _not_a_name(match.group(1).strip()):
        return match.group(1).strip()
    bracketed = _bracketed_name(text)
    if bracketed:
        return bracketed
    quoted = _QUOTED_LINE.match(text)
    if quoted and "\n" not in quoted.group(1) and not _not_a_name(quoted.group(1)):
        return quoted.group(1)
    first, _, rest = text.partition("\n")
    first, rest = first.strip(), rest.strip()
    rows = [row.strip() for row in rest.split("\n")]
    # A menu ("攻击 / 技能 / 物品 / 逃跑") or a game's answers ("はい / いいえ") has no speaker.
    if first in CHOICE_WORDS or len(rows) >= 2 and all(len(row) <= 8 for row in [first, *rows]):
        return None
    if not rest or first.endswith(_SENTENCE_END) or not _looks_like_name(first) or not re.search(r"[^\W\d_]", rest):
        return None
    words = first.split()
    # "Then Jake / showed up" goes on with a sentence; "Jake / just got…" is a chat message's sender.
    if rest[:1].islower() and (len(words) > 1 or "\n" in rest):
        return None
    # A lone title over an all-caps line is a comic bubble calling someone ("MOM / I'M HOME!"), not a name tab.
    called = words[-1].lower().strip(",") in _VOCATIVE_TITLES and len(words) > 1
    if _not_a_name(first):
        return None
    # "Display Mode" or "Fast Travel Point" under a heading is a menu entry, not a line anyone says.
    if len(rest.split()) >= 2 and "\n" not in rest and all(word[:1].isupper() for word in rest.split()) \
            and not rest.isupper() and not rest.endswith(_SENTENCE_END):
        return None
    if (_elder_label(first) or first.lower() in _VOCATIVE_TITLES) and _NARRATION.search(rest):
        return None
    if words[0].lower().strip(",") in _DISCOURSE or called:
        return None
    return first


def vietnamese(facts: dict, speaker: str, listener: str) -> tuple[str, str]:
    """(word for I, word for you) when `speaker` talks to `listener`."""
    relation = str(facts.get("relation", ""))
    if facts.get("words_by"):
        own = facts["words_by"].get(speaker, ("", ""))
        reply = facts.get("reply_to", {}).get(listener, ("", ""))
        return own[0] or reply[0], own[1] or reply[1]
    older = facts.get("older", "")
    senior = facts.get("senior", "")
    rough = facts.get("rough") is True
    gender = lambda who: facts.get(f"{who} gender") or (
        "female" if FEMALE_TITLES.search(who) else "male" if MALE_TITLES.search(who) else "unknown")
    # With the elder's gender unknown, only the junior's "em" is certain; the model picks anh or chị.
    elder = lambda who: {"female": "chị", "male": "anh"}.get(gender(who), "")
    female = lambda who: gender(who) == "female"
    top = senior if senior in (speaker, listener) else older if older in (speaker, listener) else ""
    by_gender = lambda women, men: lambda who: {"female": women, "male": men}.get(gender(who), "")
    ranked = {"parent": by_gender("mẹ", "bố"), "grandparent": by_gender("bà", "ông"), "teacher": by_gender("cô", "thầy"),
              "aunt": (lambda who: facts["kin_word"]) if facts.get("kin_word") else by_gender("cô", "chú")}
    if relation.startswith("parent") and top and facts.get("royal_parent"):
        royal = facts["royal_parent"]
        return (royal, "hoàng nhi") if top == speaker else ("nhi thần", royal)
    for role, title in ranked.items():
        if relation.startswith(role) and top:
            young = "con" if role == "parent" else "cháu" if role in ("grandparent", "aunt") else "em"
            return (title(speaker), young) if top == speaker else (young, title(listener))
    if relation.startswith("master") and top:
        return ("ta", "con") if top == speaker else ("đệ tử", "sư phụ")
    if relation.startswith("ruler") and top:
        servant = re.search(r"宫女|宮女|丫鬟|侍女|婢|太监|maid", speaker, re.I) or facts.get("serving")
        consort = re.search(r"중전|왕비|皇后|贵人|貴人|妃|嫔|嬪", speaker)
        return ("ta", "ngươi") if top == speaker else (
            "thần thiếp" if consort else "nô tì" if servant else "thần", facts.get("royal", "bệ hạ"))
    if relation.startswith("noble") and top:
        if top == speaker:
            return "", ""
        return "tôi", {"female": "tiểu thư", "male": "ngài"}.get(gender(listener), "ngài")
    if "customer" in relation:
        if senior == listener:
            return "tôi", {"female": "chị", "male": "anh"}.get(gender(listener), "bạn")
        return "tôi", "quý khách"
    if relation == "couple" and older not in (speaker, listener):
        if "female" in (gender(speaker), gender(listener)) or "male" in (gender(speaker), gender(listener)):
            husband = speaker if gender(speaker) == "male" or gender(listener) == "female" else listener
            return ("anh", "em") if husband == speaker else ("em", "anh")
        return "", ""
    if facts.get("martial") and older in (speaker, listener):
        if older == speaker:
            return by_gender("sư tỷ", "sư huynh")(speaker), by_gender("sư muội", "sư đệ")(listener)
        return by_gender("muội", "đệ")(speaker), by_gender("sư tỷ", "sư huynh")(listener)
    if relation in ("family", "siblings or cousins", "couple", "school senior and junior") and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation == "friends" and not rough and older in (speaker, listener):
        return (elder(speaker), "em") if older == speaker else ("em", elder(listener))
    if relation == "friends" or rough:
        return ("tao", "mày") if rough else ("tớ", "cậu")
    if relation in ("coworkers", "boss and employee") and top:
        return (elder(speaker), "em") if top == speaker else ("em", elder(listener) or "sếp")
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


    def terms(self, source_lang: str, text: str = "") -> dict[str, str]:
        """Glossary entries for translation models that follow terms but not instructions (HY-MT).

        Only words the line holds: HY-MT applies a term to whatever is nearest, so "我 → anh" on a line that only calls
        "おじいちゃん" turns the grandfather into the speaker. Korean 나/내/네 are also "yes" and parts of words.
        """
        langs = [source_lang] if source_lang in SELF_WORDS else list(SELF_WORDS)
        def present(word: str) -> bool:
            if not text:
                return True
            if word.isascii():
                return bool(re.search(rf"\b{re.escape(word)}\b", text, re.I))
            return word in text and not (len(word) == 1 and re.match("[가-힣]", word))

        terms = {word: self.me for lang in langs for word in SELF_WORDS[lang] if present(word)} if self.me else {}
        if self.you:
            terms.update({word: self.you for lang in langs for word in YOU_WORDS[lang] if present(word)})
        title = called_title(text) if text else ""
        if title:
            terms[title] = self.you or ({"学姐": "chị", "学长": "anh"}.get(title)
                                        or ("tiền bối" if TO_ELDER[""].search(title) else ""))
        # The word that set the pronouns ("晚辈" → vãn bối, "前辈" → tiền bối), so HY-MT puts them where they belong.
        for cue, (me, you), _reply in SELF_WORDS_CUES:
            found = cue.search(text) if text else None
            if found and me == self.me:
                terms[found.group(0)] = me
        own = {"前辈": "tiền bối", "前輩": "tiền bối", "姑娘": "cô nương", "皇上": "hoàng thượng", "娘娘": "nương nương",
               "太后": "Thái hậu", "参见": "bái kiến", "拜见": "bái kiến", "叩见": "bái kiến"}
        for word, term in own.items():
            if text and word in body_of(text) and word not in terms:
                terms[word] = term
        # "学长,这道题能教教我吗?": "me" as the object of a request, with no "you", HY moves onto the subject ("em … em").
        if title and not any(present(word) for lang in langs for word in YOU_WORDS[lang]):
            terms = {word: term for word, term in terms.items()
                     if not re.search(f"(?:教|帮|给|告诉|陪|带){re.escape(word)}", text)}
        return {word: term for word, term in terms.items() if term}


def _elder_label(who: str) -> bool:
    return any(ROLES[role].search(who) for role in ("parent and child", "grandparent and grandchild",
                                                     "aunt or uncle and niece or nephew"))


def called_title(line: str) -> str:
    """The form of address the line calls its listener by ("おじいちゃん", "사저", "Young Master"), if any."""
    body = body_of(line)
    cues = [*ROLES.values(), *TO_ELDER.values(), YOUNGER, *COUPLE.values()]
    for cue in cues:
        found = cue.search(body)
        # "作为妈妈，…" is the speaker's own role, not a call.
        if found and calls_by(cue, body) and not re.search("(?:作为|作為|身为|身為)$", body[:found.start()]):
            suffix = re.match("ちゃん|さん|様|さま|くん|君|님", body[found.end():])
            return (found.group(0) + (suffix.group(0) if suffix else "")).strip(" ,.!?~…、，！？")
    return ""


def _address(speaker: str, listener: str, facts: dict | None, guessed: bool) -> Address | None:
    pair = vietnamese(facts, speaker, listener) if facts else ("", "")
    return Address(speaker, listener, *pair, guessed) if any(pair) else None


def guess_facts(line: str, speaker: str, listener: str) -> dict | None:
    """Facts plain from one line: calling the listener big brother or 先輩, serving a customer, swearing."""
    line = re.sub(r"^[“\"‘]+", "", body_of(line).replace("\u3000", " "))
    # "手紙には、お姉ちゃん、頑張って、とだけ書いてありました": narration quoting a call.
    if re.search(r"[、,]\s*と(?:だけ)?(?:書い|言っ|言わ|叫ん|呟い|つぶやい)", line):
        return None
    if SERVING.search(line) and not called_title(line):
        return {"relation": "staff and customer", "senior": speaker, "guess": True}
    for cue, words, reply in SELF_WORDS_CUES:
        if cue.search(line):
            # "娘娘，奴婢该死": the self word fixes "I", and a title in the same line still fixes "you".
            # "臣妾参见太后": a title called in the line outranks the self word's usual listener.
            other = guess_facts(cue.sub("", line), speaker, listener)
            you = (vietnamese(other, speaker, listener)[1] if other else "") or _called_word(line)[1] or words[1]
            return {"relation": "set words", "words_by": {speaker: (words[0], you)}, "reply_to": {speaker: reply},
                    "guess": True}
    younger = YOUNGER.search(line)
    if younger:
        martial = re.search("师|師|사제|사매", younger.group(0))
        facts = {"relation": "siblings or cousins", "older": speaker, "guess": True, "martial": bool(martial)}
        gender = "female" if FEMALE_TITLES.search(younger.group(0)) else "male" if MALE_TITLES.search(younger.group(0)) else ""
        own = _self_title(line)
        if own:
            own_gender = "female" if FEMALE_TITLES.search(own) else "male" if MALE_TITLES.search(own) else ""
            facts = {**facts, f"{speaker} gender": own_gender} if own_gender else facts
        return {**facts, f"{listener} gender": gender} if martial and gender else facts
    own = _self_title(line)
    if own:
        return _elder_facts(own, speaker)
    me, you = _called_word(line)
    if you:
        return {"relation": "set words", "words_by": {speaker: (me, you)}, "reply_to": {speaker: (you if me else "", me)},
                "guess": True}
    # A doctor called 선생님/先生, or お母さん said to a patient's or pupil's mother, is not a teacher or a parent.
    # A parent to the doctor or teacher about their child, or a doctor to a patient: no teacher cue.
    clinic = re.search(r"수술|진료|환자|처방|퇴원|의사|병원|입원|우리 ?(?:아이|애|딸|아들)|제 ?(?:딸|아들)|手術|診察|患者|病院|熱|"
                       r"うちの(?:子|娘|息子)|(?:息子|娘)[はの]|手术|医生|医院|病人|发烧|我家孩子|我的孩子|我(?:儿子|女儿)", line)
    # 어머님/お母さん said to someone else's parent ("お子さんは…", "처음 뵙겠습니다"): no parent cue.
    # A doctor or teacher to a patient's or pupil's parent: 어머님 with clinic or school words, お母さん in polite
    # speech with them, 妈妈 with 您 about 孩子.
    # Only phrases a doctor or teacher would use: a daughter-in-law also says "어머님, 걱정 마세요" or mentions 약.
    others = re.search(r"お子さん|息子さん|娘さん|お嬢さん|아드님|따님|처음 ?뵙|您的孩子|你家孩子|您家孩子|您孩子|孩子(?:妈|爸|媽)|"
                       r"孩子的检查|检查结果|开点药|打针|退烧药|住院观察|感冒|出しておきます|処方|三者面談|面談|"
                       r"検査しましょう|様子を見ましょう|お大事に", line) or (
        re.search("어머님|아버님", line) and re.search(r"수술|진료|환자|퇴원|검사 결과|처방|입원|수업|상담|담임|학생|성적|감기|약 지어|"
                                                     r"아이[는가]", line))
    for role, cue in ROLES.items():
        if clinic and role.startswith("teacher") or others and role.startswith("parent"):
            continue
        title = cue.search(line)
        # Only the speaker being that role already: "我也是第一次当妈妈", not "你要当爷爷了" or "我要当爸爸了".
        opening = re.match(r"\W*([^，,、\s]{1,4})[，,、]", line)
        # "爸，我当爸爸以后才明白你有多辛苦" talks to Dad ("你"); "妈妈,我也是第一次当妈妈,你们别吵了" is Mom to the kids.
        other_call = opening and title and not re.match(r"\W*(?:作为|作為|身为|身為)", line) and (title.group(0) not in opening.group(1) or re.search("[你您](?!们|們)", line))
        named = re.escape(title.group(0)) if title else ""
        being = title and (re.search(f"我(?:(?![你要想会會将將]).){{0,6}}?(?:当|做|作为|當|作為)(?:了[^，,。]{{0,5}})?{named}", line)
                           or re.match(f"\\W*(?:作为|作為|身为|身為){named}(?![，,]\\s*你)", line))
        # "老公，我做妈妈的…": the call to someone else decides the pair, not the speaker's own role.
        if being and not other_call:
            # "妈妈,我也是第一次当妈妈": the speaker is the one being a mother.
            gender = "female" if FEMALE_TITLES.search(title.group(0)) else "male" if MALE_TITLES.search(title.group(0)) else ""
            facts = {"relation": role, "senior": speaker, "older": speaker, "guess": True}
            return {**facts, f"{speaker} gender": gender} if gender else facts
        if calls_by(cue, line):
            # "爷爷,我年轻的时候…": the elder's own youth after the call; either of them may be talking.
            if re.search("我(?:年轻|年輕|当年|當年)", line):
                return None
            facts = {"relation": role, "senior": listener, "older": listener, "guess": True}
            title = cue.search(line).group(0)
            if role.startswith("aunt"):
                facts["kin_word"] = next((word for pattern, word in KIN_WORDS if re.search(pattern, title)), "")
            if role.startswith("parent") and title in ROYAL_PARENTS:
                facts["royal_parent"] = ROYAL_PARENTS[title]
            if role.startswith("ruler"):
                facts["serving"] = bool(re.search("用膳|更衣|伺候|就寝|沐浴|梳妆|梳妝", line))
                facts["royal"] = ("Thái hậu" if "太后" in title else "nương nương" if re.search("娘娘|마마", title)
                                  else "hoàng thượng" if "皇上" in title
                                  else "điện hạ" if PRINCELY.search(title) else "bệ hạ")
            gender = "female" if FEMALE_TITLES.search(title) else "male" if MALE_TITLES.search(title) else ""
            return {**facts, f"{listener} gender": gender} if gender else facts
    taunt = re.search(r"\b(?:owned|rekt|pwned|let'?s go|get good|sit down)\W+son\b", line, re.I)
    for role, cue in TO_CHILD.items():
        if calls_by(cue, line) and not taunt:
            # Grandpa saying "kiddo" is still a grandparent.
            labelled = next((r for r in ("grandparent and grandchild", "aunt or uncle and niece or nephew")
                             if ROLES[r].search(speaker)), role)
            return {"relation": labelled, "senior": speaker, "older": speaker, "guess": True}
    for gender, cue in TO_ELDER.items():
        found = cue.search(line)
        if not found:
            continue
        martial = bool(re.search("师|師|사형|사저", found.group(0)))
        claimed = re.search(f"我(?:做|当|當|作为|作為)(?:了)?{re.escape(found.group(0))}", line[found.end():])
        if (claimed or re.match(r"\W*(?:作为|作為|身为|身為)$", line[:found.start()])) and gender and not re.match(
                r"\W*你", line[found.end():]):
            # "身为哥哥，我要保护你": the elder sibling speaking as one.
            return {"relation": "school senior and junior", "older": speaker, f"{speaker} gender": gender,
                    "guess": True, "martial": martial}
        if calls_by(cue, line):
            facts = {"relation": "school senior and junior", "older": listener, "guess": True, "martial": martial}
            return {**facts, f"{listener} gender": gender} if gender else facts
        # 先輩 and 선배님 are said only by juniors, wherever they fall in the line, unless someone else is meant.
        if not gender and not _MENTIONED.search(line[:found.start()]) and not re.match(
                r"(?:が|は|も|の|가|는|이|께서)", line[found.end():]):
            return {"relation": "school senior and junior", "older": listener, "guess": True}
        # An elder speaking of themself to the listener: "师兄替你挡着", "お姉ちゃんが守ってあげる", "오빠만 믿어".
        opening = not line[:found.start()].strip(" \"'「『(-—…")
        glued_i = re.match(r"\s*(?:我|私|僕|俺|나|내|저)", line[found.end():])
        if opening and gender and not glued_i and _SECOND_PERSON.search(line[found.end():]):
            return {"relation": "school senior and junior", "older": speaker, f"{speaker} gender": gender,
                    "guess": True, "martial": martial}
    for gender, cue in COUPLE.items():
        if calls_by(cue, line):
            parent = _PARENT_SELF.search(line)
            if parent and not parent.group(0).lower().startswith("your"):
                gender = "female" if FEMALE_TITLES.search(parent.group(0)) else "male"
                return {"relation": "parent and child", "senior": speaker, "older": speaker,
                        f"{speaker} gender": gender, "guess": True}
            if parent:
                return None
            spouse = bool(gender) or bool(re.search("여보|자기야|亲爱的|親愛的|ダーリン", line))
            facts = {"relation": "couple", "guess": True, "spouse": spouse}
            return {**facts, f"{listener} gender": gender} if gender else facts
    if ROUGH.search(line):
        return {"relation": "friends", "rough": True, "guess": True}
    return None


_SECOND_PERSON = re.compile(r"你|您|お前|おまえ|あなた|君|きみ|너|네가|널|\byou\b", re.I)
_CALLED_WORD = re.compile("|".join(sorted(CALLED_WORDS, key=len, reverse=True)))
_ELDER = (r"お姉ちゃん|お兄ちゃん|姉ちゃん|兄ちゃん|お母さん|お父さん|ママ|パパ|おばあちゃん|おじいちゃん|先生|"
          r"妈妈|爸爸|姐姐|哥哥|爷爷|奶奶|老师|师兄|师姐|師兄|師姐|"
          r"할머니|할아버지|엄마|아빠|오빠|언니|누나|형|선생님")
# At a line's start only the clearest forms count: a child also says "爸爸给你买了…" about Dad.
_SELF_TITLE = re.compile(
    rf"^\W*(?:(?P<a>(?:お)?(?:姉|兄)ちゃん)に任せ|(?P<b>{_ELDER})(?:在这|在這|在呢)|(?P<c>{_ELDER})我(?=可是|当年|當年|年轻|以前|这辈子))|"
    rf"(?:(?<=[，,。！!？?、])|(?<=は))\s*(?P<d>{_ELDER})(?:が(?:いる|ついて|守|.{{0,12}}?(?:あげる|やる))|に任せ)|"
    rf"(?<=[，,。！!？?])\s*(?P<e>{_ELDER})我?(?=给你|給你|在这|在這|在呢|会(?:保护|陪|帮|一直)|會(?:保護|陪|幫)|帮你|陪你|带你|替你|请你|保护)|"
    rf"(?P<f>{_ELDER})(?:가|이)\s?(?:(?!내가|제가)[가-힣\s]){{0,15}}?(?:줄게|할게)|(?P<g>{_ELDER})만\s?믿|"
    r"^\W*(?P<j>お母さん|お父さん|ママ|パパ|先生|お姉ちゃん|お兄ちゃん|おばあちゃん|おじいちゃん)は(?!.*(?:って(?:言|[。、！!]|$)|そうだ|らしい))(?=.*(?:よ|わ|ぞ|からね)\W*$)|"
    r"我(?:是你|这个当|這個當)(?P<h>妈|爸|媽|姐|哥|爷爷|奶奶|老师|师父)|\byour old (?P<i>grandpa|grandma|granny)\b", re.I)


def _self_title(line: str) -> str:
    found = _SELF_TITLE.search(line)
    return next((title for title in found.groups() if title), "") if found else ""


def _elder_facts(title: str, speaker: str) -> dict:
    role = next((role for role, cue in ROLES.items() if cue.search(title)), None)
    if role is None and title in ("妈", "爸", "媽"):
        role = "parent and child"
    gender = ("female" if FEMALE_TITLES.search(title) or TO_ELDER["female"].search(title) else
              "male" if MALE_TITLES.search(title) or TO_ELDER["male"].search(title) else "")
    facts = {"relation": role or "school senior and junior", "senior": speaker if role else "neither", "older": speaker,
             "guess": True, "martial": bool(re.search("师|師", title))}
    return {**facts, f"{speaker} gender": gender} if gender else facts


def _called_word(line: str) -> tuple[str, str]:
    found = _CALLED_WORD.search(line)
    return CALLED_WORDS[found.group(0)] if found and calls_by(_CALLED_WORD, line) else ("", "")
_PARENT_WORD = re.compile(r"엄마|아빠|어머니|아버지")
_HONORED = re.compile(r"셨|시[어었고는지나네]|신다|계셔|계세|께서|주무|드시")


def calls_by(cue: re.Pattern, line: str) -> bool:
    """Whether a title in the line calls the listener ("Mom, I'm home", "Your Highness!") rather than names someone."""
    glued = _KOREAN_GLUED_CALL.search(line)
    for match in cue.finditer(line):
        if glued and match.start() == glued.start(glued.lastindex):
            return True
        before, after = line[:match.start()], line[match.end():]
        suffix = re.match(r"(?:さん|ちゃん|様|さま|くん|君|大人)", after)
        if suffix:
            after = after[suffix.end():]
        # "Uncle Joe, can I…", "Thank you, Grandpa Tobin.": the name after the title belongs to the call.
        named = match.group(0).isascii() and re.match(r"\s+[A-Z][\w'-]+(?=\s*(?:[,!?.]|$))", after)
        if named:
            after = after[named.end():]
        # "누나는 왜 이렇게…?": a sibling title as "you" in a question to them.
        asked = re.match(r"는\s*왜", after) and line.rstrip().endswith("?")
        # Palace speech names the king in the third person: "전하께서 … 낫사옵니다".
        palace_third = after.startswith("께서") and re.search("[사시하]옵", line)
        honorific = _HONORIFIC.fullmatch(match.group(0).strip()) and re.search(r"\byour\s+$", before, re.I)
        particle = re.match(r"(?:가|는|를|도|의|한테|랑|이(?![거것제])|에게|께|하고)", after) and not asked and not palace_third
        if particle or (_MENTIONED.search(before) and not honorific):
            continue
        if honorific:
            before = re.sub(r"\byour\s+$", "", before, flags=re.I)
        # A parent speaking of themself: "お母さんはね…", "爸爸给你买了…".
        if re.match(r"(?:が|は|に|を|も|の|ね|って|と|给|带|帮|陪|会|来|去|也|就|都)", after):
            continue
        if re.search("(?:を|に|が)$", before) or re.search("(?:作为|作為|身为|身為)$", before):
            continue
        hangul = re.search("[\uac00-\ud7a3]", match.group(0))
        cjk = re.search("[\u3040-\u30ff\u3400-\u9fff]", match.group(0))
        unprefixed = re.sub("(?:启禀|禀告|参见|拜见|叩见|多谢|谢谢|谢|恭送|叩谢|恭迎|求|恳请|还请|敢问)$", "", before)
        thanked = unprefixed != before and not unprefixed.strip(" \"'「『(-—…")
        # A Chinese or Japanese clause can open with the title too: "臣妾知错，求太后恕罪".
        opening = not unprefixed.strip(" \"'「『(-—…") or bool(cjk and re.search(r"[，,。！!？?]\s*$", unprefixed))
        # A subtitle often sets a call apart with a space instead of a comma: "妈 我回来了", "师兄 小心".
        spaced = re.match(r"[\u3400-\u9fff]", match.group(0)[-1:]) and re.match(r"[ \u3000]+[\u3400-\u9fff]", after)
        closing = not after.strip() or after.lstrip()[:1] in ",!?.、，！？~…。" or bool(spaced) or bool(asked)
        # Palace speech opens with the title as "you": "母后恕罪", "陛下息怒".
        palace = re.fullmatch("母后|父皇|母妃|陛下|皇上|太后|爹|娘|爸|妈|傻孩子|爱妃|愛妃|爱卿|公子|姑娘", match.group(0))
        # "哥你…", "师父弟子…", "奶奶我…": an unpunctuated subtitle still calls when "you" or "I" comes next.
        # A title before 我 with 你 later is either side ("姐姐我请你" from the elder, "姐姐我好想你" from the younger):
        # no cue for it either way.
        speaks_on = re.match(r"\s*(?:你|您|ただいま|おかえり|おはよう|待って|見て|聞いて|ごめん|ありがとう|助けて|大丈夫|"
                             r"この|その|これ|それ|お疲れ|おかわり|すみません|ちょっと|早く(?=[来行起帰早し！!、。…~\s]|$)|"
                             r"(?:来て|起きて|行こう|して|待って|見て)(?:よ|ね|ってば|ください|(?=[！!、。…~\s]|$))|"
                             r"小心|快跑|快走|快来|救我|等等|等我|别走|住手|加油|请|好(?=[\W]|$)|"
                             r"(?:师父|师傅|師父|老师|爸|妈)(?=[，,]|让|叫|找)|"
                             r"(?i:look out|watch out|help|wait|run|stop|hurry|please|wake up|get up)\b)(?!って|と言)", after) \
            and not re.search("かな(?![いりく])", after) or (
            re.match(r"\s*(?:我|弟子|徒儿|女儿|儿子|孩儿|儿臣|臣妾|奴婢)", after) and not re.search("你|请客|買單|买单", after))
        if thanked:
            return True
        if not hangul and opening and not closing and not palace and not speaks_on:
            continue
        # "엄마아빠다오셨어", "어머니아까전화하셨는데": a parent mentioned with an honorific verb, not called.
        if hangul and not closing and not palace_third and (_PARENT_WORD.match(after) or _HONORED.search(after)):
            continue
        # "엄마 나 오늘 늦어" calls; "어머니병원에 갔었어" or "어머니 말씀이" mentions: a call goes on with "I", "you", "why", "hurry".
        if hangul and opening and not closing and not palace_third and not _KOREAN_CALL_ON.match(after):
            continue
        if opening or closing:
            return True
    return False


# Words before a title that make it a mention: "your mother", "like a mother", "手紙には", "우리 엄마".
_MENTIONED = re.compile(r"(?:\b(?:your|my|his|her|our|their|the|a|an|like|this|that|told|said|says|with|to|for|ask|"
                        r"tell|visit|help|see|call)\s+|(?<![あこそど])の|には|って|と|우리\s*|내\s*|네\s*|告诉|跟|给|找|叫|让|(?<!敢)问|对|和|(?<![好是对])的|(?<![参拜叩])见|"
                        r"护住|保护|救|帮|跟着)$", re.I)
# Titles that only ever come with "your", so "Your Majesty" calls the ruler where "your mother" only mentions one.
_HONORIFIC = re.compile(r"(?:majesty|highness|grace|excellency|lordship|ladyship)", re.I)


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
    called: dict[str, str] = {}
    for said_by, line in said:
        other = b if said_by == a else a if said_by == b else None
        if other and other not in roles:
            role = next((role for role, cue in ROLES.items() if calls_by(cue, body_of(line))), None)
            if role:
                roles[other] = called[other] = role
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
    # The emperor still calls his mother 母后: being called parent outranks a title in the other's name.
    family = [name for name, role in called.items() if role.startswith(("parent", "grandparent"))]
    if len(family) == 1 and len(roles) == 2:
        roles = {family[0]: roles[family[0]]}
    dowager = [name for name in roles if "太后" in name]
    if len(roles) == 2 and len(dowager) == 1:
        roles = {dowager[0]: "parent and child" if all(r.startswith("ruler") for r in roles.values()) else roles[dowager[0]]}
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


NOBODY = "?"


def _replier(lines: list, index: int) -> str | None:
    who = lines[index][0]
    return next((w for w, _l, _a, _t in lines[index + 1:] if w and w != who), None)
_KOREAN_CALL_ON = re.compile(r"\s*(?:나|저|제가|내가|식사|밥|우리|너|왜|빨리|어디|이거|지금|야(?=\s*(?:[,.!?~…]|$))|잠깐|조심|걱정|같이)")


def _calls_a_title(body: str) -> bool:
    return (any(calls_by(cue, body) for cue in ROLES.values()) or any(cue.search(body) for cue in TO_ELDER.values())
            or bool(YOUNGER.search(body)))


def _calls_name(line: str, name: str) -> bool:
    """A name opening the line or set off by punctuation calls that person; one inside the sentence only mentions them."""
    title = r"(?:(?:Uncle|Aunt|Mr|Mrs|Ms|Miss|Dr|Captain|Sergeant|Sir|Lady)\.?\s+)?"
    suffix = r"(?:\s?(?:씨|님)|さん|君|くん|ちゃん|様|先輩|先生|-sensei|-san|-kun|-chan|아|야)?"
    return bool(re.search(rf"(?:^|[,、，!！?？]\s*)(?:启禀|禀告|参见|拜见|叩见)?{title}{re.escape(name)}{suffix}\s*(?:[,、，!！?？~…。.]|$)",
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
        # "대리님" calls the one speaker whose label ends in that title ("최 대리").
        titled = {who for title in re.findall(r"([가-힣]{1,3})님", body) for who in self._spoken
                  if who != speaker and who.endswith(title)}
        if len(titled) == 1:
            return titled.pop()
        cutoff = self.clock() - self.window_s
        recent = [who for who, _text, at, _to in reversed(self._lines) if who and who != speaker and at >= cutoff]
        if not self._spoken[speaker]:
            # A newcomer's first line answers whoever just spoke, unless it calls someone by a title ("Grandpa!").
            return None if _calls_a_title(body) else recent[0] if recent else None
        return next((who for who in recent if self._spoken[who] >= 2), recent[0] if recent else None)

    def address(self, text: str, speaker_gender: str = "") -> Address | None:
        body = body_of(text)
        if _TRANSLATOR_NOTE.match(text) or " = " in text:
            return None
        if self.settings().target_lang not in SUPPORTED:
            return None
        if GROUP.search(body) and not _calls_a_title(body):
            return None
        # Two speakers in one subtitle ("- Where were you?\n- At home."): one note cannot fit both.
        if re.search(r"\n\s*[-–－]", text):
            return None
        speaker = speaker_of(text)
        if not speaker:
            facts = guess_facts(text, "the speaker", "the listener")
            # Parents say "sweetheart" and "baby" too: only words spouses alone use take the voice's gender.
            if facts and speaker_gender and (facts.get("relation") != "couple" or facts.get("spouse")):
                facts = {"the speaker gender": speaker_gender, **facts}
            return _address("the speaker", "the listener", facts, True)
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
        if facts and facts.get("relation") == "couple" and listener and _elder_label(listener):
            # Mia answering Grandma, who called her "sweetie": the child talks up to her grandmother.
            role = next(role for role, cue in ROLES.items() if cue.search(listener))
            facts = {"relation": role, "senior": listener, "older": listener, "guess": True}
        if facts is None or (facts.get("relation") == "couple" and facts.get("guess")):
            # A speaker labelled "Mom" or "先生" talks down to whoever they are talking to, "honey" or not.
            role = next((role for role, cue in ROLES.items() if cue.search(speaker)), None)
            if role and role.startswith(("parent", "grandparent", "teacher", "aunt")):
                facts = {"relation": role, "senior": speaker, "older": speaker, "guess": True}
                gender = "female" if FEMALE_TITLES.search(speaker) else "male" if MALE_TITLES.search(speaker) else ""
                facts.update({f"{speaker} gender": gender} if gender else {})
            elif STAFF.search(speaker):
                facts = {"relation": "staff and customer", "senior": speaker, "guess": True}
            elif TO_ELDER[""].search(speaker):
                facts = {"relation": "school senior and junior", "older": speaker, "guess": True}
        elif listener:
            # A clear form of address in this very line ("Sis", "big brother", 오빠, 大哥) outranks what was stored.
            called = guess_facts(text, speaker, listener)
            if called and called.get("words_by"):
                # "儿臣…", "臣妾…": the speaker's own word for themself in this line holds whatever the judge said.
                facts = {**facts, "words_by": {**facts.get("words_by", {}), **called["words_by"]},
                         "reply_to": {**facts.get("reply_to", {}), **called["reply_to"]}}
            elif called and (called.get("older") or called.get("senior")) and not facts.get("words_by"):
                facts = {**facts, **{key: value for key, value in called.items() if key != "guess"}}
        return _address(speaker, listener, facts, bool(facts.get("guess"))) if facts else None

    def _said(self, pair: frozenset) -> list[tuple[str, str]]:
        lines = list(self._lines)
        return [(who, line) for index, (who, line, _a, to) in enumerate(lines) if who in pair
                and ((to in pair and to != who) or (to == NOBODY and _replier(lines, index) in pair - {who}))]

    def _note_cues(self, pair: frozenset) -> None:
        """Cues from every line either of the two said to the other so far, the latest ones winning."""
        lines = list(self._lines)
        for index, (who, line, _at, to) in enumerate(lines):
            if who in pair and (to in (None, NOBODY) or (to in pair and to != who)):
                other = next(iter(pair - {who}), None)
                if other is None:
                    continue
                if to in (None, NOBODY):
                    # A line said before anyone answered counts only for the one who answered it.
                    replier = _replier(lines, index)
                    if replier is not None and replier != other:
                        continue
                guess = guess_facts(line, who, other)
                # Grandma's "sweetie" to Mia: an elder's endearment, not a couple.
                if guess and guess.get("relation") == "couple" and _elder_label(who):
                    guess = None
                known = self._cues.get(pair, {})
                # One insult between siblings, or one rough line at all, does not make two people street rivals.
                rough_lines = sum(1 for w, l, _a, t in self._lines if w in pair and ROUGH.search(body_of(l)))
                if guess and guess.get("rough") and (known.get("older") or known.get("senior") or rough_lines < 2):
                    continue
                if guess and guess.get("words_by"):
                    # Each side's own words are kept: 臣妾 from the consort, 朕 from the emperor.
                    words_by = {**known.get("words_by", {}), **guess["words_by"]}
                    reply_to = {**known.get("reply_to", {}), **guess["reply_to"]}
                    self._cues[pair] = {**known, **guess, "words_by": words_by, "reply_to": reply_to, "pair": tuple(pair)}
                elif guess and (guess.get("older") or not known.get("older")) and not known.get("words_by"):
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
            # A title called with no known listener ("Grandpa!") belongs to no pair: marked so no pair takes its cue.
            others = any(who and who != speaker for who, _t, _a, _to in self._lines)
            to = listener or (NOBODY if speaker and others and _calls_a_title(body_of(text)) else None)
            self._lines.append((speaker, text.strip(), self.clock(), to))
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
