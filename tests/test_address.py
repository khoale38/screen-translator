import types

from screen_translator.address import Address, Relationships, guess_facts, merged, speaker_of, vietnamese
from screen_translator.context import ContextWindow
from screen_translator.translator import build_messages, for_hy_mt


def test_speakers_come_from_name_labels_but_not_sentences():
    assert speaker_of("Lamar: You owe me fifty bucks.") == "Lamar"
    assert speaker_of("リュウ：うるせぇ。") == "リュウ"
    assert speaker_of("Franklin\nShut up and drive!") == "Franklin"
    assert speaker_of("Wanted level 3") is None
    assert speaker_of("He said: run, now and never come back") is None
    assert speaker_of("Press E to enter the vehicle") is None


def test_facts_become_vietnamese_pronouns_both_ways():
    siblings = {"relation": "family", "older": "Minh", "Minh gender": "male", "Linh gender": "female"}
    assert vietnamese(siblings, "Linh", "Minh") == ("em", "anh")
    assert vietnamese(siblings, "Minh", "Linh") == ("anh", "em")
    homies = {"relation": "friends", "rough": True}
    assert vietnamese(homies, "Franklin", "Lamar") == ("tao", "mày")
    assert vietnamese({"relation": "friends", "rough": False}, "A", "B") == ("tớ", "cậu")
    assert vietnamese({"relation": "staff and customer"}, "Shopkeeper", "Player") == ("tôi", "quý khách")
    couple = {"relation": "couple", "older": "unknown", "Nam gender": "male", "Lan gender": "female"}
    assert vietnamese(couple, "Lan", "Nam") == ("em", "anh")
    assert vietnamese({"relation": "strangers", "B gender": "female"}, "A", "B") == ("tôi", "chị")


def test_a_plain_cue_settles_who_is_older_when_the_model_could_not():
    cues = {**guess_facts("先輩、待ってください！", "Mika", "Ryu"), "pair": ("Mika", "Ryu")}
    judged = {"relation": "friends", "older": "unknown", "rough": False}
    assert vietnamese(merged(judged, cues), "Mika", "Ryu") == ("em", "")
    assert vietnamese(merged({**judged, "Ryu gender": "male"}, cues), "Mika", "Ryu") == ("em", "anh")
    assert vietnamese(merged({**judged, "older": "Ryu", "Ryu gender": "male"}, cues), "Ryu", "Mika") == ("anh", "em")


class Judge:
    def __init__(self, answer):
        self.questions, self._answer = [], answer

    def answer(self, question):
        self.questions.append(question)
        return self._answer


def wait_for(condition, timeout=3.0):
    import time

    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    assert condition()


def test_each_pair_is_judged_in_the_background_and_its_pronouns_used_after():
    judge = Judge('{"relation": "family", "older": "Minh", "Minh gender": "male", "Linh gender": "female", '
                  '"rough": false}')
    events = []
    settings = types.SimpleNamespace(target_lang="vi", notes="A drama")
    rel = Relationships(lambda: judge, lambda: settings, events.append)
    rel.heard("Linh: Wait for me!")
    assert rel.address("Minh: Hurry up.") is None
    rel.heard("Minh: Hurry up.")
    wait_for(lambda: events)
    assert "Linh: Wait for me!\nMinh: Hurry up." in judge.questions[0] and "A drama" in judge.questions[0]
    assert rel.address("Linh: I can't run that fast.") == Address("Linh", "Minh", "em", "anh")
    assert rel.address("Minh: Fine, I'll carry it.") == Address("Minh", "Linh", "anh", "em")
    rel.clear()
    assert rel.address("Minh: Fine, I'll carry it.") is None
    rel.close()


def test_a_first_line_with_a_clear_cue_gets_pronouns_before_anyone_answers():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("Linh: Brother, wait for me!") == Address("Linh", "the listener", "em", "anh")
    assert rel.address("Lamar: Damn, homie.").me == "tao"
    settings.target_lang = "en"
    assert rel.address("Lamar: Damn, homie.") is None


def test_the_pronouns_ride_with_the_line_and_stay_in_the_history():
    address = Address("Lamar", "Franklin", "tao", "mày")
    ctx = ContextWindow()
    ctx.add("Franklin: Yo.", "Franklin: Ê mày.", note=address.note())
    messages = build_messages("Lamar: You owe me.", ctx.turns(), "en", "vi", address=address)
    assert messages[1]["content"].startswith("Franklin: Yo.\n\n(Translator's note: Lamar is talking to Franklin")
    assert messages[-1]["content"].endswith('says "tao" for I/me and "mày" for you.)')
    assert messages[-1]["item"] == "Lamar: You owe me."
    [hy] = for_hy_mt(messages)
    assert "I 翻译成 tao\n" in hy["content"] and "you 翻译成 mày\n" in hy["content"]
    assert "Lamar对Franklin说话" in hy["content"] and "Translator's note" not in hy["content"]
    assert hy["content"].endswith("\nYou owe me.") and "Lamar: You owe me." not in hy["content"]


def test_speech_without_a_name_still_follows_a_clear_cue():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("Shut up, you punk!") == Address("the speaker", "the listener", "tao", "mày")
    assert rel.address("Please sit down.") is None


def test_a_cue_said_before_the_reply_counts_for_the_reply():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    rel.heard("ミカ：先輩、待ってください！")
    reply = rel.address("リュウ：なんだよ、忙しいんだ。")
    assert reply == Address("リュウ", "ミカ", "", "em")
    assert reply.note() == 'リュウ is talking to ミカ: in Vietnamese リュウ says "em" for you.'
    rel.close()


def test_titles_in_names_and_polite_sir_do_not_make_a_shop():
    assert guess_facts("Sir Aldric: The gate must hold.", "Sir Aldric", "Mira") is None
    assert guess_facts("Mira: Yes, sir, at once.", "Mira", "Aldric") is None
    assert guess_facts("Welcome! What can I get you?", "Clerk", "Kai")["senior"] == "Clerk"
    assert guess_facts("Welcome back, prince.", "Guard", "Kai")["relation"] == "ruler and subject"


def test_family_teachers_masters_and_rulers_get_their_own_pronouns():
    mom = {"relation": "parent and child", "senior": "Mẹ Hoa", "Mẹ Hoa gender": "female"}
    assert vietnamese(mom, "Mẹ Hoa", "Tuấn") == ("mẹ", "con")
    assert vietnamese(mom, "Tuấn", "Mẹ Hoa") == ("con", "mẹ")
    grandma = {"relation": "grandparent and grandchild", "older": "Bà", "Bà gender": "female"}
    assert vietnamese(grandma, "Bà", "Minh") == ("bà", "cháu")
    teacher = {"relation": "teacher and student", "senior": "Tanaka", "Tanaka gender": "male"}
    assert vietnamese(teacher, "Haruto", "Tanaka") == ("em", "thầy")
    master = {"relation": "master and disciple", "senior": "Sư phụ"}
    assert vietnamese(master, "Sư phụ", "Lâm") == ("ta", "con")
    assert vietnamese(master, "Lâm", "Sư phụ") == ("đệ tử", "sư phụ")
    shop = {"relation": "staff and customer", "senior": "Clerk", "Kai gender": "male"}
    assert vietnamese(shop, "Clerk", "Kai") == ("tôi", "quý khách")
    assert vietnamese(shop, "Kai", "Clerk") == ("tôi", "bạn")


def test_elder_titles_count_only_when_they_call_the_listener():
    assert guess_facts("Brother, wait!", "Linh", "Minh")["older"] == "Minh"
    assert guess_facts("My brother is sick today.", "Linh", "Minh") is None
    assert guess_facts("형사님, 이쪽입니다.", "A", "B") is None
    senpai = guess_facts("先輩、待って！", "Mika", "Aoi")
    assert senpai["older"] == "Aoi" and "Aoi gender" not in senpai
    assert guess_facts("お前、宿題は？", "Teacher", "Haruto") is None


def test_the_listener_is_whoever_the_line_calls_by_name():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("Haruto: Morning.", "Misaki: Morning!", "Haruto: Ready?", "Misaki: Yes.", "Tanaka: Sit down."):
        rel.heard(line)
    assert rel._listener("Misaki", "Misaki: Tanaka-sensei, sorry I am late!") == "Tanaka"
    assert rel._listener("Misaki", "Misaki: Sorry I am late!") == "Haruto"
    rel.close()


def test_titles_settle_roles_the_judge_gets_wrong():
    from screen_translator.address import settled

    wrong = {"relation": "ruler and subject", "senior": "Sir Aldric", "older": "Sir Aldric"}
    facts = settled(wrong, "Princess Elena", "Sir Aldric", [])
    assert facts["senior"] == "Princess Elena" and facts["Princess Elena gender"] == "female"
    assert vietnamese(facts, "Sir Aldric", "Princess Elena") == ("thần", "điện hạ")
    senpai = settled({"relation": "teacher and student", "senior": "ミサキ", "older": "ミサキ", "ミサキ gender": "female"},
                     "ハルト", "ミサキ", [("ハルト", "ハルト：すみません、ミサキ先輩！")])
    assert vietnamese(senpai, "ハルト", "ミサキ") == ("em", "chị")
    teacher = settled({"relation": "coworkers"}, "ミサキ", "田中先生", [("ミサキ", "ミサキ：すみません、田中先生。")])
    assert vietnamese(teacher, "ミサキ", "田中先生") == ("em", "thầy")
    assert vietnamese({"relation": "rivals or enemies", "B gender": "female"}, "A", "B") == ("tôi", "chị")


def test_a_title_gives_pronouns_before_the_judge_answers():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    rel.heard("Grandma: Did you eat?")
    assert rel.address("Minjun: Not yet, Grandma.") == Address("Minjun", "Grandma", "cháu", "bà")
    rel.close()


def test_a_title_said_to_someone_else_does_not_count_for_the_pair():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("할머니: 밥은 먹었니?", "민준: 네, 할머니.", "할머니: 그래.", "지은: 오빠!"):
        rel.heard(line)
    pair = frozenset(("지은", "민준"))
    assert all("할머니" not in line for _who, line in rel._said(pair))
    rel.close()


def test_korean_titles_mentioning_someone_else_are_not_cues():
    assert guess_facts("할머니는 맨날 오빠 편만 들어요.", "지은", "할머니") is None
    assert guess_facts("오빠, 그 말 지난달에도 했거든?", "지은", "민준")["older"] == "민준"
    assert guess_facts("고마워, 오빠!", "지은", "민준")["older"] == "민준"


def test_an_enemy_calling_someone_master_is_still_an_enemy():
    from screen_translator.address import settled

    facts = settled({"relation": "rivals or enemies", "rough": True}, "血刀老祖", "师父", [])
    assert vietnamese(facts, "血刀老祖", "师父") == ("tao", "mày")


def test_titles_count_only_when_they_call_someone_and_family_words_work_in_speech():
    from screen_translator.address import settled

    mentioned = settled({"relation": "friends"}, "Ren", "Haruto", [("Ren", "Ren: My mom says hi.")])
    assert mentioned["relation"] == "friends"
    called = settled({"relation": "friends"}, "Tuấn", "Hoa", [("Tuấn", "Tuấn: Mom, I'm home!")])
    assert called["relation"] == "parent and child" and called["senior"] == "Hoa"
    assert vietnamese(guess_facts("Mom, I'm home!", "the speaker", "the listener"), "the speaker", "the listener") \
        == ("con", "mẹ")
    assert guess_facts("엄마가 오빠 기다리다가 주무셨어", "지은", "민준") is None
    assert guess_facts("오빠왜전화안받았어", "지은", "민준")["older"] == "민준"


def test_only_a_name_that_calls_someone_picks_the_listener():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("Ren: Hey.", "Haruto: Hi.", "Ren: Ready?", "Haruto: Yes.", "Misaki: Morning."):
        rel.heard(line)
    assert rel._listener("Ren", "Ren: Misaki, wait up!") == "Misaki"
    assert rel._listener("Ren", "Ren: I saw Misaki at the station.") == "Haruto"
    rel.close()
