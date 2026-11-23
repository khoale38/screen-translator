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
    assert vietnamese(teacher, "ミサキ", "田中先生") == ("em", "")
    assert vietnamese({**teacher, "田中先生 gender": "male"}, "ミサキ", "田中先生") == ("em", "thầy")
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


def test_possessives_quotes_and_reported_speech_are_mentions():
    from screen_translator.address import calls_by, ROLES

    parent = ROLES["parent and child"]
    assert not calls_by(parent, "Listen to your mother.")
    assert not calls_by(parent, "Mom said you can't go.")
    assert calls_by(parent, "Mom, I'm home!")
    assert calls_by(parent, "I'm home, Mom.")
    assert guess_facts("弟から届いた手紙にはお姉ちゃん。", "the speaker", "the listener") is None


def test_a_guess_from_one_line_is_marked_and_titles_set_gender():
    from screen_translator.address import settled

    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("Shut up, you punk!").guessed
    facts = settled({"relation": "friends"}, "Tuấn", "Hoa", [("Tuấn", "Tuấn: Mom, I'm home!")])
    assert vietnamese(facts, "Tuấn", "Hoa") == ("con", "mẹ")
    rel.close()


def test_a_titled_name_calls_that_person_and_a_newcomer_answers_the_last_speaker():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("Kai: Hi.", "Mia: Hey.", "Kai: Ready?", "Mia: Yes.", "Joe: Morning."):
        rel.heard(line)
    assert rel._listener("Kai", "Kai: Uncle Joe!") == "Joe"
    assert rel._listener("Ana", "Ana: Nice to meet you.") == "Joe"
    rel.close()


def test_your_majesty_little_sister_and_a_princess_in_speech():
    king = guess_facts("Thank you, Your Majesty. I live to serve.", "the speaker", "the listener")
    assert vietnamese(king, "the speaker", "the listener") == ("thần", "bệ hạ")
    princess = guess_facts("Forgive me, Your Highness.", "the speaker", "the listener")
    assert vietnamese(princess, "the speaker", "the listener") == ("thần", "điện hạ")
    sibling = guess_facts("Little sister, you forgot your lunch!", "Max", "Lily")
    assert sibling["older"] == "Max"
    assert guess_facts("我告诉师兄了", "the speaker", "the listener") is None


def test_one_rough_line_does_not_make_rivals_and_a_newcomer_calling_a_title_has_no_listener():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("Kai: Hey.", "Mia: Hi.", "Kai: You fool.", "Mia: Ready?"):
        rel.heard(line)
    assert "rough" not in rel._cues.get(frozenset(("Kai", "Mia")), {})
    assert rel._listener("Ana", "Ana: Grandpa! You're back!") is None
    rel.close()


def test_film_subtitle_first_lines_and_bracketed_names():
    assert speaker_of("Okay Mom\nI'll call you when I land.") is None
    assert speaker_of("Then Jake\nshowed up at my door.") is None
    assert speaker_of("Captain Reyes\nHold the line!") == "Captain Reyes"
    assert speaker_of("（ハルカ）お兄ちゃん　起きて！") == "ハルカ"
    assert speaker_of("(MOM) Sam, wake your brother up.") == "MOM"


def test_a_parent_speaking_of_themself_and_titles_after_verbs_are_not_calls():
    assert guess_facts("お母さんはね、ずっと心配してたのよ。", "the speaker", "the listener") is None
    assert guess_facts("爸爸给你买了礼物，快打开看看。", "the speaker", "the listener") is None
    assert guess_facts("ごめんね、お母さん、すぐ行くから。", "the speaker", "the listener")["senior"] == "the listener"
    assert guess_facts("三弟，护住师父，我来对付他！", "萧远", "林风")["older"] == "萧远"
    assert guess_facts("I'll go with Father tomorrow.", "the speaker", "the listener") is None
    assert guess_facts("Thanks, Dad.", "the speaker", "the listener")["senior"] == "the listener"


def test_korean_titles_find_their_speaker():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    for line in ("서연: 오빠!", "준호: 왜?", "최 대리: 준호 씨, 팀장님이 찾으세요."):
        rel.heard(line)
    assert rel._listener("최 대리", "최 대리: 준호 씨, 팀장님이 찾으세요.") == "준호"
    assert rel._listener("준호", "준호: 네, 대리님. 바로 갈게요.") == "최 대리"
    rel.close()


def test_closing_calls_without_a_comma_group_lines_and_a_first_line_that_only_its_replier_takes():
    assert guess_facts("Thanks Dad.", "the speaker", "the listener")["senior"] == "the listener"
    assert guess_facts("谢谢妈妈。", "the speaker", "the listener")["senior"] == "the listener"
    assert guess_facts("어머니 말씀이 맞아요.", "the speaker", "the listener") is None
    assert speaker_of("【アオイ】ねえ、タクミ！") == "アオイ"
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("Mina: Hi everyone, welcome back!") is None
    for line in ("Sir Aldric: Your Highness, stay behind me.", "Princess Elena: I can walk.", "Lyra: She's right."):
        rel.heard(line)
    rel._note_cues(frozenset(("Lyra", "Sir Aldric")))
    assert "royal" not in rel._cues.get(frozenset(("Lyra", "Sir Aldric")), {})
    rel.close()


def test_naming_a_speaker_never_asks_the_translator():
    from screen_translator.config import SettingsStore
    from screen_translator.pipeline import Pipeline

    class NoCalls:
        def translate(self, *args, **kwargs):
            raise AssertionError("the translator was called for a name")

    import tempfile, pathlib
    store = SettingsStore(pathlib.Path(tempfile.mkdtemp()) / "settings.json")
    store.update({"target_lang": "vi"})
    pipeline = Pipeline(store, lambda e: None)
    assert pipeline._labelled("村長\nよく来たな。", "Con đến rồi à.", NoCalls()) == "Trưởng làng: Con đến rồi à."
    assert pipeline._labelled("萧寒\n走吧。", "Đi thôi.", NoCalls()) == "Đi thôi."


def test_round_7_cues_parent_gender_korean_calls_nobles_and_groups():
    assert vietnamese(guess_facts("妈，我回来了！", "the speaker", "the listener"), "the speaker", "the listener") == ("con", "mẹ")
    assert guess_facts("엄마 나 오늘 늦어.", "the speaker", "the listener")["senior"] == "the listener"
    assert guess_facts("어머니병원에갔었어.", "the speaker", "the listener") is None
    assert guess_facts("Young Master, the carriage is ready.", "Maid", "Kai")["relation"] == "noble and servant"
    lady = guess_facts("My lady, the road is clear.", "Knight", "Elena")
    assert vietnamese(lady, "Knight", "Elena") == ("tôi", "tiểu thư")
    assert guess_facts("お姉ちゃんが守ってあげるからね。", "the speaker", "the listener") is None
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("You guys are the best. Mom, can I go?") is not None
    assert rel.address("- Don't lie to me, Tom.\n- I'm not lying, Mom!") is None
    rel.close()
