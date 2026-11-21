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
    assert vietnamese(merged(judged, cues), "Mika", "Ryu") == ("em", "anh")
    assert vietnamese(merged({**judged, "older": "Ryu"}, cues), "Ryu", "Mika") == ("anh", "em")


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
    assert hy["content"].endswith("\nLamar: You owe me.")


def test_speech_without_a_name_still_follows_a_clear_cue():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    assert rel.address("Shut up, you punk!") == Address("the speaker", "the listener", "tao", "mày")
    assert rel.address("Please sit down.") is None


def test_a_cue_said_before_the_reply_counts_for_the_reply():
    settings = types.SimpleNamespace(target_lang="vi", notes="")
    rel = Relationships(lambda: Judge("{}"), lambda: settings, lambda e: None)
    rel.heard("ミカ：先輩、待ってください！")
    assert rel.address("リュウ：なんだよ、忙しいんだ。") == Address("リュウ", "ミカ", "anh", "em")
    rel.close()
