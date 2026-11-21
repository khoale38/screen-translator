from screen_translator.context import ContextWindow
from screen_translator.translator import build_messages, clean_output, label_speaker


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_context_keeps_recent_lines_and_forgets_after_ttl():
    clock = Clock()
    ctx = ContextWindow(max_turns=4, ttl_s=60, clock=clock)
    for i in range(4):
        ctx.add(f"s{i}", f"t{i}")
        clock.now += 10
    assert [t.source for t in ctx.turns()] == ["s0", "s1", "s2", "s3"]
    ctx.add("s4", "t4")
    clock.now += 10
    assert [t.source for t in ctx.turns()] == ["s3", "s4"]
    clock.now += 45
    assert [t.source for t in ctx.turns()] == ["s4"]
    clock.now += 100
    assert ctx.turns() == []


def test_context_replace_last_swaps_an_extended_line():
    ctx = ContextWindow()
    ctx.add("The door was locked,", "a")
    ctx.add("The door was locked, so we left.", "b", replace_last=True)
    assert [t.source for t in ctx.turns()] == ["The door was locked, so we left."]


def test_build_messages_puts_history_as_prior_turns_with_glossary_and_notes():
    ctx = ContextWindow()
    ctx.add("ユキはまだ？", "Is Yuki not back yet?")
    messages = build_messages("行ったよ。", ctx.turns(), "ja", "en", {"ユキ": "Yuki"}, "Yuki is a girl.")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Japanese" in messages[0]["content"] and "English" in messages[0]["content"]
    assert "ユキ → Yuki" in messages[0]["content"]
    assert "Yuki is a girl." in messages[0]["content"]
    assert messages[-1]["content"] == "行ったよ。"


def test_build_messages_handles_auto_source():
    messages = build_messages("Bonjour", [], "auto", "vi")
    assert "whatever language" in messages[0]["content"] and "Vietnamese" in messages[0]["content"]


def test_clean_output_strips_thinking_and_wrapping_quotes():
    assert clean_output("<think>\nhmm\n</think>\n\n\"She went.\"") == "She went."
    assert clean_output("<think>still thinking") == ""
    assert clean_output("「行こう」") == "「行こう」"


def test_label_speaker_joins_a_name_line_onto_the_translation():
    assert label_speaker("カイ\nまさか…", "Kai\nNo way...\nyou went?") == "Kai: No way... you went?"
    assert label_speaker("カイ\nまさか…", "Kai: No way...") == "Kai: No way..."
    assert label_speaker("まさか…", "No way...\nreally?") == "No way...\nreally?"
    assert label_speaker("長い一行目のテキストがここにあります。\n二行目", "A long first sentence.\nSecond") == "A long first sentence.\nSecond"
