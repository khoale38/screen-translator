from screen_translator.context import ContextWindow
from screen_translator.translator import build_messages, build_screen_messages, clean_output, for_hy_mt, label_speaker


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
    assert clean_output("“Đi thôi!”") == "Đi thôi!"


def test_a_dropped_speaker_label_is_put_back():
    assert label_speaker("Lamar: You owe me.", "Mày nợ tao.") == "Lamar: Mày nợ tao."
    assert label_speaker("Lamar: You owe me.", "Lamar: Mày nợ tao.") == "Lamar: Mày nợ tao."
    assert label_speaker("Objective: reach the bank", "Mục tiêu: đến ngân hàng") == "Mục tiêu: đến ngân hàng"


def test_label_speaker_joins_a_name_line_onto_the_translation():
    assert label_speaker("カイ\nまさか…", "Kai\nNo way...\nyou went?") == "Kai: No way... you went?"
    assert label_speaker("カイ\nまさか…", "Kai: No way...") == "Kai: No way..."
    assert label_speaker("まさか…", "No way...\nreally?") == "No way...\nreally?"
    assert label_speaker("長い一行目のテキストがここにあります。\n二行目", "A long first sentence.\nSecond") == "A long first sentence.\nSecond"


def test_hy_mt_gets_one_short_request_with_the_earlier_lines_and_terms_above_the_text():
    ctx = ContextWindow()
    ctx.add("ユキはまだ？", "Is Yuki not back yet?")
    [request] = for_hy_mt(build_messages("行ったよ。", ctx.turns(), "ja", "en", {"ユキ": "Yuki"}, "A JRPG"))
    assert request["role"] == "user"
    assert request["content"] == (
        "参考下面的翻译：\nユキ 翻译成 Yuki\n\n背景：A JRPG\n前面的对话：\nユキはまだ？\nIs Yuki not back yet?\n"
        "屏幕上的文字：\n行ったよ。\n参考上面的信息，把下面的文本翻译成English，注意不需要翻译上文，也不要额外解释：\n行ったよ。")


def test_hy_mt_screen_requests_share_everything_but_the_item():
    screen = ["設定", "ミッション失敗"]
    a, b = (for_hy_mt(build_screen_messages(i, screen, [], "ja", "vi"))[0]["content"] for i in range(2))
    assert a.endswith("\n設定") and b.endswith("\nミッション失敗")
    assert a.split("屏幕上的文字：")[0] == b.split("屏幕上的文字：")[0]


def test_hy_mt_is_not_shown_the_rest_of_the_screen():
    screen = ["設定", "ミッション失敗", "Lv 41"]
    prompt = for_hy_mt(build_screen_messages(1, screen, [], "ja", "vi"))[0]["content"]
    assert "設定" not in prompt and "Lv 41" not in prompt and prompt.count("ミッション失敗") == 2


def test_a_models_introduction_to_its_answer_is_dropped_but_labels_are_kept():
    assert clean_output("Dưới đây là bản dịch của câu:\n\"Anh đi đâu vậy?\"") == "Anh đi đâu vậy?"
    assert clean_output("Here's the translation:\nWhere are you going?") == "Where are you going?"
    assert clean_output("Mục tiêu:\nĐến ngân hàng") == "Mục tiêu:\nĐến ngân hàng"


def test_hy_mt_gets_screen_labels_whole_and_name_boxes_without_the_name():
    screen = ["剩余时间：02:15", "设置"]
    prompt = for_hy_mt(build_screen_messages(0, screen, [], "zh", "vi"))[0]["content"]
    assert prompt.endswith("剩余时间：02:15") and "前面的对话" not in prompt
    boxed = for_hy_mt(build_messages("ミサキ\nまた遅刻？", [], "ja", "vi"))[0]["content"]
    assert boxed.endswith("\nまた遅刻？") and "ミサキ\nまた遅刻？" not in boxed


def test_a_dropped_name_comes_back_as_the_viewer_reads_it():
    names = {"ミサキ": "Misaki"}
    assert label_speaker("ミサキ：また遅刻？", "Lại muộn à?", names) == "Misaki: Lại muộn à?"
    assert label_speaker("ミサキ\nまた遅刻？", "Lại muộn à?", names) == "Misaki: Lại muộn à?"
    assert label_speaker("ミサキ\nまた遅刻？", "Misaki\nLại muộn à?", names) == "Misaki: Lại muộn à?"
