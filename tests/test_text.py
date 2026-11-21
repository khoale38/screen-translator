from screen_translator.ocr import OcrLine
from screen_translator.text import Stabilizer, assemble


def line(text, x, y, w=0.3, h=0.05, conf=0.9):
    return OcrLine(text, conf, (x, y, w, h))


def test_assemble_orders_rows_and_joins_cjk_without_spaces():
    lines = [line("聞かなかった。", 0.1, 0.46), line("止めたけど", 0.1, 0.40)]
    assert assemble(lines) == "止めたけど聞かなかった。"


def test_assemble_joins_latin_with_spaces_and_sorts_same_row_by_x():
    lines = [line("world", 0.5, 0.401), line("Hello", 0.1, 0.40), line("again", 0.1, 0.46)]
    assert assemble(lines) == "Hello world again"


def test_assemble_drops_low_confidence_noise():
    assert assemble([line("Hi there", 0.1, 0.1), line("~#", 0.1, 0.3, conf=0.1)]) == "Hi there"


def test_stabilizer_waits_for_typewriter_text_to_settle():
    s = Stabilizer(stable_ticks=2)
    events = [s.feed(t) for t in ["こん", "こんにち", "こんにちは", "こんにちは", "こんにちは"]]
    emitted = [e for e in events if e]
    assert len(emitted) == 1
    assert emitted[0].text == "こんにちは"


def test_stabilizer_accepts_a_subtitle_that_reads_slightly_differently_over_a_moving_scene():
    s = Stabilizer(stable_ticks=2)
    reads = ["Get in the car before the police arrive.", "Get in the car before the po1ice arrive. l",
             "Get in the car before the police arrive!"]
    emitted = [e for e in (s.feed(t) for t in reads) if e]
    assert len(emitted) == 1 and emitted[0].text.startswith("Get in the car")


def test_stabilizer_moves_on_to_a_new_subtitle():
    s = Stabilizer(stable_ticks=2)
    events = [s.feed(t) for t in ["Lose the cops.", "Lose the cops.", "Go to the safehouse.", "Go to the safehouse."]]
    assert [e.text for e in events if e] == ["Lose the cops.", "Go to the safehouse."]


def test_stabilizer_does_not_repeat_and_clears_when_text_disappears():
    s = Stabilizer(stable_ticks=2)
    assert [s.feed("Hello there") for _ in range(4)].count(None) == 3
    assert s.feed("") is None
    cleared = s.feed("")
    assert cleared and cleared.kind == "clear"


def test_stabilizer_flags_text_that_extends_the_previous_line():
    s = Stabilizer(stable_ticks=1)
    first = s.feed("The door was locked,")
    second = s.feed("The door was locked, so we climbed through the window.")
    assert first and not first.extends_previous
    assert second and second.extends_previous


def test_assemble_keeps_a_speaker_label_on_its_own_line():
    lines = [line("カイ", 0.05, 0.10, h=0.04), line("ユキはまだ帰ってこないの？", 0.05, 0.30, h=0.06)]
    assert assemble(lines) == "カイ\nユキはまだ帰ってこないの？"


def test_assemble_joins_tightly_wrapped_lines_of_one_sentence():
    lines = [line("The door was", 0.1, 0.40, h=0.06), line("locked.", 0.1, 0.48, h=0.06)]
    assert assemble(lines) == "The door was locked."


def test_assemble_splits_a_name_label_even_when_it_sits_close_to_the_dialogue():
    lines = [line("アオイ", 0.033, 0.133, w=0.111, h=0.24), line("彼女、昨日から連絡がないんだ。", 0.031, 0.533, w=0.553, h=0.28)]
    assert assemble(lines) == "アオイ\n彼女、昨日から連絡がないんだ。"


def test_hud_symbols_and_lone_characters_are_not_text():
    lines = [line("• . • •", 0.1, 0.30), line("Hold L1 to open your loadout", 0.1, 0.36),
             line("$", 0.6, 0.10), line("5,46.43", 0.6, 0.20), line("N", 0.8, 0.60)]
    assert assemble(lines) == "Hold L1 to open your loadout"


def test_bullets_are_trimmed_but_dialogue_ellipses_stay():
    assert assemble([line("• Press R1 to access your items", 0.1, 0.4)]) == "Press R1 to access your items"
    assert assemble([line("…行ったよ。", 0.1, 0.4)]) == "…行ったよ。"


def test_controller_button_misreads_become_the_button_name():
    assert assemble([line("Hold LI to open your loadout", 0.1, 0.4)]) == "Hold L1 to open your loadout"
    assert assemble([line("Press R| to access your items", 0.1, 0.4)]) == "Press R1 to access your items"
    assert assemble([line("Hold Lt to open your loadout", 0.1, 0.4)]) == "Hold L1 to open your loadout"
    assert assemble([line("Hold L! to open your loadout", 0.1, 0.4)]) == "Hold L1 to open your loadout"
    assert assemble([line("Lt. Dan was right", 0.1, 0.4)]) == "Lt. Dan was right"
    assert assemble([line("LIKE AND SUBSCRIBE", 0.1, 0.4)]) == "LIKE AND SUBSCRIBE"


def test_a_taller_ocr_box_on_the_first_row_does_not_split_the_paragraph():
    rows = [OcrLine("Use the adrenaline from your", 1.0, (0.024, 0.276, 0.19, 0.033)),
            OcrLine("loadout to overpower your", 1.0, (0.022, 0.313, 0.2, 0.020)),
            OcrLine("for a short time.", 1.0, (0.046, 0.344, 0.11, 0.023))]
    assert assemble(rows) == "Use the adrenaline from your loadout to overpower your for a short time."


def test_ocr_prefers_a_runner_up_guess_that_keeps_the_button():
    from screen_translator.ocr import prefer_button_reading

    guesses = ["Hold L to open your loadout", "Hold LI to open your loadout", "Hold Lt to open your loadout"]
    assert prefer_button_reading(guesses) == "Hold LI to open your loadout"
    assert prefer_button_reading(["Hold to open your loadout", "Hold LI to open your loadout"]) == \
        "Hold LI to open your loadout"
    assert prefer_button_reading(["Hold on, Lisa", "Hold on, LI sa"]) == "Hold on, Lisa"
    assert prefer_button_reading(["Press R1 to access", "Press RI to access"]) == "Press R1 to access"
