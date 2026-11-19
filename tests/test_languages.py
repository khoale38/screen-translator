from screen_translator.languages import already_in


def test_text_already_in_the_target_language_is_recognised():
    assert already_in("vi", "@bach7934 cất súng luôn kia a")
    assert already_in("ja", "まさか、あの森に行ったんじゃ…")
    assert already_in("zh", "你今天去哪里了")
    assert already_in("ko", "어디 가세요?")
    assert already_in("ru", "Где ты был?")


def test_foreign_text_is_not_mistaken_for_the_target():
    assert not already_in("vi", "Hold L1 to open your loadout")
    assert not already_in("vi", "Ça va très bien, merci")
    assert not already_in("zh", "まさか、あの森に行ったんじゃ…")
    assert not already_in("ja", "你今天去哪里了")
    assert not already_in("en", "Anything at all")
