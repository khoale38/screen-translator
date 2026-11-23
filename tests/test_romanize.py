from screen_translator.romanize import romanized


def test_kana_and_hangul_names_are_spelled_the_usual_way():
    assert romanized("ミサキ") == "Misaki"
    assert romanized("リュウ") == "Ryu"
    assert romanized("しょうた") == "Shota"
    assert romanized("ミッキー") == "Mikki"
    assert romanized("김민수") == "Kim Minsu"
    assert romanized("지은") == "Jieun"
    assert romanized("田中") is None and romanized("Ken") is None


def test_role_words_are_titles_not_names():
    from screen_translator.romanize import role_label

    assert romanized("할머니") is None and role_label("할머니") == "Bà"
    assert role_label("강 부장") == "Trưởng phòng Kang"
    assert romanized("민수") == "Minsu"
