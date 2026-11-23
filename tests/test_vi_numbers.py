from screen_translator.natural_voices import prepare
from screen_translator.vi_numbers import spoken_integer, spoken_numbers


def test_vietnamese_numbers_are_read_the_way_people_say_them():
    assert [spoken_integer(n) for n in (15, 21, 24, 105, 1005, 2024)] == [
        "mười lăm", "hai mươi mốt", "hai mươi tư", "một trăm linh năm", "một nghìn không trăm linh năm",
        "hai nghìn không trăm hai mươi tư"]
    assert spoken_numbers("Pin còn 15% lúc 7:30, giá $999.") == \
        "Pin còn mười lăm phần trăm lúc bảy giờ ba mươi, giá chín trăm chín mươi chín đô."
    assert spoken_numbers("Pin 5000mAh, nặng 1.200 gram, 3,5 inch.") == \
        "Pin năm nghìn mAh, nặng một nghìn hai trăm gram, ba phẩy năm inch."


def test_only_vietnamese_dubs_spell_numbers_out():
    import unicodedata

    assert "mười lăm" in unicodedata.normalize("NFC", prepare("Còn 15%.", "vi"))
    assert "15" in prepare("15% left.", "en")
