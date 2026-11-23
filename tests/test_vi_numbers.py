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



def test_versions_dates_units_and_codes_are_read_without_failing():
    cases = {
        "iOS 18.2.1.": "iOS mười tám chấm hai chấm một.",
        "Đếm ngược 3,2,1!": "Đếm ngược ba, hai, một!",
        "Ngày 05.03.1998.": "Ngày năm tháng ba năm một nghìn chín trăm chín mươi tám.",
        "Hẹn ngày 20/11 nhé.": "Hẹn ngày hai mươi tháng mười một nhé.",
        "Cho 200g bột và 1/2 cốc sữa.": "Cho hai trăm gam bột và nửa cốc sữa.",
        "Lúc 20h30.": "Lúc hai mươi giờ ba mươi.",
        "Nướng ở 180°C.": "Nướng ở một trăm tám mươi độ C.",
        "Combo x2!": "Combo nhân hai!",
        "Chỉ 50k, hay 50.000đ.": "Chỉ năm mươi nghìn, hay năm mươi nghìn đồng.",
        "Số 0912 345 678.": "Số không chín một hai ba bốn năm sáu bảy tám.",
        "Điệp viên 007.": "Điệp viên không không bảy.",
        "Trời -5 độ.": "Trời âm năm độ.",
        "Hạ 3/5 kẻ địch.": "Hạ ba trên năm kẻ địch.",
    }
    assert {text: spoken_numbers(text) for text in cases} == cases


def test_only_vietnamese_dubs_spell_numbers_out():
    import unicodedata

    assert "mười lăm" in unicodedata.normalize("NFC", prepare("Còn 15%.", "vi"))
    assert "15" in prepare("15% left.", "en")
