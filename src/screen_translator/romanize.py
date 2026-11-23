"""Character names in Latin letters, by fixed rules: kana in Hepburn, hangul in Revised Romanization.

A small model asked for a name in Latin letters often returns it unchanged, or translates its meaning.
"""

from __future__ import annotations

import re

_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o", "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "た": "ta", "ち": "chi", "つ": "tsu", "て": "te",
    "と": "to", "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no", "は": "ha", "ひ": "hi", "ふ": "fu",
    "へ": "he", "ほ": "ho", "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo", "や": "ya", "ゆ": "yu",
    "よ": "yo", "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro", "わ": "wa", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go", "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze",
    "ぞ": "zo", "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do", "ば": "ba", "び": "bi", "ぶ": "bu",
    "べ": "be", "ぼ": "bo", "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
}
_DIGRAPHS = {"ゃ": "a", "ゅ": "u", "ょ": "o"}

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_VOWELS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu",
           "eu", "ui", "i"]
_FINALS = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t",
           "ng", "t", "t", "k", "t", "p", "t"]
_SURNAMES = {"김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho", "윤": "Yoon",
             "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin", "권": "Kwon", "황": "Hwang",
             "안": "Ahn", "송": "Song", "유": "Yoo", "홍": "Hong"}

KANA_NAME = re.compile("[ぁ-ヿー・]+")
HANGUL_NAME = re.compile("[가-힣 ]+")
NOT_LATIN = re.compile("[぀-ヿ㐀-鿿가-힯]")


def _kana(name: str) -> str:
    text = "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in name.replace("・", " "))
    sounds: list[str] = []
    double = False
    for ch in text:
        if ch == "っ":
            double = True
            continue
        if ch == "ー":
            continue
        if ch in "ぁぃぅぇぉ" and sounds and sounds[-1][:1].isalpha() and (len(sounds[-1]) > 1 or sounds[-1] == "u"):
            # A small vowel changes the sound before it: ティ is ti, ファ is fa, ヴィ is vi, シェ is she.
            last = sounds.pop()
            stem = {"te": "t", "de": "d", "fu": "f", "vu": "v", "tsu": "ts", "u": "w", "to": "t", "do": "d"}.get(
                last, last[:-1] if last.endswith(("i", "u")) else last)
            sounds.append(stem + _KANA[ch])
            continue
        if ch in _DIGRAPHS and sounds and sounds[-1].endswith("i"):
            last = sounds.pop()
            sounds.append(last[:-1] + _DIGRAPHS[ch] if last in ("shi", "chi", "ji") else last[:-1] + "y" + _DIGRAPHS[ch])
            continue
        sound = _KANA.get(ch, ch)
        if double and sound[:1].isalpha():
            sound = ("t" if sound.startswith("ch") else sound[0]) + sound
        double = False
        sounds.append(sound)
    # Names drop the long vowel in everyday spelling: Ryu, Shota, Kyoko.
    spelled = "".join(sounds).replace("ou", "o").replace("uu", "u")
    return " ".join(word.capitalize() for word in spelled.split())


def _syllables(word: str) -> str:
    out = ""
    for ch in word:
        code = ord(ch) - 0xAC00
        if not 0 <= code < 11172:
            out += ch
            continue
        out += _INITIALS[code // 588] + _VOWELS[code % 588 // 28] + _FINALS[code % 28]
    return out


def _hangul(name: str) -> str:
    words = name.split()
    if len(words) == 1 and len(name) == 3 and name[0] in _SURNAMES:
        return f"{_SURNAMES[name[0]]} {_syllables(name[1:]).capitalize()}"
    if len(words) == 2 and words[0] in _SURNAMES:
        return f"{_SURNAMES[words[0]]} {_syllables(words[1]).capitalize()}"
    return " ".join(_syllables(word).capitalize() for word in words)


# Answers a game offers ("はい / いいえ"): never a speaker's name tab, and kept apart even with no question above.
CHOICE_WORDS = {"はい", "いいえ", "Yes", "No", "YES", "NO", "OK", "Cancel", "是", "否", "예", "아니요", "아니오", "接受", "拒绝",
                "确定", "取消", "買う", "売る", "やめる", "Buy", "Sell", "Back", "买", "卖", "구매", "판매", "취소", "확인"}

# Kinship and job titles used as labels: they are said in Vietnamese, not spelled out.
ROLES_VI = {
    "할머니": "Bà", "할아버지": "Ông", "엄마": "Mẹ", "어머니": "Mẹ", "아빠": "Bố", "아버지": "Bố", "선생님": "Thầy",
    "오빠": "Anh", "형": "Anh", "언니": "Chị", "누나": "Chị", "팀장": "Trưởng nhóm", "팀장님": "Trưởng nhóm",
    "부장": "Trưởng phòng", "부장님": "Trưởng phòng", "과장": "Trưởng ban", "과장님": "Trưởng ban", "대리": "Phó phòng",
    "대리님": "Phó phòng", "사장": "Giám đốc", "사장님": "Giám đốc",
    "おばあちゃん": "Bà", "おじいちゃん": "Ông", "お母さん": "Mẹ", "おかあさん": "Mẹ", "母": "Mẹ", "お父さん": "Bố",
    "おとうさん": "Bố", "父": "Bố", "先生": "Thầy", "せんせい": "Thầy", "お兄ちゃん": "Anh", "おにいちゃん": "Anh",
    "お姉ちゃん": "Chị", "おねえちゃん": "Chị", "师父": "Sư phụ", "师傅": "Sư phụ", "妈妈": "Mẹ", "爸爸": "Bố",
    "奶奶": "Bà", "爷爷": "Ông", "老师": "Thầy",
    "村長": "Trưởng làng", "村人": "Dân làng", "兵士": "Binh sĩ", "騎士": "Hiệp sĩ", "店主": "Chủ tiệm", "メイド": "Hầu gái",
    "システム": "Hệ thống", "老板娘": "Bà chủ", "店小二": "Tiểu nhị", "掌门": "Chưởng môn", "村长": "Trưởng làng",
    "상인": "Thương nhân", "경비병": "Lính gác", "대장장이": "Thợ rèn", "점원": "Nhân viên", "기사": "Hiệp sĩ",
    "시스템": "Hệ thống",
    "太后": "Thái hậu", "皇上": "Hoàng thượng", "皇帝": "Hoàng đế", "王爷": "Vương gia", "宫女": "Cung nữ",
    "服务员": "Phục vụ", "师姐": "Sư tỷ", "师兄": "Sư huynh", "师妹": "Sư muội", "师弟": "Sư đệ", "師姐": "Sư tỷ",
    "師兄": "Sư huynh", "丫鬟": "Nha hoàn", "侍女": "Thị nữ", "公主": "Công chúa", "皇后": "Hoàng hậu", "将军": "Tướng quân",
    "伯父": "Bác", "叔父": "Chú", "岳父": "Bố vợ", "岳母": "Mẹ vợ", "师母": "Sư mẫu", "姑母": "Cô", "舅舅": "Cậu", "姨妈": "Dì",
    "伯母": "Bác gái", "婶婶": "Thím", "陛下": "Bệ hạ", "丞相": "Thừa tướng", "太监": "Thái giám", "殿下": "Điện hạ",
    "娘娘": "Nương nương", "우편함": "Hộp thư", "ログインボーナス": "Thưởng đăng nhập", "お知らせ": "Thông báo", "公告": "Thông báo",
    "알림": "Thông báo", "공지": "Thông báo", "안내": "Hướng dẫn", "경고": "Cảnh báo", "보상": "Phần thưởng",
    "퀘스트": "Nhiệm vụ", "이벤트": "Sự kiện", "우편": "Thư", "파티 초대": "Mời nhóm", "친구 요청": "Lời mời kết bạn",
    "길드 초대": "Mời bang hội", "執事": "Quản gia", "鍛冶屋": "Thợ rèn", "謎の男": "Người đàn ông bí ẩn",
    "謎の少女": "Cô gái bí ẩn", "神秘人": "Người bí ẩn", "铁匠": "Thợ rèn", "老板": "Ông chủ", "商人": "Thương nhân",
    "士兵": "Binh sĩ", "守卫": "Lính gác", "侍卫": "Thị vệ", "管家": "Quản gia", "医生": "Bác sĩ", "护士": "Y tá",
    "警察": "Cảnh sát", "店员": "Nhân viên", "村民": "Dân làng", "旁白": "Người dẫn chuyện", "ナレーション": "Người dẫn chuyện",
    "少女": "Thiếu nữ", "少年": "Thiếu niên", "老人": "Ông lão", "药铺掌柜": "Chủ tiệm thuốc", "의사": "Bác sĩ", "경찰": "Cảnh sát", "간호사": "Y tá", "선생": "Thầy", "학생": "Học sinh", "손님": "Khách",
    "직원": "Nhân viên", "주인": "Chủ", "기자": "Phóng viên", "형사": "Thanh tra", "군인": "Quân nhân", "병사": "Binh sĩ",
    "마법사": "Pháp sư", "용사": "Dũng sĩ", "마왕": "Ma vương", "왕자": "Hoàng tử", "공주": "Công chúa", "촌장": "Trưởng làng",
    "노인": "Ông lão", "할멈": "Bà lão", "아이": "Đứa trẻ", "소녀": "Cô gái", "소년": "Cậu bé", "남자": "Người đàn ông",
    "여자": "Người phụ nữ", "전하": "Điện hạ", "중전": "Trung điện", "중전마마": "Trung điện nương nương", "대비": "Đại phi", "세자": "Thế tử",
    "상궁": "Thượng cung", "내관": "Nội quan", "왕": "Vua", "姫": "Công chúa", "兄": "Anh", "爹": "Bố",
    "Mom": "Mẹ", "Mum": "Mẹ", "Mother": "Mẹ", "Dad": "Bố", "Father": "Bố", "Grandma": "Bà", "Grandpa": "Ông",
    "大人": "Đại nhân", "掌柜": "Chưởng quỹ", "长老": "Trưởng lão", "宗主": "Tông chủ",
}


# Hán-Việt readings of common characters in Chinese names: 林风 is Lâm Phong, as Vietnamese viewers know them.
_HAN_VIET = """
血Huyết 默Mặc 魔Ma 尊Tôn 邪Tà 鬼Quỷ 妖Yêu 圣聖Thánh 神Thần 仙Tiên 佛Phật 帝Đế 皇Hoàng 后Hậu 宗Tông 门門Môn 派Phái 阁閣Các
楼樓Lâu 殿Điện 宫宮Cung 堂Đường 院Viện 山Sơn 峰Phong 谷Cốc 洞Động 海Hải 江Giang 河Hà 湖Hồ 城Thành 国國Quốc 州Châu
啸嘯Khiếu 傲Ngạo 狂Cuồng 绝絕Tuyệt 无無Vô 天Thiên 地Địa 玄Huyền 黄黃Hoàng 宇Vũ 宙Trụ 洪Hồng 荒Hoang 日Nhật 星Tinh 辰Thần
晨Thần 曦Hy 昊Hạo 苍蒼Thương 穹Khung 霸Bá 战戰Chiến 刀Đao 枪槍Thương 弓Cung 箭Tiễn 盾Thuẫn 甲Giáp 锋鋒Phong 刃Nhận
无無Vô 儿兒Nhi 妃Phi 妾Thiếp 忌Kỵ 逍Tiêu 凯凱Khải 瑟Sắt 婉Uyển 灵Linh 容Dung 莫Mạc 霸Bá 武Vũ 侠俠Hiệp 剑劍Kiếm
宝Bảo 钗釵Thoa 黛Đại 玉Ngọc 湘Tương 云Vân 熙Hy 嬛Hoàn 甄Chân 华華Hoa 皇Hoàng 贵貴Quý 淑Thục 德Đức 贤賢Hiền 惠Huệ
静Tĩnh 怜憐Liên 香Hương 兮Hề 曼Mạn 菁Tinh 蕊Nhụy 瑛Anh 璎瓔Anh 珞Lạc 弘Hoằng 历歷Lịch 胤Dận 禛Chân 祯禎Trinh 康Khang
乾Càn 隆Long 雍Ung 正Chính 顺順Thuận 治Trị 魏Ngụy 尔爾Nhĩ 杰Kiệt 峻Tuấn 旻Mân 晗Hàm 昭Chiêu 炎Viêm 焱Diễm 烈Liệt
猛Mãnh 豹Báo 狼Lang 鹰鷹Ưng 凰Hoàng 麟Lân 雀Tước 蛇Xà 魂Hồn 魄Phách 霆Đình 震Chấn 岳Nhạc 嵩Tung 昆Côn 仑崙Lôn
王Vương 李Lý 张張Trương 刘劉Lưu 陈陳尘塵Trần 杨楊阳陽Dương 黄黃皇Hoàng 赵趙Triệu 吴吳Ngô 周朱舟Chu 徐Từ 孙孫尊Tôn 马馬Mã 胡狐湖Hồ 郭Quách
何荷霞河Hà 高Cao 林琳霖Lâm 罗羅La 郑鄭Trịnh 梁Lương 谢謝Tạ 宋Tống 唐Đường 许許Hứa 韩韓寒Hàn 冯馮Phùng 邓鄧Đặng 曹Tào 彭Bành 曾Tăng
肖萧蕭霄Tiêu 田Điền 董Đổng 袁Viên 潘Phan 于Vu 蒋蔣Tưởng 蔡泰太Thái 余Dư 杜Đỗ 叶葉Diệp 程Trình 苏蘇Tô 魏Ngụy 吕呂Lã 丁Đinh 任Nhậm 沈Thẩm
姚Diêu 卢盧Lư 姜Khương 崔Thôi 钟鍾Chung 谭譚Đàm 陆陸六Lục 汪Uông 范Phạm 金Kim 石Thạch 廖Liêu 贾賈Giả 夏贺賀Hạ 韦韋薇Vi 傅Phó 方芳Phương
白Bạch 邹鄒Trâu 孟Mạnh 熊Hùng 秦Tần 邱丘Khâu 江Giang 尹Doãn 薛Tiết 段Đoàn 雷Lôi 侯Hầu 龙龍Long 史Sử 陶涛濤桃Đào 黎Lê 顾顧Cố 毛Mao
龚龔宫宮Cung 邵Thiệu 万萬Vạn 钱錢Tiền 严嚴Nghiêm 武宇雨Vũ 戴Đái 莫Mạc 孔Khổng 向Hướng 汤湯Thang 慕Mộ 容蓉Dung 上Thượng 官Quan 欧歐Âu
司思Tư 令Lệnh 楚Sở 燕Yến 凌Lăng 柳Liễu 云雲芸Vân 风風峰枫楓Phong 花华華Hoa 月Nguyệt 雪Tuyết 冷Lãnh 君军軍Quân 墨Mặc 宁寧Ninh 安An 洛乐樂Lạc
秋Thu 春Xuân 冬东東Đông 南男Nam 北Bắc 西Tây 明铭銘Minh 伟偉Vĩ 强強Cường 平Bình 杰傑Kiệt 辉輝Huy 刚剛Cương 勇Dũng 毅Nghị 俊Tuấn 磊Lỗi
超Siêu 浩昊Hạo 轩軒Hiên 晨辰神宸Thần 天Thiên 子紫Tử 文Văn 志Chí 国國Quốc 建Kiến 成诚誠Thành 海Hải 山Sơn 森Sâm 木沐Mộc 水Thủy 火Hỏa 玉Ngọc
珠Châu 宝寶Bảo 美Mỹ 丽麗Lệ 娟Quyên 秀Tú 英樱櫻Anh 敏Mẫn 静靜Tĩnh 婷Đình 雅Nhã 晓曉Hiểu 小Tiểu 大Đại 红紅鸿鴻Hồng 兰蘭Lan 梅Mai 菊Cúc
莲蓮Liên 凤鳳Phượng 虎Hổ 鹤鶴Hạc 清青Thanh 蓝藍岚嵐Lam 星Tinh 璃离離Ly 夜Dạ 影Ảnh 光Quang 灵靈玲铃鈴Linh 心Tâm 念Niệm 爱愛Ái 情晴Tình 诗詩Thi
书書Thư 琴Cầm 冰Băng 霜Sương 露Lộ 烟煙嫣Yên 若Nhược 如Như 婉Uyển 柔Nhu 瑶瑤遥遙Dao 琪Kỳ 欣Hân 怡Di 佳Giai 慧Tuệ 智Trí 仁人Nhân 义義Nghĩa
信Tín 德Đức 忠中Trung 孝Hiếu 正Chính 元Nguyên 长長Trường 永Vĩnh 福Phúc 康Khang 荣榮Vinh 贵貴Quý 富Phú 凡帆Phàm 逸Dật 飞飛菲Phi
鹏鵬Bằng 远遠Viễn 言Ngôn 歌哥Ca 剑劍Kiếm 松Tùng 竹Trúc 杏Hạnh 熙曦Hy 瑾Cẩn 瑜Du 琼瓊Quỳnh 妍Nghiên 娜Na 倩Thiến 萱Huyên 涵Hàm
悦悅Duyệt 一Nhất 二Nhị 三Tam 四Tứ 五Ngũ 七Thất 八Bát 九Cửu 十Thập 阿A 老Lão 叔Thúc 伯Bá 弟Đệ 妹Muội 姐Tỷ 兄Huynh 公Công 主Chủ
爷爺家嘉Gia 后Hậu 帝Đế 女Nữ 师師Sư 真Chân 仙Tiên 魔Ma 道Đạo 少Thiếu 新Tân 豪Hào 健Kiện 博Bác 哲Triết 然Nhiên 旭Húc 泽澤Trạch 睿Duệ
景Cảnh 瑞Thụy 祥翔Tường 依伊Y 可Khả 馨Hinh 芷Chỉ 素Tố 洁潔Khiết 翠Thúy 蝶Điệp 凝Ngưng 渊淵Uyên 玄Huyền 苍蒼Thương 恒恆Hằng 卫衛Vệ
定Định 宏Hoành 铁鐵Thiết 承Thừa 航Hàng 川Xuyên 波Ba 溪Khê 泉Tuyền 黑Hắc 银銀Ngân 音Âm 母Mẫu 父Phụ 娘Nương 夫Phu 雁Nhạn 珊San
彤Đồng
"""
HAN_VIET = {ch: m.group(2) for m in re.finditer("([\u3400-\u9fff]+)(\\S+)", _HAN_VIET) for ch in m.group(1)}


def han_viet(name: str) -> str | None:
    """A Chinese name in Hán-Việt ("林风" -> "Lâm Phong", "林师姐" -> "Lâm Sư tỷ"); None when a character is unknown."""
    name = name.strip()
    if not 1 <= len(name) <= 5:
        return None
    for role, label in sorted(ROLES_VI.items(), key=lambda item: -len(item[0])):
        if name.endswith(role) and 1 < len(role) < len(name) and "\u4e00" <= role[0] <= "\u9fff":
            head = han_viet(name[:-len(role)])
            return f"{head} {label}" if head else None
    if not all(ch in HAN_VIET for ch in name):
        return None
    return " ".join(HAN_VIET[ch] for ch in name)


def role_label(name: str) -> str | None:
    """A title used as a speaker's label, in Vietnamese: 할머니 is Bà, "강 부장" is "Trưởng phòng Kang"."""
    name = name.strip()
    if name in ROLES_VI:
        return ROLES_VI[name]
    surname, _, title = name.partition(" ")
    if title in ROLES_VI and len(surname) == 1:
        return f"{ROLES_VI[title]} {_SURNAMES.get(surname) or romanized(surname) or surname}"
    return None


def romanized(name: str) -> str | None:
    """The name in Latin letters when it is written only in kana or hangul; None otherwise."""
    name = name.strip()
    if name in ROLES_VI or role_label(name):
        return None
    if KANA_NAME.fullmatch(name):
        return _kana(name)
    if HANGUL_NAME.fullmatch(name):
        # Two words are a person only as surname and given name ("김 철수"); "파티 초대" is a popup title.
        words = name.split()
        if len(words) > 1 and not (len(words[0]) == 1 and words[0] in _SURNAMES):
            return None
        return _hangul(name)
    return None
