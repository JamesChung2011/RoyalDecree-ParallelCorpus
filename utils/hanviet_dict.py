"""
Sino-Vietnamese (Hán-Việt) reading dictionary utility.
Maps CJK ideographs to their Sino-Vietnamese phonetic transcriptions.
Used for lexical and semantic similarity evaluation between Hán Nôm and Quốc Ngữ.
"""

import re
import unicodedata
from typing import List, Dict, Optional

# Basic high-frequency Sino-Vietnamese readings mapping for classical historical texts
# Covers common Nguyễn dynasty imperial administrative and historical vocabulary
HANVIET_MAP = {
    # Imperial & Titles
    "明": "minh", "命": "mệnh", "政": "chính", "要": "yếu", "節": "tiết", "次": "thứ", "奏": "tấu",
    "皇": "hoàng", "上": "thượng", "帝": "đế", "朕": "trẫm", "臣": "thần", "等": "đẳng",
    "機": "cơ", "密": "mật", "院": "viện", "國": "quốc", "史": "sử", "館": "quán",
    "嗣": "tự", "德": "đức", "紹": "thiệu", "治": "trị", "嘉": "gia", "隆": "long",
    "聖": "thánh", "旨": "chỉ", "諭": "dụ", "奉": "phụng", "欽": "khâm", "此": "thử",
    
    # Governance & Administration
    "爲": "vi", "为": "vi", "有": "hữu", "紀": "kỷ", "綱": "cương", "法": "pháp", "度": "độ",
    "之": "chi", "施": "thi", "公": "công", "平": "bình", "正": "chính", "大": "đại", "體": "thể",
    "苟": "cẩu", "操": "thao", "得": "đắc", "其": "kỳ", "則": "tắc", "不": "bất", "勞": "lao",
    "而": "nhi", "成": "thành", "若": "nhược", "昔": "tích", "唐": "đường", "虞": "ngu",
    "典": "điển", "謨": "mô", "夏": "hạ", "商": "thương", "周": "chu", "訓": "huấn", "誥": "cáo",
    "統": "thống", "灼": "chước", "然": "nhiên", "可": "khả", "鏡": "kính", "降": "hàng",
    "及": "cập", "後": "hậu", "世": "thế", "貞": "trinh", "觀": "quán", "開": "khai", "元": "nguyên",
    "宋": "tống", "三": "tam", "朝": "triều", "寶": "bảo", "篇": "thiên", "亦": "diệc", "足": "túc",
    "見": "kiến", "一": "nhất", "代": "đại", "所": "sở", "在": "tại", "我": "ngã",
    
    # Common verbs, adjectives, particles
    "實": "thực", "飾": "sức", "鼎": "đỉnh", "新": "tân", "百": "bách", "廢": "phế", "修": "tu",
    "舉": "cử", "求": "cầu", "領": "lãnh", "樞": "khu", "本": "bản", "於": "ư", "敬": "kính",
    "天": "thiên", "祖": "tổ", "勤": "cần", "愛": "ái", "民": "dân", "制": "chế", "兵": "binh",
    "恤": "tuất", "刑": "hình", "揆": "khuê", "文": "văn", "奮": "phấn", "武": "vũ", "賢": "hiền",
    "審": "thẩm", "官": "quan", "重": "trọng", "農": "nông", "崇": "sùng", "儉": "kiệm",
    "與": "dữ", "夫": "phu", "築": "trúc", "城": "thành", "堡": "bảo", "壯": "tráng", "封": "phong",
    "疆": "cương", "事": "sự", "溯": "tố", "自": "tự", "年": "năm", "至": "chí", "茲": "tư",
    "十": "thập", "八": "bát", "純": "thuần", "光": "quang", "前": "tiền", "古": "cổ", "請": "thỉnh",
    "派": "phái", "出": "xuất", "通": "thông", "曉": "hiểu", "學": "học", "者": "giả", "將": "tương",
    "諸": "chư", "朱": "châu", "批": "phê", "並": "tịnh", "起": "khởi", "居": "cư", "註": "chú",
    "編": "biên", "分": "phân", "凡": "phàm", "類": "loại", "纂": "toản", "書": "thư", "輪": "luân",
    "往": "vãng", "最": "tối", "辦": "biện", "付": "phó", "梓": "tử", "以": "dĩ", "下": "hạ",
    "傳": "truyền", "俾": "tỷ", "知": "tri", "端": "đoan", "乎": "hồ", "是": "thị", "疏": "sớ",
    "左": "tả", "待": "đãi", "漏": "lậu", "會": "hội", "再": "tái", "曰": "viết", "隋": "tùy",
    "室": "thất", "當": "đương", "紀": "kỷ", "略": "lược", "地": "địa", "九": "cửu", "月": "nguyệt",
    "日": "nhật", "宜": "nghi", "詳": "tường", "小": "tiểu", "相": "tương", "附": "phụ", "且": "thả",
    "須": "tu", "合": "hợp", "隨": "tùy", "勢": "thế", "否": "phủ", "則": "tắc", "復": "phức",
    "進": "tiến", "呈": "trình", "語": "ngữ", "多": "đa", "意": "ý", "但": "đãn", "攝": "nhiếp",
    "已": "dĩ", "備": "bị", "載": "tải", "初": "sơ", "設": "thiết", "屬": "thuộc", "内": "nội",
    "閣": "các", "恭": "cung", "錄": "lục", "問": "vấn", "何": "hà", "人": "nhân", "專": "chuyên",
    "未": "vị", "竣": "tuấn", "乃": "nãi", "履": "lý", "考": "khảo", "並": "tịnh", "置": "trí",
    "交": "giao", "商": "thương", "榷": "xác", "興": "hưng", "工": "công", "部": "bộ", "吏": "lại",
    "戶": "hộ", "禮": "lễ", "刑": "hình", "兵": "binh", "工": "công", "總": "tổng", "督": "đốc",
    "巡": "tuần", "撫": "phủ", "布": "bố", "政": "chính", "按": "án", "察": "sát", "道": "đạo",
    "府": "phủ", "縣": "huyện", "州": "châu", "鎭": "trấn", "省": "tỉnh", "京": "kinh", "城": "thành",
}


def nom_to_hanviet(text: str) -> List[str]:
    """Convert Hán Nôm text to a list of Hán-Việt readings."""
    readings = []
    for ch in text:
        if ch in HANVIET_MAP:
            readings.append(HANVIET_MAP[ch])
        elif '\u4e00' <= ch <= '\u9fff':
            # Chinese character without explicit map entry
            readings.append(ch)
    return readings


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text for comparison."""
    text = unicodedata.normalize('NFC', text.lower())
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_semantic_overlap(nom_text: str, vn_text: str) -> float:
    """
    Compute overlap ratio between Nôm Hán-Việt readings and Vietnamese words.
    Returns float between 0.0 and 1.0.
    """
    if not nom_text or not vn_text:
        return 0.0

    nom_readings = set(nom_to_hanviet(nom_text))
    vn_words = set(normalize_vietnamese_text(vn_text).split())

    if not nom_readings or not vn_words:
        return 0.0

    # Count common vocabulary
    intersection = nom_readings.intersection(vn_words)
    # Exclude single generic characters
    filtered_matches = {w for w in intersection if len(w) > 1}

    # If there are meaningful matches
    if len(filtered_matches) >= 2 or len(intersection) >= 3:
        return min(1.0, len(intersection) / max(len(nom_readings), 1) * 2.0)
    return 0.0

