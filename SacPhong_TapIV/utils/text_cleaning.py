"""
Text cleaning and normalization utilities for OCR pipeline.

Handles:
1. Hán Nôm (Chinese characters): CJK normalization, CJK punctuation, whitespace removal.
2. Quốc Ngữ (Vietnamese): Unicode NFC normalization, diacritics repair, punctuation spacing.
3. Filtering: Page numbers, running headers, scan noise, footnote markers.
"""

import re
import unicodedata
from typing import Tuple, Optional, List, Dict


# ==============================================================================
# 1. Hán Nôm (Chinese characters) text cleaning
# ==============================================================================

# Valid CJK Unicode ranges
def is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF) or   # CJK Unified Ideographs
        (0x3400 <= cp <= 0x4DBF) or   # CJK Extension A
        (0x20000 <= cp <= 0x2A6DF) or # CJK Extension B
        (0x2A700 <= cp <= 0x2B73F) or # CJK Extension C
        (0x2B740 <= cp <= 0x2B81F) or # CJK Extension D
        (0x2B820 <= cp <= 0x2CEAF) or # CJK Extension E
        (0x2CEB0 <= cp <= 0x2EBEF) or # CJK Extension F
        (0x30000 <= cp <= 0x3134F) or # CJK Extension G
        (0xF900 <= cp <= 0xFAFF) or   # CJK Compatibility Ideographs
        (0x2F800 <= cp <= 0x2FA1F) or # CJK Compat Ideographs Supp
        (0x3000 <= cp <= 0x303F) or   # CJK Symbols and Punctuation
        (0xFF01 <= cp <= 0xFF60) or   # Fullwidth forms
        ch in '、。！？；：「」『』【】〈〉《》〔〕（）\n'
    )


def clean_nom_text(text: str) -> str:
    """
    Clean Hán Nôm OCR text per academic rubric:
    - Retain only authentic CJK characters and traditional CJK punctuation
    - Zero spaces between characters
    - Deduplicate consecutive punctuation
    """
    if not text:
        return ""

    # Strip control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Filter non-CJK characters
    cleaned_chars = [ch for ch in text if is_cjk_char(ch)]
    result = ''.join(cleaned_chars)

    # Nôm must have no whitespace between characters
    result = re.sub(r'\s+', '', result)

    # Deduplicate punctuation
    result = re.sub(r'。{2,}', '。', result)
    result = re.sub(r'、{2,}', '、', result)
    result = re.sub(r'；{2,}', '；', result)

    return result.strip()


# ==============================================================================
# 2. Quốc Ngữ (Vietnamese) text cleaning & Errata corrections
# ==============================================================================

# Deterministic errata map for common OCR misreads and PDF digitization errors
VN_ERRATA_REPLACEMENTS = [
    # Historical names, titles, and reigns
    ("Khai-ngugên",         "Khai-nguyên"),
    ("Khai.ngugên",         "Khai-nguyên"),
    ("Khai.nguyên",         "Khai-nguyên"),
    ("(nh. quán",           "Trinh-quán"),
    ("(nh.quán",            "Trinh-quán"),
    ("T.inh.quán",          "Trinh-quán"),
    ("T.inh-quán",          "Trinh-quán"),
    ("Tinh.quán",           "Trinh-quán"),
    ("chính-ycu",           "chính-yếu"),
    ("chính gếu",           "chính-yếu"),
    ("Chính gếu",           "Chính-yếu"),
    ("bộ Ainh-mnênh",       "bộ Minh-mệnh"),
    ("Ainh-mnênh",          "Minh-mệnh"),
    ("Minh menh",           "Minh Mệnh"),
    ("menh chính",          "mệnh chính"),
    ("đệ-nhi-kở",           "đệ-nhị-kỷ"),
    ("dể-nhị-hộ",           "đệ-nhị-kỷ"),
    ("đề nhị- kỹ",          "đệ-nhị-kỷ"),
    ("dệ-nhịsEp",           "đệ-nhị-kỷ"),
    ("tườug",               "tường"),
    ("rö- rệt",             "rõ ràng"),

    # Numbers and corrupted OCR quantities
    ("6U người",            "60 người"),
    ("1U ngày",             "10 ngày"),
    ("tuôi tới 7U",         "tuổi tới 70"),

    # Syllables with incorrect tone marks from scan bleed
    ("mười tắm năm",        "mười tám năm"),
    ("tháng tắm năm",       "tháng tám năm"),
    ("lấy tắm người",       "lấy tám người"),
    ("vắng tắm hơi",        "vắng tăm hơi"),
    ("yên tắm tĩnh",        "yên tâm tĩnh"),
    ("Trắm",                "Trẫm"),
    ("trắm",                "trẫm"),
    ("®mhững",              "những"),
    ("®",                   ""),

    # Words following woodblock markers and chapter headers
    ("TÓNG MỤC",            "TỔNG MỤC"),
    ("TÓNG",                "TỔNG"),
    ("TỎNG-TÀI",            "TỔNG-TÀI"),
    ("RĂẮM",                "NĂM"),
    ("XIINH-MỆNH",          "MINH-MỆNH"),
    ("Quyên IV",            "Quyển IV"),
    ("Quyên ",              "Quyển "),
    ("Quau Khâm-thiên-giám", "Quan Khâm-thiên-giám"),
    ("Quau ",               "Quan "),
    ("xín chiếu",           "xin chiếu"),
    ("thử muời",            "thứ mười"),
    ("thử muờ",             "thứ mười"),
    ("Xinh nghĩ",           "Kính nghĩ"),
    ("MƯƠA",                "MƯƠI"),
    ("thớtím",              "thứ chín"),
    ("thứ tím",             "thứ chín"),
    ("Mùa thủ",             "Mùa thu"),
    ("nhuận thêm một thắng", "nhuận thêm một tháng"),
    ("JJinh-mệnh",          "Minh-mệnh"),
    ("Thiánh-thượng",       "Thánh-thượng"),
    ("cháu phê",            "châu phê"),
    ("phđ-biến",            "phổ-biến"),
    ("khỏng chút",          "không chút"),
    ("khỏng có",            "không có"),
    ("khỏng",               "không"),

    # Micro-orthography and dot-in-word scan noise
    ("bát. phầm",           "bát-phẩm"),
    ("bát phầm",            "bát-phẩm"),
    ("chánh lc. phầm",      "chánh lục-phẩm"),
    ("lc. phầm",            "lục-phẩm"),
    ("lc phầm",             "lục-phẩm"),
    ("Quản-tào-cainh",      "Quản-tào-chính"),
    ("Thừa- thiên",         "Thừa-thiên"),
    ("Thửa- ty",            "Thừa-ty"),
    ("Thửa-thiên",          "Thừa-thiên"),
    ("Phủ-đoãn",            "Phủ-doãn"),
    ("tuồi",                "tuổi"),
    ("bàn đình:",           "bàn định:"),
    ("Đình. thần",          "Đình-thần"),
    ("địa. phương",         "địa-phương"),
    ("Huấn- đạo",           "Huấn-đạo"),
    ("thiết. yếu",          "thiết-yếu"),
    ("yếu. chỉ",            "yếu-chỉ"),
    ("khó. nhọc",           "khó-nhọc"),
    ("Cơ-mảt",              "Cơ-mật"),
    ("nguy thẳng",          "ngay thẳng"),
    ("giữ vỡng",            "giữ vững"),
    ("Phủ-đồng-ri",         "Phủ-đồng-tri"),
    ("Tòng-ng8- phầm",      "Tòng-ngũ-phẩm"),
    ("Thượng-bẳo-thiếu-khanh", "Thượng-bảo-thiếu-khanh"),
    ("tâảu cử",             "tâu cử"),

    # Typography and quotation marks
    ("\u2039 ",             "\u00ab"),   # ‹ → «
    (" \u203a",             "\u00bb"),   # › → »
    (" l\u203a",            "\u00bb"),   # l› → »
    (" \u00cc",             "\u00bb"),   # Ì → »
    (" \u00cf",             "\u00bb"),   # Ï → »
    (" t,",                 "."),
]

# Character mappings for OCR confusions
VN_CHAR_FIXES = [
    ('ð', 'đ'),
    ('Ð', 'Đ'),
    ('\u2018', "'"),
    ('\u2019', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
    ('…', '...'),
]


def normalize_brackets(text: str) -> str:
    """
    Standardize all brackets in Vietnamese text strictly per project rule:
    - Only square brackets [ ] or guillemets « » are allowed.
    - Quote/angle bracket variations (‹, ›, <, >) -> « »
    - All other brackets ((, ), {, }, 【, 】, 〔, 〕, 〈, 〉) default to [ ]
    - Standardize broken woodblock folio markers ({5-b], |4-bJ, [10-b ]) -> [X-a] / [X-b]
    """
    if not text:
        return ""

    # 1. Standardize woodblock markers first before other conversions
    # Matches patterns like {5-b], [1-b}, [10. a], |4-bJ, (1a]
    text = re.sub(
        r'(?:\[|\(|\{|\/|\|)\s*(\d{1,2})\s*[-._\s]*\s*([abAB])\s*(?:\]|\)|\}|\/|\||[Jj])',
        r'[\1-\2]',
        text
    )

    # 2. Convert guillemet variations to standard French quotation marks « »
    text = text.replace('‹', '«').replace('›', '»')
    text = re.sub(r'<\s*([^<>]+?)\s*>', r'«\1»', text)

    # 3. Default all other bracket styles to square brackets [ ]
    open_brackets = ['(', '{', '【', '〔', '〈']
    for b in open_brackets:
        text = text.replace(b, '[')

    close_brackets = [')', '}', '】', '〕', '〉']
    for b in close_brackets:
        text = text.replace(b, ']')

    # 4. Clean up any redundant empty brackets or duplicate brackets
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\[\s*\[', '[', text)
    text = re.sub(r'\]\s*\]', ']', text)

    return text


def clean_vn_text(text: str) -> str:
    """
    Clean Vietnamese text:
    - Standardize Unicode to NFC (canonical decomposition followed by canonical composition)
    - Apply deterministic OCR errata corrections
    - Remove orphaned footnote indicators while preserving woodblock markers ([1a], [3-b]...)
    - Format correct spacing around punctuation
    - Normalize all brackets strictly to [ ] or « » (defaulting other brackets to [ ])
    """
    if not text:
        return ""

    # Strip control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Normalize Unicode NFC
    text = unicodedata.normalize('NFC', text)

    # Apply errata replacements
    for bad, good in VN_ERRATA_REPLACEMENTS:
        text = text.replace(bad, good)

    # Character confusion fixes
    for old, new in VN_CHAR_FIXES:
        text = text.replace(old, new)

    # Remove isolated footnote indicators: e.g. (1), [1], (†), (3)
    # Be careful not to remove woodblock markers like [1a], [2-b], [3.4]
    text = re.sub(r'\(\s*[†\d]+\s*\)', '', text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Standardize all brackets strictly to [ ] or « »
    text = normalize_brackets(text)

    # Ensure space after punctuation (.,;:!?) when followed by a letter
    text = re.sub(r'([.!?;:,])([A-ZĐa-zđÀ-ỹ])', r'\1 \2', text)

    # Remove unwanted space before punctuation
    text = re.sub(r'\s+([.!?;:,])', r'\1', text)

    return text.strip()


def clean_woodblock_markers(text: str) -> str:
    """
    Clean editorial woodblock markers per user specification:
    - Recognize every single square bracket [...] as a woodblock marker.
    - Discard noise/junk characters immediately following the square bracket.
    - Only accept the first meaningful word (or meaningful word if run through spell check).
    """
    if not text:
        return ""

    # 1. Match any paired bracket [...] followed by noise symbols until the first meaningful word
    pattern = r'\[[^\]]*\](?:\s*[:;,.*+=‡~—\-«»\d/\\#|]+\s*|\s*s\s+|\s*)'
    text = re.sub(pattern, ' ', text)

    # 2. Fix OCR artifacts where 't' or other letters became '[' or ']'
    text = text.replace('[rên', 'trên')

    # 3. Clean any remaining unclosed or unopened brackets [ or ] (e.g. enumeration markers '1]', '2]')
    text = re.sub(r'\b\d+\s*\]', '', text)
    text = re.sub(r'\[|\]', '', text)

    # 4. Standardize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 5. Apply errata replacements
    for bad, good in VN_ERRATA_REPLACEMENTS:
        text = text.replace(bad, good)

    # 6. Format punctuation
    text = re.sub(r'\s+([,.:;!?])', r'\1', text)
    text = re.sub(r'([,.:;!?])([A-ZĐa-zđÀ-ỹ])', r'\1 \2', text)

    return text.strip()


# ==============================================================================
# Woodblock Folio Marker Recognition & Normalization
# ==============================================================================

CORRUPTED_FOLIO_DIGITS: Dict[str, str] = {
    '†': '1', '!': '1', '‡': '4', '1H': '11', 'S0': '50', '%2': '52', '4Š': '45',
}

CORRUPTED_FOLIO_FACES: Dict[str, str] = {
    '4': 'a', 's': 'a', '6': 'b', '8': 'b', '+': 'b', 'thượng': 'a', 'hạ': 'b',
}


def parse_woodblock_marker(content: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse the interior of any square bracket to detect woodblock leaf markers:
    Returns (folio_number, face 'a'|'b') or (None, None).
    Handles historical woodblock terminology ('thượng' -> a, 'hạ' -> b)
    and common OCR font confusions on digits and leaf faces.
    """
    cleaned = content.strip()
    if not cleaned:
        return None, None

    lower = cleaned.lower()
    if lower in ('thượng', 'hạ'):
        return None, CORRUPTED_FOLIO_FACES[lower]

    for bad, good in CORRUPTED_FOLIO_DIGITS.items():
        cleaned = cleaned.replace(bad, good)

    m = re.match(r'^(\d{1,2})[-._\s]*([abAB4sS68+])$', cleaned)
    if m:
        folio = int(m.group(1))
        face_raw = m.group(2).lower()
        face = CORRUPTED_FOLIO_FACES.get(face_raw, face_raw)
        return folio, face

    return None, None


def read_woodblock_markers(text: str) -> List[Dict]:
    """
    Read and parse every square bracket occurrence in the text as a woodblock marker.
    For each marker, determines:
    - raw: Original bracketed string e.g. '[†-s]'
    - canonical: Normalized standard form e.g. '[1-a]'
    - folio: Integer leaf number (1..60)
    - face: 'a' (recto / thượng) or 'b' (verso / hạ)
    - first_word: The first authentic meaningful word immediately following the marker.
    """
    if not text:
        return []

    markers = []
    # Match any square bracket and capture text following it
    pattern = re.compile(r'\[([^\]]*)\](?:\s*[:;,.*+=‡~—\-«»\d/\\#|]+\s*|\s*s\s+|\s*)(\S+)?')
    for m in pattern.finditer(text):
        content = m.group(1)
        raw_word = m.group(2) or ""
        folio, face = parse_woodblock_marker(content)

        # Spellcheck / correct the first word if present in errata
        clean_word = raw_word
        for bad, good in VN_ERRATA_REPLACEMENTS:
            if clean_word == bad:
                clean_word = good
                break

        canonical = f"[{folio}-{face}]" if folio and face else (f"[{face}]" if face else f"[{content}]")
        markers.append({
            "raw": m.group(0),
            "content": content,
            "folio": folio,
            "face": face,
            "canonical": canonical,
            "first_word": clean_word,
            "start": m.start(),
            "end": m.end()
        })

    return markers


def normalize_woodblock_markers(text: str) -> str:
    """
    Standardize all recognized woodblock folio markers in running text into [X-a] / [X-b] format,
    cleaning all noise punctuation and accepting the first meaningful word immediately following.
    """
    if not text:
        return ""

    def replace_marker(m):
        content = m.group(1)
        folio, face = parse_woodblock_marker(content)
        if face and not folio:
            return f" [{face}] "
        if folio and face:
            return f" [{folio}-{face}] "
        return f" [{content}] "

    # Pattern matches bracket and strips trailing noise symbols before the next word
    pattern = r'\[([^\]]*)\](?:\s*[:;,.*+=‡~—\-«»\d/\\#|]+\s*|\s*s\s+|\s*)'
    text = re.sub(pattern, replace_marker, text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==============================================================================
# 3. Filtering and Layout Cleansing
# ==============================================================================

def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers (Arabic or Roman numerals)."""
    text = re.sub(r'^\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def remove_headers(text: str, book_name: str = "") -> str:
    """Remove repeated book title headers from content lines."""
    patterns = [
        "MINH MỆNH CHÍNH YẾU",
        "MINH-MỆNH CHÍNH-YẾU",
        "MINH MỆNH CHÍNH YẾU TOÀN THƯ",
    ]
    if book_name:
        patterns.extend([
            book_name.upper(),
            book_name,
            book_name.replace(' ', '-').upper()
        ])

    for pat in set(patterns):
        text = text.replace(pat, '')

    return text.strip()


def is_running_header_box(text: str, y_max: float = 0) -> bool:
    """
    Check if a box is a running header at the top margin (y_max <= 260).
    Filters book title variations, chapter volume numbers, and scan margin debris.
    """
    t_clean = text.strip()
    if not t_clean:
        return True

    # Header zone is top margin (y <= 260)
    if y_max > 260:
        return False

    # Book title corruptions in running header
    if re.search(r'(MỆNH|CHÍNH|YẾU|CHUNN|KHÊNH|MINH|MIÌINH|IIINH)', t_clean, re.IGNORECASE):
        return True

    # Volume headers: QUYỂN / QUYỀN <num> <page>
    if re.search(r'^(QUY[ÊỀỂỄỆ][N\s]|QUY[EÈẺẼẸ][N\s])', t_clean, re.IGNORECASE):
        return True

    # Isolated scan codes or short margin characters
    if len(t_clean) <= 4 or re.match(r'^[a-zA-Z0-9\s.:,%#"\'\\-_/]+$', t_clean):
        return True

    return False


def is_noise_box(text: str, min_chars: int = 1, y_max: float = 0) -> bool:
    """Identify if a bounding box is scanner noise, blank, or unwanted artifact."""
    if not text or len(text.strip()) < min_chars:
        return True

    t_clean = text.strip()

    # Bounding box without any alphanumeric characters (e.g. '^', '®', '#.', '"')
    if not any(c.isalnum() for c in t_clean):
        return True

    # Pure standalone numbers
    if re.match(r'^\d{1,4}$', t_clean):
        return True

    # Pure Roman numerals
    if re.match(r'^[IVXLCDM]+$', t_clean):
        return True

    # Top margin running header check
    if y_max > 0 and is_running_header_box(t_clean, y_max):
        return True

    return False
