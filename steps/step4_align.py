"""
Step 4: Sentence & Paragraph Alignment between Hán Nôm and Quốc Ngữ.

Aligns Hán Nôm woodblock half-pages (ch/) with modern Vietnamese translations (vi/)
at the whole-woodblock leaf level (1 pair per woodblock image).
Implements:
  1. Section-aware historical partitioning (8 authentic sections from prefaces to Quyển 4)
  2. Proportional clause-level text distribution ensuring balanced lengths and 0 micro-fragments
  3. Proper sentence_id naming: <img_id without .jpg>.1
  4. Single-line formatting (continuous CJK text, normalized spaces for Vietnamese)
  5. Footnote exclusion (omits bottom-margin translator footnotes per rubric page 7)
  6. UTF-8 with BOM CSV and color-coded XLSX output

Usage:
    python -m steps.step4_align [--config config.yaml]
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.hanviet_dict import nom_to_hanviet, compute_semantic_overlap
from utils.text_cleaning import clean_woodblock_markers


# Authentic historical book divisions
# (Section Name, Nôm Start Page, Nôm End Page, Vietnamese Start Page, Vietnamese End Page)
SECTIONS = [
    ("Tiết thứ tấu", 228, 237, 9, 13),
    ("Lời biểu", 238, 249, 15, 19),
    ("Phàm lệ", 250, 251, 21, 22),
    ("Tổng mục", 254, 260, 23, 27),
    ("Quyển 1", 262, 351, 31, 71),
    ("Quyển 2", 352, 431, 75, 115),
    ("Quyển 3", 433, 522, 117, 163),
    ("Quyển 4", 523, 632, 167, 224),
]


def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_overrides(overrides_path="alignment_overrides.yaml") -> dict:
    if os.path.exists(overrides_path):
        import yaml
        try:
            with open(overrides_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def load_label_file(label_path: str) -> Dict[str, List[Dict]]:
    result = {}
    if not os.path.exists(label_path):
        return result

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            img_name = os.path.basename(parts[0])
            try:
                boxes = json.loads(parts[1])
                result[img_name] = boxes
            except json.JSONDecodeError:
                result[img_name] = []

    return result


def extract_page_num_from_filename(filename: str) -> int:
    match = re.search(r'\.(\d{3,})\.jpg$', filename)
    return int(match.group(1)) if match else 0


def get_full_text_nom(boxes: List[Dict]) -> str:
    """Nôm text: continuous characters, no spaces, no newlines per rubric."""
    chars = [b.get("transcription", "").strip() for b in boxes if b.get("transcription")]
    joined = "".join(chars)
    return re.sub(r'\s+', '', joined).strip()


def get_clean_page_vn_text(boxes: List[Dict]) -> str:
    """Extract clean Vietnamese text from boxes, strictly excluding bottom footnotes per rubric."""
    footnote_start = None
    for i, b in enumerate(boxes):
        pts = b.get("points", [])
        y0 = min(p[1] for p in pts) if pts else 0
        t = b.get("transcription", "").strip()
        # Footnote section starts near bottom margin with markers like (1), (:), (~), (t), etc.
        if y0 >= 1880 and re.match(r'^[(\[]?[†\d:*~=\-—]+[)\]]?', t):
            footnote_start = i
            break

    kept_boxes = boxes[:footnote_start] if footnote_start is not None else boxes
    lines = [b.get("transcription", "").strip() for b in kept_boxes if b.get("transcription")]
    joined = " ".join(lines)
    return clean_woodblock_markers(joined)


def split_into_fine_tokens(text: str) -> List[str]:
    """Split text by natural punctuation boundaries: sentence ends, semicolons, colons, commas."""
    tokens = re.split(r'(?<=[.!?;,:—«»])\s+', text)
    cleaned_tokens = []
    for t in tokens:
        t = t.strip()
        if t:
            cleaned_tokens.append(t)
    return cleaned_tokens


def build_alignment(cfg: dict) -> List[Dict]:
    book_name = cfg["book_name"]
    output_dir = Path(cfg["output_dir"]) / book_name

    ch_label_path = output_dir / "ch" / "Label.txt"
    vi_label_path = output_dir / "vi" / "Label.txt"

    ch_data = load_label_file(str(ch_label_path))
    vi_data = load_label_file(str(vi_label_path))

    sorted_nom_images = sorted(ch_data.keys(), key=extract_page_num_from_filename)
    num_nom = len(sorted_nom_images)
    sections_cfg = cfg.get("sections")
    if sections_cfg:
        sections_to_use = [
            (s["name"], s["nom_start"], s["nom_end"], s["vi_start"], s["vi_end"])
            for s in sections_cfg
        ]
    else:
        sections_to_use = SECTIONS

    print(f"Loaded {num_nom} Nôm woodblock pages across {len(sections_to_use)} sections.")

    alignments = []

    for sec_name, ns, ne, vs, ve in sections_to_use:
        # 1. Collect Nôm pages in this section (preserving order)
        sec_nom_images = [
            img for img in sorted_nom_images
            if ns <= extract_page_num_from_filename(img) <= ne
        ]

        if not sec_nom_images:
            continue

        # 2. Collect full raw Vietnamese text for this section
        sec_vi_boxes = []
        for p in range(vs, ve + 1):
            fname = f"{book_name}.{p:03d}.jpg"
            if fname in vi_data:
                boxes = vi_data[fname]
                footnote_start = None
                for i, b in enumerate(boxes):
                    pts = b.get("points", [])
                    y0 = min(pt[1] for pt in pts) if pts else 0
                    t = b.get("transcription", "").strip()
                    if y0 >= 1880 and re.match(r'^[(\[]?[†\d:*~=\-—]+[)\]]?', t):
                        footnote_start = i
                        break
                kept = boxes[:footnote_start] if footnote_start is not None else boxes
                lines = [b.get("transcription", "").strip() for b in kept if b.get("transcription")]
                if lines:
                    sec_vi_boxes.append(" ".join(lines))

        raw_sec_vi = " ".join(sec_vi_boxes)

        # 1. Non-marker brackets with spaces, diacritics, or long glosses -> « ... »
        def bracket_sub(m):
            c = m.group(1).strip()
            if ' ' in c or len(c) > 6 or re.search(r'[à-ỹÀ-Ỹ]', c):
                return f'«{c}»'
            return m.group(0)
        raw_sec_vi = re.sub(r'\[([^\]»\n]+)[\]»]', bracket_sub, raw_sec_vi)

        # 2. Standardize known OCR confusions in markers (Tập 1 and Tập 2)
        rep = [
            ('|4-bJ', '[4-b]'), ('{5-b]', '[5-b]'), ('[3.4]', '[3-a]'), ('[‡-a]', '[4-a]'),
            ('[†-s]', '[1-a]'), ('[4+]', '[4-b]'), ('[!-b]', '[1-b]'), ('[4-s]', '[4-a]'),
            ('[7-4]', '[7-a]'), ('[1H-b]', '[11-b]'), ('[12-4]', '[12-a]'), ('[20-4]', '[20-a]'),
            ('[24-4]', '[24-a]'), ('[28-6]', '[28-b]'), ('[S0-b]', '[50-b]'), ('[%2-a]', '[52-a]'),
            ('[36.4]', '[36-a]'), ('[38-4]', '[38-a]'), ('[41-8]', '[41-b]'), ('[43-4]', '[43-a]'),
            ('[4Š-b]', '[45-b]'), ('[50-4]', '[50-a]'), ('[54-8]', '[54-b]'), ('(hiểu sói', 'hiểu sói'),
            ('[100 năm]', ''),
            ('[đa]', '[4-a]'), ('[3k]', '[3-b]'), ('[56]', '[5-b]'), ('[7e]', '[7-a]'),
            ('[1la]', '[11-a]'), ('[Q0a]', '[20-a]'), ('[8e]', '[8-a]'), ('[&b]', '[8-b]'),
            ('[30-b]', '[10-b]'), ('[†la]', '[11-a]'), ('[†1b]', '[11-b]'), ('[†b]', '[12-b]'),
            ('[14k]', '[14-b]'), ('[15h]', '[15-b]'), ('[1⁄s]', '[17-a]'), ('[90s]', '[20-a]'),
            ('[Q5a]', '[23-a]'), ('[Q4e]', '[24-a]'), ('[Q4b]', '[24-b]'), ('[Q6e]', '[26-a]'),
            ('[Q96b]', '[26-b]'), ('[Q9a]', '[29-a]'), ('[506]', '[30-b]'), ('[3%b]', '[31-b]'),
            ('[56e]', '[36-a]'), ('[78-b]', '[38-b]'), ('[40s]', '[40-a]'), ('[4u]', '[43-b]'),
            ('[58-a]', '[50-a]'), ('[51h]', '[51-a]'), ('[fb]', '[1-b]'), ('[5ae]', '[3-a]'),
            ('[1⁄4a]', '[14-a]'), ('[16s]', '[16-a]'), ('[18s]', '[18-a]'), ('[91e]', '[21-a]'),
            ('[39s]', '[22-a]'), ('[55e]', '[33-a]'), ('[55s]', '[35-a]'), ('[45e]', '[43-a]'),
            ('[46e]', '[46-a]'), ('[47ae]', '[47-a]'), ('[50s]', '[50-a]'), ('[50%]', '[50-b]'),
            ('[1-b»', '[1-b]'), ('[15-a»', '[15-a]'), ('[6-b»', '[6-b]'), ('[48-a»', '[48-a]'),
            ('[49-a»', '[49-a]'), ('[6-a»', '[6-a]'),
        ]
        for r1, r2 in rep:
            raw_sec_vi = raw_sec_vi.replace(r1, r2)

        # 3. Analyze Nôm text and character weights
        nom_info = []
        for img in sec_nom_images:
            nom_text = get_full_text_nom(ch_data[img])
            nom_info.append((img, nom_text, len(nom_text)))

        num_pages = len(nom_info)

        # 4. Locate all woodblock markers in raw_sec_vi
        cands = list(re.finditer(r'\[\s*([0-9a-zA-Z%]+)[-._\s]*([abABeEsSkKhH%])\s*\]', raw_sec_vi))
        curr_target = -1
        filtered_anchors = []
        for m in cands:
            d_str = m.group(1).upper()
            f_str = m.group(2).lower()
            if f_str in ['e', 's']: f_str = 'a'
            if f_str in ['k', 'h', '%', '6']: f_str = 'b'
            try:
                d = int(d_str)
            except ValueError:
                continue
            exp_folio = (curr_target // 2) + 1 if curr_target >= 0 else 1
            options = [d]
            if d == 9: options.append(2)
            if d == 5: options.append(3)
            if d == 19: options.append(12)
            if d == 15: options.append(13)
            if d == 1: options.append(11)
            if 90 <= d <= 99:
                options.append(20 + (d - 90))
            if 50 <= d <= 59:
                if exp_folio <= 45: options.append(30 + (d - 50))
                else: options.append(d)
            if 40 <= d <= 49:
                options.append(d)
                if d == 49 and exp_folio <= 43: options.append(42)
                if d == 45 and exp_folio <= 43: options.append(43)
                if d == 47 and exp_folio <= 43: options.append(43)
            if d == 29 and 31 <= exp_folio <= 33: options.append(32)
            if d == 36 and exp_folio >= 50: options.append(56)
            if d == 55 and exp_folio >= 50: options.append(53); options.append(55)
            if d == 54 and exp_folio >= 50: options.append(54)
            if d == 45 and exp_folio >= 50: options.append(54)

            valid = [
                (opt - 1) * 2 + (0 if f_str == 'a' else 1)
                for opt in options
                if curr_target < (opt - 1) * 2 + (0 if f_str == 'a' else 1) < num_pages
            ]
            if valid:
                best_t = min(valid)
                filtered_anchors.append({
                    'target_idx': best_t,
                    'folio': best_t // 2 + 1,
                    'face': 'a' if best_t % 2 == 0 else 'b',
                    'start': m.start(),
                    'end': m.end()
                })
                curr_target = best_t

        # 5. Partition raw_sec_vi into num_pages slots using anchors
        page_texts = ["" for _ in range(num_pages)]

        if not filtered_anchors:
            tokens = split_into_fine_tokens(raw_sec_vi)
            page_tokens = [[] for _ in range(num_pages)]
            for i, tok in enumerate(tokens):
                page_tokens[i % num_pages].append(tok)
            page_texts = [" ".join(pt) for pt in page_tokens]
        else:
            first_a = filtered_anchors[0]
            if first_a["target_idx"] > 0:
                pre_text = raw_sec_vi[:first_a["start"]].strip()
                tokens = split_into_fine_tokens(pre_text)
                k = first_a["target_idx"]
                for idx, tok in enumerate(tokens):
                    page_texts[idx % k] += " " + tok

            for i in range(len(filtered_anchors)):
                curr_a = filtered_anchors[i]
                start_pos = curr_a["end"]
                end_pos = filtered_anchors[i+1]["start"] if i + 1 < len(filtered_anchors) else len(raw_sec_vi)
                chunk = raw_sec_vi[start_pos:end_pos].strip()

                curr_idx = curr_a["target_idx"]
                next_idx = filtered_anchors[i+1]["target_idx"] if i + 1 < len(filtered_anchors) else num_pages

                span_len = next_idx - curr_idx
                if span_len == 1:
                    page_texts[curr_idx] = chunk
                else:
                    sub_nom = [nom_info[curr_idx + j][2] for j in range(span_len)]
                    total_sub = sum(sub_nom) or 1
                    tokens = split_into_fine_tokens(chunk)
                    if len(tokens) < span_len:
                        tokens = [t.strip() for t in re.split(r'(?<=[,!?;:—])\s+|\s+', chunk) if t.strip()]
                    tot_vi = sum(len(t) for t in tokens)

                    targets = [max(w, 15) / max(total_sub, 1) * tot_vi for w in sub_nom]
                    s_t = sum(targets) or 1.0
                    targets = [t * tot_vi / s_t for t in targets]
                    cum_t = []
                    c = 0
                    for t in targets:
                        c += t
                        cum_t.append(c)

                    p_toks = [[] for _ in range(span_len)]
                    c_page = 0
                    c_ass = 0
                    for i_tok, tok in enumerate(tokens):
                        rem_pages = (span_len - 1) - c_page
                        rem_toks = len(tokens) - i_tok
                        if rem_toks <= rem_pages and len(p_toks[c_page]) > 0:
                            c_page += 1
                        elif c_page < span_len - 1 and c_ass + len(tok) > cum_t[c_page]:
                            if len(p_toks[c_page]) > 0:
                                c_page += 1
                        p_toks[c_page].append(tok)
                        c_ass += len(tok)

                    for j in range(span_len):
                        if not p_toks[j] and j > 0 and len(p_toks[j-1]) > 1:
                            p_toks[j].append(p_toks[j-1].pop())
                        page_texts[curr_idx + j] = " ".join(p_toks[j]).strip()

        # 6. Build aligned records
        for i, (img, nom_text, l_nom) in enumerate(nom_info):
            raw_vn = page_texts[i]
            vn_text = clean_woodblock_markers(raw_vn)
            vn_text = re.sub(r'\s+', ' ', vn_text).strip()
            base_id = img.replace(".jpg", "")

            alignments.append({
                "img_id": img,
                "sentence_id": f"{base_id}.1",
                "SinoNom_Char": nom_text,
                "ChuQN_txt": vn_text,
            })

        print(f"  ✓ {sec_name:15s}: {num_pages:3d} pages aligned via {len(filtered_anchors)} woodblock anchors")

    # Apply manual overrides if present in alignment_overrides.yaml
    overrides = load_overrides()
    if overrides:
        overridden_count = 0
        for row in alignments:
            img = row["img_id"]
            p_num = extract_page_num_from_filename(img)
            val = None
            if img in overrides:
                val = overrides[img]
            elif p_num in overrides:
                val = overrides[p_num]
            elif str(p_num) in overrides:
                val = overrides[str(p_num)]

            if val is not None:
                if isinstance(val, str):
                    row["ChuQN_txt"] = re.sub(r'\s+', ' ', val).strip()
                    overridden_count += 1
                elif isinstance(val, dict) and "ChuQN_txt" in val:
                    row["ChuQN_txt"] = re.sub(r'\s+', ' ', str(val["ChuQN_txt"])).strip()
                    overridden_count += 1
        if overridden_count > 0:
            print(f"  ★ Applied {overridden_count} manual overrides from alignment_overrides.yaml!")

    print(f"\n✓ Generated {len(alignments)} section-aware balanced pairs across {num_nom} Nôm pages!")
    return alignments


def build_sentence_alignment(page_records: List[Dict]) -> List[Dict]:
    """
    Subdivide page-level alignments into granular sentence-level pairs:
    - Splits Vietnamese text by sentence terminals (. ! ?)
    - Maps Nôm text proportionally while snapping to classical boundary particles (也, 矣, 哉, 耳, 者, 曰, 奏)
    - Generates sentence_id: <img_id without .jpg>.<sentence_number>
    """
    sentence_records = []

    for row in page_records:
        img_id = row['img_id']
        base_id = img_id.replace('.jpg', '')
        nom_text = str(row.get('SinoNom_Char', '')) if pd.notna(row.get('SinoNom_Char')) else ''
        vn_text = str(row.get('ChuQN_txt', '')) if pd.notna(row.get('ChuQN_txt')) else ''

        # Clean internal dot before lowercase words
        vn_text = re.sub(r'([A-Za-zĐđÀ-ỹ]+)\.\s+([a-zđà-ỹ])', r'\1 \2', vn_text)

        # Split on sentence terminals followed by uppercase or quote
        raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-ZĐ«])', vn_text)
        vn_sents = [s.strip() for s in raw_sents if len(s.strip()) > 5]
        if not vn_sents:
            vn_sents = [vn_text]

        num_s = len(vn_sents)
        if num_s == 1:
            sentence_records.append({
                'img_id': img_id,
                'sentence_id': f'{base_id}.1',
                'SinoNom_Char': nom_text,
                'ChuQN_txt': vn_text
            })
            continue

        total_nom = len(nom_text)

        # If total Nôm characters is small, merge sentences so every sentence has at least ~8 Nôm chars
        max_sents = max(1, total_nom // 8)
        if len(vn_sents) > max_sents:
            merged = []
            chunk = []
            for s in vn_sents:
                chunk.append(s)
                if len(merged) < max_sents - 1 and len(' '.join(chunk)) >= len(vn_text) / max_sents:
                    merged.append(' '.join(chunk))
                    chunk = []
            if chunk:
                merged.append(' '.join(chunk))
            vn_sents = merged

        num_s = len(vn_sents)
        if num_s == 1:
            sentence_records.append({
                'img_id': img_id,
                'sentence_id': f'{base_id}.1',
                'SinoNom_Char': nom_text,
                'ChuQN_txt': vn_sents[0]
            })
            continue

        total_vn = sum(len(s) for s in vn_sents) or 1

        nom_slices = []
        curr_idx = 0
        cum_vn = 0
        boundary_chars = '也矣哉耳者曰日奏。！？'

        for i, vs in enumerate(vn_sents):
            if i == num_s - 1:
                nom_slices.append(nom_text[curr_idx:])
            else:
                cum_vn += len(vs)
                ideal_cut = int(round(cum_vn * total_nom / total_vn))
                best_cut = ideal_cut
                win_start = max(curr_idx + 1, ideal_cut - 4)
                win_end = min(total_nom - 1, ideal_cut + 5)
                for pos in range(win_start, win_end):
                    if nom_text[pos] in boundary_chars:
                        best_cut = pos + 1
                        break

                rem_sents = num_s - 1 - i
                max_allowed_end = total_nom - rem_sents
                end_idx = min(max_allowed_end, max(curr_idx + 1, best_cut))
                nom_slices.append(nom_text[curr_idx:end_idx])
                curr_idx = end_idx

        for i in range(num_s):
            sentence_records.append({
                'img_id': img_id,
                'sentence_id': f'{base_id}.{i+1}',
                'SinoNom_Char': nom_slices[i],
                'ChuQN_txt': vn_sents[i]
            })

    return sentence_records


def save_alignment(alignments: List[Dict], cfg: dict):
    book_name = cfg["book_name"]
    output_dir = Path(cfg["output_dir"]) / book_name

    # 1. Whole-Woodblock Half-Leaf Alignment (401 pairs)
    csv_path = output_dir / f"{book_name}.csv"
    df = pd.DataFrame(alignments, columns=["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"])
    df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    print(f"Written Woodblock CSV: {csv_path} ({len(df)} rows)")

    xlsx_path = output_dir / f"{book_name}.xlsx"
    save_colored_xlsx(df, str(xlsx_path))
    print(f"Written Woodblock XLSX: {xlsx_path}")

    # 2. Fine-grained Sentence-Level Sub-divided Alignment (~1,447 pairs)
    sent_alignments = build_sentence_alignment(alignments)
    sent_csv_path = output_dir / f"{book_name}_sentences.csv"
    sent_df = pd.DataFrame(sent_alignments, columns=["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"])
    sent_df.to_csv(str(sent_csv_path), index=False, encoding="utf-8-sig")
    print(f"Written Sentence CSV:  {sent_csv_path} ({len(sent_df)} rows)")

    sent_xlsx_path = output_dir / f"{book_name}_sentences.xlsx"
    save_colored_xlsx(sent_df, str(sent_xlsx_path))
    print(f"Written Sentence XLSX: {sent_xlsx_path}")


def save_colored_xlsx(df: pd.DataFrame, xlsx_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Alignment"

    headers = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    black_font = Font(name="Times New Roman", size=11, color="000000")
    red_font = Font(name="Times New Roman", size=11, color="FF0000")
    nom_font = Font(name="NomNaTongLight", size=12, color="000000")

    wrap_align = Alignment(vertical="top", wrap_text=True)

    for row_idx, row in df.iterrows():
        r = row_idx + 2

        c1 = ws.cell(row=r, column=1, value=row["img_id"])
        c1.alignment = Alignment(vertical="top")
        c2 = ws.cell(row=r, column=2, value=row["sentence_id"])
        c2.alignment = Alignment(vertical="top")

        nom_text = str(row["SinoNom_Char"] or "")
        qn_text = str(row["ChuQN_txt"] or "")

        # Color evaluation:
        if nom_text.strip() and qn_text.strip():
            font = nom_font
            qn_f = black_font
        else:
            font = red_font
            qn_f = red_font

        c3 = ws.cell(row=r, column=3, value=nom_text)
        c3.font = font
        c3.alignment = wrap_align

        c4 = ws.cell(row=r, column=4, value=qn_text)
        c4.font = qn_f
        c4.alignment = wrap_align

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["D"].width = 65

    try:
        wb.save(xlsx_path)
    except PermissionError:
        print(f"  [Warning] '{xlsx_path}' is open in Excel; saving to temporary XLSX instead.")
        wb.save(xlsx_path.replace(".xlsx", "_temp.xlsx"))


def main():
    parser = argparse.ArgumentParser(description="Align Nôm and Vietnamese at woodblock level")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    alignments = build_alignment(cfg)
    save_alignment(alignments, cfg)
    print("\n✓ Section-aware balanced alignment complete!")


if __name__ == "__main__":
    main()
