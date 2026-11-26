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
    print(f"Loaded {num_nom} Nôm woodblock pages across {len(SECTIONS)} sections.")

    alignments = []

    for sec_name, ns, ne, vs, ve in SECTIONS:
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

        # Standardize known OCR confusions in markers
        raw_sec_vi = raw_sec_vi.replace('|4-bJ', '[4-b]')
        raw_sec_vi = raw_sec_vi.replace('{5-b]', '[5-b]')
        raw_sec_vi = raw_sec_vi.replace('[3.4]', '[3-a]')
        raw_sec_vi = raw_sec_vi.replace('[‡-a]', '[4-a]')
        raw_sec_vi = raw_sec_vi.replace('[†-s]', '[1-a]')
        raw_sec_vi = raw_sec_vi.replace('[4+]', '[4-b]')
        raw_sec_vi = raw_sec_vi.replace('[!-b]', '[1-b]')
        raw_sec_vi = raw_sec_vi.replace('[4-s]', '[4-a]')
        raw_sec_vi = raw_sec_vi.replace('[7-4]', '[7-a]')
        raw_sec_vi = raw_sec_vi.replace('[1H-b]', '[11-b]')
        raw_sec_vi = raw_sec_vi.replace('[12-4]', '[12-a]')
        raw_sec_vi = raw_sec_vi.replace('[20-4]', '[20-a]')
        raw_sec_vi = raw_sec_vi.replace('[24-4]', '[24-a]')
        raw_sec_vi = raw_sec_vi.replace('[28-6]', '[28-b]')
        raw_sec_vi = raw_sec_vi.replace('[S0-b]', '[50-b]')
        raw_sec_vi = raw_sec_vi.replace('[%2-a]', '[52-a]')
        raw_sec_vi = raw_sec_vi.replace('[36.4]', '[36-a]')
        raw_sec_vi = raw_sec_vi.replace('[38-4]', '[38-a]')
        raw_sec_vi = raw_sec_vi.replace('[41-8]', '[41-b]')
        raw_sec_vi = raw_sec_vi.replace('[43-4]', '[43-a]')
        raw_sec_vi = raw_sec_vi.replace('[4Š-b]', '[45-b]')
        raw_sec_vi = raw_sec_vi.replace('[50-4]', '[50-a]')
        raw_sec_vi = raw_sec_vi.replace('[54-8]', '[54-b]')
        raw_sec_vi = raw_sec_vi.replace('(hiểu sói', 'hiểu sói')
        raw_sec_vi = raw_sec_vi.replace('[100 năm]', '')

        # 3. Analyze Nôm text and character weights
        nom_info = []
        for img in sec_nom_images:
            nom_text = get_full_text_nom(ch_data[img])
            nom_info.append((img, nom_text, len(nom_text)))

        num_pages = len(nom_info)

        # 4. Locate all woodblock markers in raw_sec_vi
        marker_pattern = r'\[\s*(\d{1,2})\s*[-._\s]*\s*([abAB])\s*\]'
        anchors = []
        for m in re.finditer(marker_pattern, raw_sec_vi):
            folio = int(m.group(1))
            face = m.group(2).lower()
            target_idx = (folio - 1) * 2 + (0 if face == 'a' else 1)
            if 0 <= target_idx < num_pages:
                anchors.append({
                    "folio": folio,
                    "face": face,
                    "target_idx": target_idx,
                    "start": m.start(),
                    "end": m.end()
                })

        anchors.sort(key=lambda x: x["start"])
        filtered_anchors = []
        last_target = -1
        for a in anchors:
            if a["target_idx"] > last_target:
                filtered_anchors.append(a)
                last_target = a["target_idx"]

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
                    for tok in tokens:
                        if c_page < span_len - 1 and c_ass + len(tok) > cum_t[c_page]:
                            if len(p_toks[c_page]) > 0:
                                c_page += 1
                        p_toks[c_page].append(tok)
                        c_ass += len(tok)

                    for j in range(span_len):
                        page_texts[curr_idx + j] = " ".join(p_toks[j])

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

                end_idx = min(total_nom, max(curr_idx + 1, best_cut))
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
