"""Step 2D - Audit Hán OCR against PRINTED Sino-Vietnamese readings.

This is a NON-DESTRUCTIVE diagnostic. It does not revise the source OCR,
produce official alignment pairs, or claim to verify the Chinese text.

Run from the project root:
    .venv\\Scripts\\python.exe -m steps.step2d_phonetic_check

Only Python standard-library modules are required. All XLSX input files are
read directly as Office Open XML; this script does not alter the dictionaries.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
TONE = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
NOTE_PATTERN = re.compile(r"^\s*[\(（].*[\)）]\s*$")
PHIEN_PATTERN = re.compile(r"^\s*phi[eê]n\s+âm\s*:\s*", re.IGNORECASE)
DICH_PATTERN = re.compile(r"^\s*d[iị]ch\s+ngh[iĩ]a\s*:\s*", re.IGNORECASE)


def normalize_reading(text: str) -> str:
    """Normalize capitalization and tone-mark placement (hoà == hòa).

    IMPORTANT: remove NO lexical diacritics (â/ă/ê/ô/ơ/ư/đ remain distinct).
    """
    decomposed = unicodedata.normalize("NFD", str(text).strip().casefold())
    plain = []
    found_tones = []
    for symbol in decomposed:
        if symbol in TONE:
            found_tones.append(symbol)
        else:
            plain.append(symbol)
    return unicodedata.normalize("NFC", "".join(plain) + "".join(sorted(found_tones)))


def iter_spreadsheet_rows(path: Path):
    """Stream first worksheet of a conventional Excel XLSX dictionary.

    Supports shared strings and inline strings, avoids loading 100k rows as a
    workbook, and requires no third-party packages on the student's PC.
    """
    with zipfile.ZipFile(path) as source:
        names = set(source.namelist())
        strings = []
        if "xl/sharedStrings.xml" in names:
            with source.open("xl/sharedStrings.xml") as stream:
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag == XML_NS + "si":
                        strings.append("".join(t.text or "" for t in element.iter(XML_NS + "t")))
                        element.clear()
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            raise RuntimeError(f"Missing {sheet_name} in {path}")
        with source.open(sheet_name) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != XML_NS + "row":
                    continue
                row = {}
                for cell in element.findall(XML_NS + "c"):
                    address = cell.get("r", "")
                    match = re.match(r"[A-Z]+", address)
                    if not match:
                        continue
                    value_tag = cell.find(XML_NS + "v")
                    value = (value_tag.text or "") if value_tag is not None else ""
                    if cell.get("t") == "s":
                        value = strings[int(value)] if value else ""
                    elif cell.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(XML_NS + "t"))
                    row[match.group()] = value.strip()
                yield row
                element.clear()


def find_dictionary(folder: Path, suffix: str, required=True) -> Path | None:
    matches = sorted((p for p in folder.glob("*.xlsx") if p.name.casefold().endswith(suffix.casefold())))
    if not matches:
        if required:
            raise FileNotFoundError(
                f"Missing dictionary ending in {suffix!r} inside {folder}. "
                "Copy the provided XLSX dictionaries into this folder first."
            )
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple copies of {suffix} found in {folder}; keep just one: {matches}")
    return matches[0]


def read_reading_dictionary(path: Path):
    char_to_readings = defaultdict(set)
    reading_to_chars = defaultdict(set)
    total = 0
    for row in iter_spreadsheet_rows(path):
        spelling, glyph = row.get("A", ""), row.get("B", "")
        if spelling == "QuocNgu" and glyph == "SinoNom":
            continue
        if len(glyph) != 1 or not spelling:
            continue
        syllable = normalize_reading(spelling)
        char_to_readings[glyph].add(syllable)
        reading_to_chars[syllable].add(glyph)
        total += 1
    return char_to_readings, reading_to_chars, total


def read_visual_similarity(path: Path | None):
    similars = {}
    if path is None:
        return similars
    for row in iter_spreadsheet_rows(path):
        glyph, repr_list = row.get("A", ""), row.get("B", "")
        if len(glyph) != 1 or not repr_list.startswith("["):
            continue
        try:
            candidate_list = ast.literal_eval(repr_list)
        except (ValueError, SyntaxError):
            continue
        if isinstance(candidate_list, list):
            similars[glyph] = [v for v in candidate_list if isinstance(v, str) and len(v) == 1]
    return similars


def load_ppocr_labels(file_path: Path):
    records = {}
    if not file_path.is_file():
        raise FileNotFoundError(f"OCR file missing: {file_path}")
    with file_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                image, boxes_json = line.split("\t", 1)
                boxes = json.loads(boxes_json)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid PPOCRLabel line {line_number} in {file_path}: {error}") from error
            file_stem = Path(image.replace("\\", "/")).stem
            doc_id = re.sub(r"_\d+$", "", file_stem)
            records.setdefault(doc_id, []).append({"image": image, "boxes": boxes})
    return records


def box_coords(box):
    points = box.get("points", [])
    if len(points) != 4:
        return (0.0, 0.0)
    try:
        center_x = sum(float(point[0]) for point in points) / len(points)
        center_y = sum(float(point[1]) for point in points) / len(points)
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)
    return (center_x, center_y)


def extract_han(records):
    combined = []
    for record in records:
        boxed = record["boxes"]
        # Chinese documents are normally vertical, right-to-left.
        ordered = sorted(boxed, key=lambda b: (-box_coords(b)[0], box_coords(b)[1]))
        combined.extend(str(box.get("transcription", "")) for box in ordered)
    # Retain only Chinese ideographs; OCR Latin/noise is excluded but never corrected.
    result = []
    for character in "".join(combined):
        name = unicodedata.name(character, "")
        if name.startswith("CJK UNIFIED IDEOGRAPH") or name.startswith("CJK COMPATIBILITY IDEOGRAPH"):
            result.append(character)
    return "".join(result)


def extract_phien_am(records):
    fragments = []
    skipped_notes = []
    state = "before"
    for record in records:
        boxes = sorted(record["boxes"], key=lambda b: (box_coords(b)[1], box_coords(b)[0]))
        for box in boxes:
            raw = " ".join(str(box.get("transcription", "")).split())
            if not raw:
                continue
            if PHIEN_PATTERN.match(raw):
                state = "phien"
                raw = PHIEN_PATTERN.sub("", raw)
            if DICH_PATTERN.match(raw):
                state = "done"
                break
            if state == "phien" and raw:
                if NOTE_PATTERN.match(raw):
                    skipped_notes.append(raw)
                else:
                    fragments.append(raw)
        if state == "done":
            break
    text = " ".join(fragments)
    syllables = WORD_PATTERN.findall(text)
    return text, syllables, skipped_notes, state == "done"


def sequence_alignment(chars, syllables, readings):
    """Edit-distance alignment for DIAGNOSIS only; not sentence alignment.

    0 = dictionary-compatible, 1 = incompatible substitution or a gap.
    Since missing OCR characters can shift the sequences, EVERY result still
    requires image-based verification.
    """
    rows = len(chars)
    cols = len(syllables)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    path = [[None] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = i
        path[i][0] = "han_only"
    for j in range(1, cols + 1):
        dp[0][j] = j
        path[0][j] = "phonetic_only"
    for i, char in enumerate(chars, 1):
        allowed = readings.get(char, set())
        for j, syllable in enumerate(syllables, 1):
            normalized = normalize_reading(syllable)
            is_compatible = normalized in allowed
            options = [
                (dp[i - 1][j - 1] + (0 if is_compatible else 1), 0, "compatible" if is_compatible else "incompatible"),
                (dp[i - 1][j] + 1, 1, "han_only"),
                (dp[i][j - 1] + 1, 2, "phonetic_only"),
            ]
            cost, _, operation = min(options)
            dp[i][j] = cost
            path[i][j] = operation
    aligned = []
    i, j = rows, cols
    while i > 0 or j > 0:
        operation = path[i][j]
        if operation in ("compatible", "incompatible"):
            aligned.append((chars[i - 1], syllables[j - 1], operation))
            i -= 1
            j -= 1
        elif operation == "han_only":
            aligned.append((chars[i - 1], "", operation))
            i -= 1
        else:
            aligned.append(("", syllables[j - 1], operation))
            j -= 1
    return list(reversed(aligned))


def suggest_similar(glyph, phonetic, similar, main_map):
    if not glyph or not phonetic:
        return []
    wanted = normalize_reading(phonetic)
    result = []
    for char in similar.get(glyph, []):
        if wanted in main_map.get(char, set()) and char not in result:
            result.append(char)
    return result[:8]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    project = Path(args.project_root).resolve()
    source = project / "output" / args.book_folder / "02_ocr_tho"
    dictionary_folder = (project / args.dictionary_folder).resolve()
    main_path = find_dictionary(dictionary_folder, "QuocNgu_SinoNom_TongHop3.xlsx")
    extra_path = find_dictionary(dictionary_folder, "QuocNgu_SinoNom_Dic.xlsx", required=False)
    similarity_path = find_dictionary(dictionary_folder, "SinoNom_Similar_Dic_v2.xlsx", required=False)

    print("Loading main Hán-Việt dictionary:", main_path.name)
    main, reverse, main_entries = read_reading_dictionary(main_path)
    extra, _, extra_entries = read_reading_dictionary(extra_path) if extra_path else ({}, {}, 0)
    similarity = read_visual_similarity(similarity_path)
    print(f"Loaded {main_entries:,} main rows, {extra_entries:,} supplementary rows, "
          f"{len(similarity):,} visual-similarity entries.")

    han_records = load_ppocr_labels(source / "han" / "Label.txt")
    viet_records = load_ppocr_labels(source / "viet" / "Label.txt")
    doc_ids = sorted(set(han_records) | set(viet_records))
    if not doc_ids:
        raise RuntimeError("No OCR records found")
    output = project / "output" / args.book_folder / "02_ocr_tho" / "phonetic_validation_draft"
    if output.exists() and not args.force:
        raise FileExistsError(f"Report folder exists: {output}. Use --force to REGENERATE reports only.")
    output.mkdir(parents=True, exist_ok=True)

    review_rows = []
    summary_rows = []
    phien_text_blocks = []
    for doc_id in doc_ids:
        han = extract_han(han_records.get(doc_id, []))
        phien_text, syllables, notes, found_translation = extract_phien_am(viet_records.get(doc_id, []))
        phien_text_blocks.append("### " + doc_id + "\n" + phien_text + "\n")
        aligned = sequence_alignment(han, syllables, main)
        counts = Counter(status for _, _, status in aligned)
        for index, (character, syllable, status) in enumerate(aligned, 1):
            normalized = normalize_reading(syllable) if syllable else ""
            supplement = (character != "" and syllable != "" and
                          normalized in extra.get(character, set()))
            suggestions = suggest_similar(character, syllable, similarity, main) if status == "incompatible" else []
            review_rows.append({
                "doc_id": doc_id,
                "alignment_position": index,
                "ocr_han_char": character,
                "printed_phien_am_ocr": syllable,
                "dictionary_status": status.upper(),
                "main_dictionary_readings": ", ".join(sorted(main.get(character, set()))) if character else "",
                "secondary_dictionary_has_pair": "YES" if supplement else "NO",
                "similar_shape_and_reading_suggestions": " ".join(suggestions),
                "verified_against_image": "NO",
                "review_notes": "",
            })
        summary_rows.append({
            "doc_id": doc_id,
            "han_ocr_characters": len(han),
            "phien_am_ocr_syllables": len(syllables),
            "compatible_diagnostic_pairs": counts["compatible"],
            "incompatible_pairs": counts["incompatible"],
            "han_only": counts["han_only"],
            "phien_am_only": counts["phonetic_only"],
            "complete_dich_nghia_heading_detected": "YES" if found_translation else "NO",
            "skipped_editorial_notes": " | ".join(notes),
            "diagnostic_status": "MISSING_OCR" if not han or not syllables else (
                "INSUFFICIENT_CHINESE_OCR" if len(han) < (len(syllables) * 0.7) else "UNVERIFIED_NEEDS_MANUAL_REVIEW"
            ),
        })
        print(f"{doc_id}: Chinese {len(han):3d} chars; phonetic {len(syllables):3d} syllables; "
              f"dictionary-compatible diagnostic pairs {counts['compatible']:3d} (NOT verified)")

    write_csv(output / "diagnostic_summary.csv", [
        "doc_id", "han_ocr_characters", "phien_am_ocr_syllables", "compatible_diagnostic_pairs",
        "incompatible_pairs", "han_only", "phien_am_only", "complete_dich_nghia_heading_detected",
        "skipped_editorial_notes", "diagnostic_status",
    ], summary_rows)
    write_csv(output / "character_review.csv", [
        "doc_id", "alignment_position", "ocr_han_char", "printed_phien_am_ocr", "dictionary_status",
        "main_dictionary_readings", "secondary_dictionary_has_pair", "similar_shape_and_reading_suggestions",
        "verified_against_image", "review_notes",
    ], review_rows)
    (output / "phien_am_ocr_draft.txt").write_text("\n".join(phien_text_blocks), encoding="utf-8")
    report = {
        "WARNING": "DIAGNOSTIC ONLY: not ground truth, not the teacher's 95% test, no automatic text correction.",
        "dictionaries": {"main": main_path.name, "supplementary": extra_path.name if extra_path else None,
                         "visual_similarity": similarity_path.name if similarity_path else None},
        "documents": summary_rows,
        "counts": {"documents": len(doc_ids), "review_rows": len(review_rows)},
    }
    (output / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nReports created (NO OCR files changed):", output)
    print("Review first: diagnostic_summary.csv, then character_review.csv")
    return output


def main():
    parser = argparse.ArgumentParser(description="Non-destructive Chinese OCR / Hán-Việt dictionary diagnostic")
    parser.add_argument("--project-root", default=".", help="Project folder containing output/")
    parser.add_argument("--book-folder", default="Sac_phong_Tap_IV", help="Book subfolder within output/")
    parser.add_argument("--dictionary-folder", default="dictionaries", help="Folder containing the three XLSX dictionaries")
    parser.add_argument("--force", action="store_true", help="Regenerate diagnostic reports (never rewrites OCR Labels)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
