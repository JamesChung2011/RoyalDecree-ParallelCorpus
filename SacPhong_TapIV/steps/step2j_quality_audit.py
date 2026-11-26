r"""Step 2J: read-only quality diagnostics for Step 2I Chinese / Vietnamese OCR.

Usage (run from project root):
    .\.venv\Scripts\python.exe -m steps.step2j_quality_audit --dry-run
    .\.venv\Scripts\python.exe -m steps.step2j_quality_audit --pilot
    .\.venv\Scripts\python.exe -m steps.step2j_quality_audit

Never corrects source text, never assigns VERIFIED status, never calculates the
teacher's evaluation metric. Dictionary-compatible alignment is diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from steps.step2d_phonetic_check import (
    find_dictionary,
    normalize_reading,
    read_reading_dictionary,
    sequence_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
STEP2I = BOOK / "02_ocr_tho" / "step2i_batch_draft"
OUTPUT = BOOK / "02_ocr_tho" / "step2j_quality_draft"
DICTS = ROOT / "dictionaries"
DOC_ID_RE = re.compile(r"^AUTO\.\d{3}\.\d{2}$")
SYLLABLE_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
CJK_START_END = ((0x3400, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF))
SUMMARY_FIELDS = [
    "doc_id", "pdf_page", "vi_status", "han_status", "han_model",
    "han_characters", "phien_am_syllables", "dich_nghia_words",
    "compatible_diagnostic_pairs", "incompatible_pairs", "han_only", "phien_am_only",
    "dictionary_compatibility_diagnostic_pct", "character_syllable_ratio",
    "priority", "flags", "verified", "ready_for_training"
]
REVIEW_FIELDS = SUMMARY_FIELDS + ["recommendation"]
CHARACTER_FIELDS = [
    "doc_id", "position", "han_ocr_character", "phien_am_ocr_syllable",
    "dictionary_status", "possible_readings_in_main_dictionary", "verified_against_image"
]


def is_han(character: str) -> bool:
    return any(lo <= ord(character) <= hi for lo, hi in CJK_START_END)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError("JSON must be an object")
    return obj


def select_input_run(path: str) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Step 2I input folder not found: {candidate}")
        return candidate
    if not STEP2I.exists():
        raise FileNotFoundError(f"No Step 2I results folder found: {STEP2I}")
    candidates = sorted(
        (p for p in STEP2I.iterdir() if p.is_dir() and (p / "source_manifest.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not candidates:
        raise FileNotFoundError("No Step 2I run with source_manifest.json found")
    return candidates[0]


def scan_records(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    records = {}
    for path in sorted(folder.glob("AUTO.*.json")):
        if DOC_ID_RE.fullmatch(path.stem):
            records[path.stem] = path
    return records


def evaluate_one(doc_id: str, viet_file: Path | None, han_file: Path | None, dictionary: dict):
    flags = set()
    character_rows = []
    vi_status = "NOT_GENERATED"
    han_status = "NOT_GENERATED"
    han_model = ""
    pdf_page = ""
    han_text = ""
    phien_text = ""
    dich_text = ""
    vi_flags = []
    han_flags = []

    if viet_file is None:
        flags.add("VI_PENDING")
    else:
        try:
            vi_data = load_json(viet_file)
            vi_result = vi_data.get("result", {})
            if not isinstance(vi_result, dict):
                raise ValueError("Missing result object")
            vi_status = str(vi_result.get("status", "UNKNOWN"))
            pdf_page = vi_data.get("pdf_page", "")
            phien_text = str(vi_result.get("phien_am", ""))
            dich_text = str(vi_result.get("dich_nghia", ""))
            vi_flags = list(vi_result.get("flags", []))
            if vi_status not in ("OCR_DRAFT_UNVERIFIED", "OCR_NEEDS_REVIEW"):
                flags.add("VI_ENGINE_ERROR")
            if any("MISSING" in f for f in vi_flags):
                flags.add("VI_MISSING_HEADING")
            if len(phien_text.strip()) < 30:
                flags.add("PHIEN_AM_MISSING_OR_SHORT")
            if len(dich_text.strip()) < 30:
                flags.add("DICH_NGHIA_MISSING_OR_SHORT")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            vi_status = "CORRUPT_RECORD"
            flags.add("VI_CORRUPT_RECORD")
            vi_flags.append(str(exc))

    if han_file is None:
        flags.add("HAN_PENDING")
    else:
        try:
            han_data = load_json(han_file)
            han_result = han_data.get("result", {})
            if not isinstance(han_result, dict):
                raise ValueError("Missing result object")
            han_status = str(han_result.get("status", "UNKNOWN"))
            han_model = str(han_result.get("model", ""))
            pdf_page = han_data.get("pdf_page", pdf_page)
            han_text = str(han_result.get("chinese_candidate", ""))
            han_flags = list(han_result.get("flags", []))
            if han_status not in ("OCR_DRAFT_UNVERIFIED", "OCR_NEEDS_REVIEW"):
                flags.add("HAN_ENGINE_ERROR")
            if any("REPETITION" in x for x in han_flags):
                flags.add("HAN_REPETITION_SUSPECTED")
            if han_text.count("□") > 0:
                flags.add("HAN_UNREADABLE_MARKERS")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            han_status = "CORRUPT_RECORD"
            flags.add("HAN_CORRUPT_RECORD")
            han_flags.append(str(exc))

    chars = [char for char in han_text if is_han(char)]
    words = SYLLABLE_RE.findall(phien_text)
    chinese_count, phonetic_count = len(chars), len(words)
    dich_count = len(SYLLABLE_RE.findall(dich_text))
    ratio = chinese_count / phonetic_count if phonetic_count else None
    compatible, incompatible, han_only, phonetic_only = (0, 0, 0, 0)
    diagnostic_percentage = None

    if viet_file and han_file:
        if not chars:
            flags.add("HAN_EMPTY")
        if not words:
            flags.add("PHIEN_AM_EMPTY")
        if chars and words:
            aligned = sequence_alignment(chars, words, dictionary)
            counts = Counter(state for _, _, state in aligned)
            compatible = counts["compatible"]
            incompatible = counts["incompatible"]
            han_only = counts["han_only"]
            phonetic_only = counts["phonetic_only"]
            diagnostic_percentage = round(100 * compatible / max(chinese_count, phonetic_count), 1)
            if abs(chinese_count - phonetic_count) / max(chinese_count, phonetic_count) > .15:
                flags.add("CHARACTER_COUNT_GAP_OVER_15PCT")
            if diagnostic_percentage < 65.0:
                flags.add("LOW_DICTIONARY_COMPATIBILITY")
            for index, (char, syllable, status) in enumerate(aligned, start=1):
                if status == "compatible":
                    continue
                character_rows.append({
                    "doc_id": doc_id,
                    "position": index,
                    "han_ocr_character": char,
                    "phien_am_ocr_syllable": syllable,
                    "dictionary_status": status.upper(),
                    "possible_readings_in_main_dictionary": " | ".join(sorted(dictionary.get(char, set()))) if char else "",
                    "verified_against_image": "NO",
                })

    # A saved OCR draft is not verified. Every completed record still requires a
    # sampled image-based evaluation before inclusion in an official corpus.
    has_both = viet_file is not None and han_file is not None
    if not has_both:
        priority = "PENDING"
        recommendation = "Wait for missing Step 2I OCR output; no need to review this entry yet"
    elif flags:
        priority = "HIGH"
        recommendation = "Review crop, Phiên âm OCR and Chinese characters against the source image"
    else:
        priority = "REVIEW_SAMPLE"
        recommendation = "Requires image-based QA sampling; NEVER automatically accepted as verified"

    row = {
        "doc_id": doc_id,
        "pdf_page": pdf_page,
        "vi_status": vi_status,
        "han_status": han_status,
        "han_model": han_model,
        "han_characters": chinese_count,
        "phien_am_syllables": phonetic_count,
        "dich_nghia_words": dich_count,
        "compatible_diagnostic_pairs": compatible if diagnostic_percentage is not None else "",
        "incompatible_pairs": incompatible if diagnostic_percentage is not None else "",
        "han_only": han_only if diagnostic_percentage is not None else "",
        "phien_am_only": phonetic_only if diagnostic_percentage is not None else "",
        "dictionary_compatibility_diagnostic_pct": diagnostic_percentage if diagnostic_percentage is not None else "",
        "character_syllable_ratio": round(ratio, 3) if ratio is not None else "",
        "priority": priority,
        "flags": ";".join(sorted(flags)),
        "verified": "NO",
        "ready_for_training": "NO",
    }
    return row, {**row, "recommendation": recommendation}, character_rows


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Step2J quality-control diagnostics (read-only)")
    parser.add_argument("--source", default="", help="Exact Step2I output run folder, optional")
    parser.add_argument("--pilot", action="store_true", help="Check only documents with both OCR results (up to 5)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect input files without reading dictionaries or writing outputs")
    args = parser.parse_args()

    source = select_input_run(args.source)
    vi_files = scan_records(source / "vi")
    han_files = scan_records(source / "han")
    ids = sorted(set(vi_files) | set(han_files))
    if args.pilot:
        ids = [doc_id for doc_id in ids if doc_id in vi_files and doc_id in han_files][:5]
    source_meta = load_json(source / "source_manifest.json") if (source / "source_manifest.json").is_file() else {}
    expected = source_meta.get("manifest_count")

    print("STEP 2J - OCR QUALITY AUDIT (DIAGNOSTIC ONLY)")
    print("Input Step 2I run:", source)
    print("Vietnamese OCR records:", len(vi_files))
    print("Chinese OCR records:", len(han_files))
    print("Manifest target:", expected)
    print("Documents selected:", len(ids))
    if args.dry_run:
        print("DRY RUN COMPLETE: no dictionary loaded; no output written.")
        return
    if not ids:
        raise SystemExit("No Step 2I JSON files found in vi/ or han/.")

    main_dictionary = find_dictionary(DICTS, "QuocNgu_SinoNom_TongHop3.xlsx")
    print("Loading main dictionary:", main_dictionary.name)
    mapping, _, entries = read_reading_dictionary(main_dictionary)
    print("Dictionary entries loaded:", entries)

    summary_rows = []
    review_rows = []
    character_rows = []
    for doc_id in ids:
        row, review, characters = evaluate_one(doc_id, vi_files.get(doc_id), han_files.get(doc_id), mapping)
        summary_rows.append(row)
        review_rows.append(review)
        character_rows.extend(characters)
        if row["priority"] != "PENDING":
            print(f"{doc_id}: Han={row['han_characters']}, Phiên âm={row['phien_am_syllables']}, "
                  f"dictionary diagnostic={row['dictionary_compatibility_diagnostic_pct']}%, priority={row['priority']}")

    report_counts = dict(Counter(row["priority"] for row in summary_rows))
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    dest = OUTPUT / stamp
    dest.mkdir(parents=True, exist_ok=False)
    write_csv(dest / "quality_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(dest / "review_queue.csv", REVIEW_FIELDS,
              sorted(review_rows, key=lambda r: ({"HIGH": 0, "REVIEW_SAMPLE": 1, "PENDING": 2}.get(r["priority"], 3), r["doc_id"])))
    write_csv(dest / "character_diagnostics.csv", CHARACTER_FIELDS, character_rows)
    report = {
        "stage": "Step 2J", "source_step2i": str(source), "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_source_records": expected, "available_vi": len(vi_files), "available_han": len(han_files),
        "documents_in_report": len(ids), "priority_counts": report_counts,
        "verified_transcriptions": 0, "teacher_95_percent_metric_measured": False,
        "warning": "Unverified OCR and dictionary compatibility are not transcription accuracy. No data corrected."
    }
    (dest / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Output:", dest)
    print("Priority counts:", report_counts)
    print("NO input OCR files changed. All diagnostics are UNVERIFIED.")


if __name__ == "__main__":
    main()
