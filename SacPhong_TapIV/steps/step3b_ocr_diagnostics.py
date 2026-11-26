r"""Step 3B: read-only quality diagnostics for the merged Step 3A OCR drafts.

Run from your HanNom_Project_SacPhong project root:

    .\.venv\Scripts\python.exe -m steps.step3b_ocr_diagnostics --dry-run
    .\.venv\Scripts\python.exe -m steps.step3b_ocr_diagnostics --limit 5
    .\.venv\Scripts\python.exe -m steps.step3b_ocr_diagnostics --all

Requires the existing steps.step2d_phonetic_check module and the two
QuocNgu_SinoNom dictionaries already used by the project.

This is NOT sentence alignment and NOT the teacher's 95% accuracy score.
It never changes OCR files, dictionaries, source manifests, or reviewed data.
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
BOOK_DEFAULT = ROOT / "output" / "Sac_phong_Tap_IV"
DOC_RE = re.compile(r"^AUTO\.\d{3}\.\d{2}$")
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
HAN_RANGES = ((0x3400, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF))

SUMMARY_FIELDS = [
    "doc_id", "pdf_page", "step3a_status", "source_pair_status",
    "phien_am_source", "han_characters", "phien_am_syllables", "dich_nghia_words",
    "length_difference_pct", "main_dictionary_compatible", "supplementary_only_pairs",
    "dictionary_incompatible", "han_only", "phien_am_only",
    "main_dictionary_diagnostic_pct", "combined_dictionary_diagnostic_pct",
    "priority", "flags", "eligible_for_alignment_experiment", "verified",
    "ready_for_training",
]
REVIEW_FIELDS = SUMMARY_FIELDS + ["next_action"]
ISSUE_FIELDS = [
    "doc_id", "alignment_position", "han_character", "phien_am_syllable",
    "dictionary_status", "main_dictionary_readings", "supplementary_dictionary_readings",
    "verified_against_image",
]


def book_path() -> Path:
    config = ROOT / "config.yaml"
    if config.is_file():
        try:
            import yaml
            data = yaml.safe_load(config.read_text(encoding="utf-8-sig")) or {}
            if isinstance(data, dict):
                return ROOT / str(data.get("output_dir", "output")) / str(
                    data.get("book_name", "Sac_phong_Tap_IV")
                )
        except (ImportError, OSError, ValueError, TypeError):
            pass
    return BOOK_DEFAULT


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def choose_source(folder: Path, explicit: str) -> Path:
    if explicit:
        run = Path(explicit).expanduser().resolve()
        if not (run / "document_drafts.jsonl").is_file():
            raise FileNotFoundError(f"Not a Step 3A run: {run}")
        return run
    if not folder.is_dir():
        raise FileNotFoundError(f"Step 3A output folder not found: {folder}")
    candidates = [p for p in folder.glob("run_*") if (p / "document_drafts.jsonl").is_file()]
    if not candidates:
        raise FileNotFoundError("No Step 3A document_drafts.jsonl found")
    # Prefer the most complete run (291 documents) rather than a newer 4-ID test.
    def rank(path: Path) -> tuple[int, str]:
        with (path / "document_drafts.jsonl").open(encoding="utf-8-sig") as file:
            count = sum(bool(s.strip()) for s in file)
        return count, path.name
    return max(candidates, key=rank)


def load_merged_rows(file: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with file.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {number} is not a JSON object")
            doc_id = str(record.get("doc_id", ""))
            if not DOC_RE.fullmatch(doc_id):
                raise ValueError(f"Bad document ID at line {number}: {doc_id!r}")
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID in source: {doc_id}")
            seen.add(doc_id)
            rows.append(record)
    return sorted(rows, key=lambda x: x["doc_id"])


def han_chars(text: str) -> list[str]:
    return [c for c in text if any(start <= ord(c) <= stop for start, stop in HAN_RANGES)]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def repeated_fragment(text: str) -> bool:
    condensed = re.sub(r"\s+", "", text)
    if "□" * 8 in condensed:
        return True
    return bool(re.search(r"(.{2,8})\1{5,}", condensed))


def clean_source_flags(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(f).strip() for f in value if str(f).strip()}


def score_record(record: dict, main: dict, extra: dict) -> tuple[dict, dict, list[dict]]:
    doc_id = record["doc_id"]
    han_text = str(record.get("han", "") or "")
    reading_text = str(record.get("phien_am", "") or "")
    trans_text = str(record.get("dich_nghia", "") or "")
    chinese = han_chars(han_text)
    syllables = words(reading_text)
    translation = words(trans_text)
    flags = clean_source_flags(record.get("flags", []))

    if record.get("source_pair_status") not in ("", None, "PROVISIONAL_PAIR_UNVERIFIED"):
        flags.add("SOURCE_PAIR_UNRESOLVED")
    if record.get("status") == "BLOCKED_SOURCE_PAIRING":
        flags.add("SOURCE_PAIR_UNRESOLVED")
    if not chinese:
        flags.add("MISSING_HAN")
    if not syllables:
        flags.add("MISSING_PHIEN_AM")
    if not translation:
        flags.add("MISSING_DICH_NGHIA")
    if len(syllables) > 0 and len(syllables) < 15:
        flags.add("PHIEN_AM_TOO_SHORT")
    if len(translation) > 0 and len(translation) < 15:
        flags.add("DICH_NGHIA_TOO_SHORT")
    if repeated_fragment(han_text):
        flags.add("HAN_REPETITION_PATTERN")
    if "□" in han_text:
        flags.add("UNREADABLE_HAN_MARKER")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in han_text + reading_text + trans_text):
        flags.add("CONTROL_CHARACTER_IN_TEXT")
    if record.get("phien_am_source") in ("Step2K_v2", "Step2K_v3"):
        flags.add("RECOVERED_READING_NEEDS_IMAGE_REVIEW")

    max_count = max(len(chinese), len(syllables))
    count_gap = round(abs(len(chinese) - len(syllables)) * 100 / max_count, 1) if max_count else None
    if count_gap is not None and count_gap > 15:
        flags.add("HAN_PHIEN_COUNT_GAP_GT_15PCT")

    stats = Counter()
    issues = []
    pct_main = None
    pct_combined = None
    if chinese and syllables:
        # Edit-distance alignment tolerates OCR insertion/deletion without
        # pretending its matches are image-verified character truth.
        alignment = sequence_alignment(chinese, syllables, main)
        for index, (char, syllable, match_type) in enumerate(alignment, 1):
            status = match_type.upper()
            if match_type == "incompatible" and normalize_reading(syllable) in extra.get(char, set()):
                status = "SUPPLEMENTARY_ONLY"
            stats[status] += 1
            if status == "COMPATIBLE":
                continue
            issues.append({
                "doc_id": doc_id,
                "alignment_position": index,
                "han_character": char,
                "phien_am_syllable": syllable,
                "dictionary_status": status,
                "main_dictionary_readings": " | ".join(sorted(main.get(char, set()))[:12]) if char else "",
                "supplementary_dictionary_readings": " | ".join(sorted(extra.get(char, set()))[:12]) if char else "",
                "verified_against_image": "NO",
            })
        denominator = max(len(chinese), len(syllables))
        pct_main = round(100 * stats["COMPATIBLE"] / denominator, 1)
        pct_combined = round(100 * (stats["COMPATIBLE"] + stats["SUPPLEMENTARY_ONLY"]) / denominator, 1)
        if pct_main < 55:
            flags.add("LOW_DICTIONARY_COMPATIBILITY_DIAGNOSTIC")
        if stats["HAN_ONLY"] + stats["PHONETIC_ONLY"] > max(5, 0.1 * denominator):
            flags.add("MANY_ALIGNMENT_GAPS")

    # Only an experimental permit; never certifies model output.
    blocked = ("SOURCE_PAIR_UNRESOLVED" in flags)
    complete = bool(chinese and syllables and translation)
    allowed = bool(complete and not blocked)
    critical = blocked or (not complete) or ("HAN_REPETITION_PATTERN" in flags)
    serious = any(name in flags for name in (
        "HAN_REPETITION_DETECTED", "HAN_PHIEN_COUNT_GAP_GT_15PCT",
        "LOW_DICTIONARY_COMPATIBILITY_DIAGNOSTIC", "MANY_ALIGNMENT_GAPS",
        "HAN_SOURCE_REPETITION_DETECTED", "CONTROL_CHARACTER_IN_TEXT"
    ))
    if blocked:
        priority = "BLOCKED"
        action = "Resolve whether the decree image belongs to this printed Vietnamese text; do not auto-align."
    elif critical:
        priority = "CRITICAL"
        action = "Inspect source crop and re-run or correct the missing/repetitive OCR before alignment."
    elif serious:
        priority = "HIGH"
        action = "Check original image, printed Phiên âm and suspicious character-level positions."
    elif "RECOVERED_READING_NEEDS_IMAGE_REVIEW" in flags or (pct_main is not None and pct_main < 80):
        priority = "REVIEW"
        action = "Verify sampled image and recovered reading; dictionary score is not accuracy."
    else:
        priority = "SAMPLE_QA"
        action = "Include in manually checked quality sample; cannot label as verified automatically."

    summary = {
        "doc_id": doc_id,
        "pdf_page": record.get("pdf_page", ""),
        "step3a_status": record.get("status", ""),
        "source_pair_status": record.get("source_pair_status", ""),
        "phien_am_source": record.get("phien_am_source", ""),
        "han_characters": len(chinese),
        "phien_am_syllables": len(syllables),
        "dich_nghia_words": len(translation),
        "length_difference_pct": count_gap if count_gap is not None else "",
        "main_dictionary_compatible": stats["COMPATIBLE"] if pct_main is not None else "",
        "supplementary_only_pairs": stats["SUPPLEMENTARY_ONLY"] if pct_main is not None else "",
        "dictionary_incompatible": stats["INCOMPATIBLE"] if pct_main is not None else "",
        "han_only": stats["HAN_ONLY"] if pct_main is not None else "",
        "phien_am_only": stats["PHONETIC_ONLY"] if pct_main is not None else "",
        "main_dictionary_diagnostic_pct": pct_main if pct_main is not None else "",
        "combined_dictionary_diagnostic_pct": pct_combined if pct_combined is not None else "",
        "priority": priority,
        "flags": "; ".join(sorted(flags)),
        "eligible_for_alignment_experiment": "YES" if allowed else "NO",
        "verified": "NO",
        "ready_for_training": "NO",
    }
    return summary, {**summary, "next_action": action}, issues


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3B: merged OCR diagnostics, no text changes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=5, help="Default five-document test")
    parser.add_argument("--all", action="store_true", help="Audit all merged documents")
    parser.add_argument("--ids", nargs="+", default=[], help="Specific AUTO.xxx.xx IDs")
    parser.add_argument("--source", default="", help="Optional exact Step 3A run folder")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    book = book_path()
    source = choose_source(book / "03_clean_merge_draft", args.source)
    rows = load_merged_rows(source / "document_drafts.jsonl")
    expected = read_json(source / "summary.json").get("manifest_target", len(rows))
    if args.ids:
        available = {r["doc_id"] for r in rows}
        unknown = sorted(set(args.ids) - available)
        if unknown:
            parser.error("Unknown document IDs: " + ", ".join(unknown))
        selected = [r for r in rows if r["doc_id"] in set(args.ids)]
    else:
        selected = rows if args.all else rows[:args.limit]
    print("STEP 3B - OCR DRAFT QUALITY SCREEN (NOT GROUND TRUTH)")
    print("Step 3A source:", source)
    print("Documents in source:", len(rows), "/ target:", expected)
    print("Selected:", len(selected))
    if args.dry_run:
        print("DRY RUN: no dictionaries loaded, no output written.")
        return
    if not selected:
        raise SystemExit("No merged OCR drafts selected")
    dictionaries = ROOT / "dictionaries"
    main_file = find_dictionary(dictionaries, "QuocNgu_SinoNom_TongHop3.xlsx")
    extra_file = find_dictionary(dictionaries, "QuocNgu_SinoNom_Dic.xlsx", required=False)
    print("Loading main dictionary:", main_file.name)
    main_map, _, entries = read_reading_dictionary(main_file)
    if extra_file:
        extra_map, _, extra_rows = read_reading_dictionary(extra_file)
    else:
        extra_map, extra_rows = {}, 0
    print(f"Dictionary entries: main={entries:,}, supplemental={extra_rows:,}")

    summaries, reviews, issue_rows = [], [], []
    for idx, record in enumerate(selected, 1):
        summary, review, issues = score_record(record, main_map, extra_map)
        summaries.append(summary)
        reviews.append(review)
        issue_rows.extend(issues)
        if idx % 50 == 0 or idx == len(selected):
            print(f"Audited {idx}/{len(selected)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = book / "03_ocr_qa_draft" / f"run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "qa_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(output / "review_queue.csv", REVIEW_FIELDS, [r for r in reviews if r["priority"] != "SAMPLE_QA"])
    write_csv(output / "character_issues.csv", ISSUE_FIELDS, issue_rows)
    priority_counts = dict(Counter(r["priority"] for r in summaries))
    flag_counts = Counter(f.strip() for r in summaries for f in r["flags"].split(";") if f.strip())
    ranked = [r["main_dictionary_diagnostic_pct"] for r in summaries if r["main_dictionary_diagnostic_pct"] != ""]
    ranked = sorted(ranked)
    median = ranked[len(ranked)//2] if ranked else None
    report = {
        "stage": "Step 3B",
        "source_step3a": str(source),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records_total_in_step3a_run": len(rows),
        "records_selected": len(selected),
        "manifest_target": expected,
        "priority_counts": priority_counts,
        "top_flags": flag_counts.most_common(20),
        "review_queue_count": sum(r["priority"] != "SAMPLE_QA" for r in summaries),
        "character_issue_rows": len(issue_rows),
        "median_main_dictionary_diagnostic_pct": median,
        "verified_transcriptions": 0,
        "ready_for_training": False,
        "teacher_95_percent_metric_measured": False,
        "warnings": [
            "Dictionary-compatible alignment is a noisy OCR diagnostic, not transcription accuracy.",
            "A high dictionary match does not prove that Chinese text matches the source image.",
            "Printed Phiên âm is OCR text and may be incorrect or include editorial notes.",
            "All earlier Step 1/2/3A source files are unchanged.",
            "No sentence-level alignment or automatic character substitution performed.",
        ],
    }
    (output / "audit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Report:", output)
    print("Priorities:", priority_counts)
    print("Review queue:", report["review_queue_count"])
    print("Chinese/Vietnamese source text UNCHANGED. Every record UNVERIFIED.")


if __name__ == "__main__":
    main()
