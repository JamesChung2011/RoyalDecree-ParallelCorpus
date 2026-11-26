"""Step 4A — read-only DOCUMENT-LEVEL corpus candidates + character/reading diagnostics.

This is NOT sentence-level alignment, image verification, or the teacher's 95% metric.
The printed Dịch nghĩa is preserved at DOCUMENT granularity; it is not aligned
sentence-by-sentence by guessing where Chinese text should end.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m steps.step4a_document_phonetic_align --dry-run
    .\\.venv\\Scripts\\python.exe -m steps.step4a_document_phonetic_align --limit 5
    .\\.venv\\Scripts\\python.exe -m steps.step4a_document_phonetic_align --all
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
    find_dictionary, normalize_reading, read_reading_dictionary, sequence_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
ID_FORMAT = re.compile(r"^AUTO\.\d{3}\.\d{2}$")
SYLLABLES = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
HAN_RANGES = ((0x3400, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF))

TRIPLET_FIELDS = [
    "pair_id", "unit", "doc_id", "doc_type", "pdf_page", "han",
    "phien_am", "dich_nghia", "ghi_chu", "qa_priority", "phien_am_source",
    "dictionary_compatible_pct_diagnostic", "verified", "ready_for_training",
]
ALIGN_FIELDS = [
    "doc_id", "alignment_position", "han_index", "phien_am_index",
    "han_character", "phien_am_syllable", "dictionary_status", "verified",
]
EXCLUDED_FIELDS = [
    "doc_id", "pdf_page", "status", "qa_priority", "reason",
    "han_characters", "phien_am_syllables", "dich_nghia_words", "verified",
]
SUMMARY_FIELDS = [
    "doc_id", "pdf_page", "status", "qa_priority", "han_characters",
    "phien_am_syllables", "dich_nghia_words", "dictionary_compatible",
    "dictionary_incompatible", "han_only", "phien_am_only",
    "compatibility_pct_diagnostic", "notes",
]


def is_han(c: str) -> bool:
    n = ord(c)
    return any(start <= n <= stop for start, stop in HAN_RANGES)


def load_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 3A source: {path}")
    seen: set[str] = set()
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"Expected JSON object at {path.name}:{number}")
            doc_id = rec.get("doc_id")
            if not isinstance(doc_id, str) or not ID_FORMAT.fullmatch(doc_id):
                raise ValueError(f"Unexpected ID at line {number}: {doc_id!r}")
            if doc_id in seen:
                raise ValueError(f"Repeated doc_id {doc_id} in {path}")
            seen.add(doc_id)
            records.append(rec)
    return sorted(records, key=lambda row: row["doc_id"])


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_step3a(path: str) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
        if not (candidate / "document_drafts.jsonl").is_file():
            raise FileNotFoundError("--source-3a must point to a Step 3A run folder")
        return candidate
    base = BOOK / "03_clean_merge_draft"
    candidates = [p for p in base.glob("run_*") if (p / "document_drafts.jsonl").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No Step 3A run found in {base}")
    def rank(folder: Path) -> tuple[int, str]:
        with (folder / "document_drafts.jsonl").open(encoding="utf-8-sig") as source:
            n = sum(bool(line.strip()) for line in source)
        return n, folder.name
    return max(candidates, key=rank)


def choose_step3b(path: str, expected_ids: set[str]) -> Path:
    if path:
        chosen = Path(path).expanduser().resolve()
        if not (chosen / "qa_summary.csv").is_file():
            raise FileNotFoundError("--source-3b must point to a Step 3B run folder")
        return chosen
    base = BOOK / "03_ocr_qa_draft"
    candidates = [p for p in base.glob("run_*") if (p / "qa_summary.csv").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No Step 3B audit found in {base}")
    def rank(folder: Path) -> tuple[int, int, str]:
        with (folder / "qa_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            ids = {r["doc_id"] for r in csv.DictReader(stream)}
        return len(ids & expected_ids), len(ids), folder.name
    return max(candidates, key=rank)


def write_csv(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def inspect_pair(rec: dict, qa: dict | None, dictionary: dict) -> tuple[dict | None, dict | None, list[dict], dict]:
    doc_id = rec["doc_id"]
    han = str(rec.get("han") or "")
    phien_am = str(rec.get("phien_am") or "")
    dich_nghia = str(rec.get("dich_nghia") or "")
    chars = [c for c in han if is_han(c)]
    syllables = SYLLABLES.findall(phien_am)
    trans_words = SYLLABLES.findall(dich_nghia)
    priority = (qa or {}).get("priority", "NO_QA")
    flags = rec.get("flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    pairing_blocked = (
        rec.get("status") == "BLOCKED_SOURCE_PAIRING"
        or rec.get("source_pair_status") not in (None, "", "PROVISIONAL_PAIR_UNVERIFIED")
        or priority == "BLOCKED"
    )
    complete = bool(chars and syllables and trans_words)
    reason = ""
    if qa is None:
        reason = "MISSING_STEP3B_QUALITY_AUDIT"
    elif pairing_blocked:
        reason = "SOURCE_PAIRING_UNRESOLVED"
    elif not complete:
        reason = "MISSING_HAN_OR_PHIEN_AM_OR_DICH_NGHIA"
    elif rec.get("draft_alignment_eligible") is not True:
        reason = "STEP3A_INELIGIBLE_FOR_DRAFT_ALIGNMENT"
    if reason:
        excluded = {
            "doc_id": doc_id, "pdf_page": rec.get("pdf_page", ""),
            "status": rec.get("status", ""), "qa_priority": priority, "reason": reason,
            "han_characters": len(chars), "phien_am_syllables": len(syllables),
            "dich_nghia_words": len(trans_words), "verified": "NO",
        }
        summary = {
            "doc_id": doc_id, "pdf_page": rec.get("pdf_page", ""),
            "status": "EXCLUDED_FROM_PROVISIONAL_TRIPLETS", "qa_priority": priority,
            "han_characters": len(chars), "phien_am_syllables": len(syllables),
            "dich_nghia_words": len(trans_words), "dictionary_compatible": "",
            "dictionary_incompatible": "", "han_only": "", "phien_am_only": "",
            "compatibility_pct_diagnostic": "", "notes": reason,
        }
        return None, excluded, [], summary
    if max(len(chars), len(syllables)) > 1200:
        note = "DIAGNOSTIC_SKIPPED_UNUSUALLY_LONG_DOCUMENT"
        alignment = []
    else:
        alignment = sequence_alignment(chars, syllables, dictionary)
        note = ""
    stats: Counter[str] = Counter()
    indices = {"han": 0, "reading": 0}
    align_rows: list[dict] = []
    for position, (char, syl, status) in enumerate(alignment, 1):
        if char:
            indices["han"] += 1
        if syl:
            indices["reading"] += 1
        stats[status] += 1
        align_rows.append({
            "doc_id": doc_id, "alignment_position": position,
            "han_index": indices["han"] if char else "",
            "phien_am_index": indices["reading"] if syl else "",
            "han_character": char, "phien_am_syllable": syl,
            "dictionary_status": status.upper(), "verified": "NO",
        })
    denom = max(len(chars), len(syllables))
    pct = round(100.0 * stats["compatible"] / denom, 1) if alignment and denom else ""
    notes = ["DOCUMENT_LEVEL_ONLY_NOT_SENTENCE_ALIGNED", "UNVERIFIED_SOURCE_OCR"]
    if priority in {"HIGH", "CRITICAL"}:
        notes.append("QUALITY_AUDIT_" + priority)
    if rec.get("phien_am_source") in {"Step2K_v2", "Step2K_v3"}:
        notes.append("RECOVERED_PHIEN_AM_REVIEW_REQUIRED")
    if note:
        notes.append(note)
    if flags:
        notes.extend(sorted(set(str(v) for v in flags if v)))
    note_text = "; ".join(dict.fromkeys(notes))
    triple = {
        "pair_id": f"DOC_{doc_id}", "unit": "DOCUMENT", "doc_id": doc_id,
        "doc_type": "Sắc phong (provisional)", "pdf_page": rec.get("pdf_page", ""),
        "han": han, "phien_am": phien_am, "dich_nghia": dich_nghia,
        "ghi_chu": note_text, "qa_priority": priority,
        "phien_am_source": rec.get("phien_am_source", ""),
        "dictionary_compatible_pct_diagnostic": pct,
        "verified": "NO", "ready_for_training": "NO",
    }
    summary = {
        "doc_id": doc_id, "pdf_page": rec.get("pdf_page", ""),
        "status": "DOCUMENT_TRIPLET_UNVERIFIED", "qa_priority": priority,
        "han_characters": len(chars), "phien_am_syllables": len(syllables),
        "dich_nghia_words": len(trans_words), "dictionary_compatible": stats["compatible"] if alignment else "",
        "dictionary_incompatible": stats["incompatible"] if alignment else "",
        "han_only": stats["han_only"] if alignment else "",
        "phien_am_only": stats["phonetic_only"] if alignment else "",
        "compatibility_pct_diagnostic": pct, "notes": note_text,
    }
    return triple, None, align_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4A: provisional document-level alignment and character diagnostics")
    parser.add_argument("--dry-run", action="store_true", help="Report inputs; do not import dictionary or write files")
    parser.add_argument("--limit", type=int, default=5, help="Default number of documents for test")
    parser.add_argument("--all", action="store_true", help="Process all documents")
    parser.add_argument("--ids", nargs="+", default=[], help="Specific document IDs")
    parser.add_argument("--source-3a", default="", help="Optional exact Step 3A run directory")
    parser.add_argument("--source-3b", default="", help="Optional exact Step 3B run directory")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    src3a = choose_step3a(args.source_3a)
    records = load_lines(src3a / "document_drafts.jsonl")
    src3b = choose_step3b(args.source_3b, {rec["doc_id"] for rec in records})
    audit = read_csv(src3b / "qa_summary.csv")
    qa_by_id = {r["doc_id"]: r for r in audit}
    if len(qa_by_id) != len(audit):
        raise RuntimeError("Duplicate IDs in Step 3B quality report")
    if args.ids:
        requested = set(args.ids)
        unknown = requested - {rec["doc_id"] for rec in records}
        if unknown:
            parser.error("Unknown IDs: " + ", ".join(sorted(unknown)))
        selected = [r for r in records if r["doc_id"] in requested]
    else:
        selected = records if args.all else records[:args.limit]
    print("STEP 4A — DOCUMENT-LEVEL OCR ALIGNMENT PROPOSALS (NOT VERIFIED)")
    print("Source Step 3A:", src3a)
    print("Source Step 3B:", src3b)
    print(f"Available merged records: {len(records)}; QA rows: {len(audit)}; selected: {len(selected)}")
    if len(qa_by_id) < len(records):
        print("WARNING: Quality audit does not cover every source record")
    if args.dry_run:
        print("DRY RUN: no dictionary loaded, no output files generated.")
        return
    dictionary_file = find_dictionary(ROOT / "dictionaries", "QuocNgu_SinoNom_TongHop3.xlsx")
    print("Loading existing main Hán-Việt dictionary:", dictionary_file.name)
    main, _, count = read_reading_dictionary(dictionary_file)
    print("Dictionary entries:", count)
    triplets: list[dict] = []
    excluded: list[dict] = []
    alignments: list[dict] = []
    summary: list[dict] = []
    for i, rec in enumerate(selected, 1):
        triple, exclusion, align_rows, diagnostic = inspect_pair(rec, qa_by_id.get(rec["doc_id"]), main)
        if triple is not None:
            triplets.append(triple)
        if exclusion is not None:
            excluded.append(exclusion)
        alignments.extend(align_rows)
        summary.append(diagnostic)
        if i % 50 == 0 or i == len(selected):
            print(f"Processed {i}/{len(selected)}")
    timestamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    dest = BOOK / "04_alignment_draft" / timestamp
    dest.mkdir(parents=True, exist_ok=False)
    write_csv(dest / "document_triplets_draft.csv", TRIPLET_FIELDS, triplets)
    write_csv(dest / "excluded_documents.csv", EXCLUDED_FIELDS, excluded)
    write_csv(dest / "han_phien_character_diagnostics.csv", ALIGN_FIELDS, alignments)
    write_csv(dest / "alignment_summary.csv", SUMMARY_FIELDS, summary)
    report = {
        "stage": "Step 4A", "unit": "DOCUMENT_NOT_SENTENCE",
        "source_step3a": str(src3a), "source_step3b": str(src3b),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "available_documents": len(records), "selected_documents": len(selected),
        "provisional_document_triplets": len(triplets),
        "excluded_documents": len(excluded), "character_diagnostic_rows": len(alignments),
        "qa_priorities_in_selected": dict(Counter((qa_by_id.get(r["doc_id"]) or {}).get("priority", "NO_QA") for r in selected)),
        "verified_document_pairs": 0, "verified_sentence_pairs": 0,
        "teacher_95_percent_metric_measured": False,
        "ready_for_training": False,
        "warnings": [
            "One row per document, not one row per sentence. The final seven-field sentence TSV is NOT produced here.",
            "Dictionary-based character/syllable matching is diagnostic, not image-verified accuracy.",
            "The Dịch nghĩa is kept intact at document level and has NOT been matched semantically to Chinese sentences.",
            "Blocked source pairs, missing text and incomplete drafts were excluded from provisional triplets.",
            "All prior OCR, recovery, audit and source files were left unchanged.",
        ],
    }
    (dest / "alignment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Draft document triplets: {len(triplets)}; excluded: {len(excluded)}; diagnostics: {len(alignments)}")
    print("Output:", dest)
    print("NO VERIFIED SENTENCE ALIGNMENT WAS PERFORMED.")


if __name__ == "__main__":
    main()
