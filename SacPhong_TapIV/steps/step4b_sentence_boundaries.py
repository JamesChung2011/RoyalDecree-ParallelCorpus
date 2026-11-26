r"""Step 4B: conservative sentence-boundary proposals for the Sắc phong corpus.

This is NOT automatic semantic three-way sentence alignment and NOT ground truth.
It never modifies the input OCR, Step 3 or Step 4A files.

Usage (from the project root):
    .\.venv\Scripts\python.exe -m steps.step4b_sentence_boundaries --dry-run
    .\.venv\Scripts\python.exe -m steps.step4b_sentence_boundaries --limit 5
    .\.venv\Scripts\python.exe -m steps.step4b_sentence_boundaries --all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "output" / "Sac_phong_Tap_IV" / "04_alignment_draft"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "Sac_phong_Tap_IV" / "04_sentence_boundaries_draft"

HAN_PUNCTUATION = set("。！？!?；")
VI_PUNCTUATION = set(".!?")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_input(explicit: str | None) -> tuple[Path, list[dict[str, str]]]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        rows = read_csv(path)
        return path, rows
    candidates = list(DEFAULT_INPUT_ROOT.glob("run_*/document_triplets_draft.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No Step 4A document_triplets_draft.csv found under "
            + str(DEFAULT_INPUT_ROOT)
        )
    choices = [(len(read_csv(p)), p.parent.name, p) for p in candidates]
    _, _, path = max(choices, key=lambda x: (x[0], x[1]))
    return path, read_csv(path)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def char_is_han(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
        or 0x30000 <= code <= 0x323AF
    )


def han_count(text: str) -> int:
    return sum(char_is_han(char) for char in text)


def reading_syllables(text: str) -> int:
    # A diagnostic only: OCR punctuation, bracketed notes, and line breaks
    # make this count different from a verified philological transcription.
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def repetitive_han(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if re.search(r"(.{2,18}?)\1{2,}", compact):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    freq = Counter(lines)
    return any(len(line) >= 4 and occurrences >= 3 for line, occurrences in freq.items())


def split_han_lines(text: str) -> list[dict[str, str]]:
    """OCR's newline is not necessarily a sentence boundary."""
    segments = []
    for line in normalize(text).splitlines():
        line = line.strip()
        if not line:
            continue
        # Distinguish an OCR layout line from explicit punctuation-based splits.
        start = 0
        for position, char in enumerate(line):
            if char in HAN_PUNCTUATION:
                part = line[start:position + 1].strip()
                if part:
                    segments.append({
                        "text": part,
                        "segment_kind": "HAN_PUNCTUATED_FRAGMENT_NOT_VERIFIED",
                        "boundary_basis": "SOURCE_PUNCTUATION",
                    })
                start = position + 1
        tail = line[start:].strip()
        if tail:
            segments.append({
                "text": tail,
                "segment_kind": "HAN_OCR_LINE_NOT_SENTENCE",
                "boundary_basis": "OCR_LAYOUT_LINE",
            })
    return segments


def is_editorial_note(line: str) -> bool:
    normalized = line.strip().lstrip("*‘’'\"“” ").casefold()
    return normalized.startswith("nay là ") or normalized.startswith("chú thích:")


def separate_notes(text: str) -> tuple[str, list[str]]:
    body = []
    notes = []
    for raw in normalize(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if is_editorial_note(line):
            notes.append(line)
        else:
            body.append(line)
    return " ".join(body), notes


def split_vi_sentences(text: str) -> list[dict[str, str]]:
    body, notes = separate_notes(text)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return [
            {"text": note, "segment_kind": "EDITORIAL_NOTE_NOT_DECREE_SENTENCE", "boundary_basis": "EDITORIAL_NOTE"}
            for note in notes
        ]

    # Split only at explicit full stops, exclamation marks and question marks.
    # OCR punctuation can be wrong; these are candidate boundaries only.
    # Never split simply because the scan wrapped to a new line.
    result = []
    current = []
    for index, char in enumerate(body):
        current.append(char)
        if char not in VI_PUNCTUATION:
            continue
        next_char = body[index + 1] if index + 1 < len(body) else ""
        prev_char = body[index - 1] if index else ""
        # Avoid decimal numbers and simple repeated abbreviation dots, e.g. v.v.
        if char == "." and prev_char.isdigit() and next_char.isdigit():
            continue
        if char == "." and next_char.isalpha() and not next_char.isspace():
            continue
        if next_char and not next_char.isspace() and next_char not in ')]}”"':
            continue
        # Closing quote / bracket after punctuation belongs to this sentence.
        # For simplicity do not consume it here: flush on following whitespace.
        if not next_char or next_char.isspace():
            candidate = "".join(current).strip()
            if candidate:
                result.append({
                    "text": candidate,
                    "segment_kind": "VI_SENTENCE_BOUNDARY_CANDIDATE",
                    "boundary_basis": "OCR_TERMINAL_PUNCTUATION",
                })
            current = []
    tail = "".join(current).strip()
    if tail:
        result.append({
            "text": tail,
            "segment_kind": "VI_UNCLOSED_FRAGMENT_NOT_SENTENCE",
            "boundary_basis": "NO_TERMINAL_PUNCTUATION",
        })
    for note in notes:
        result.append({
            "text": note,
            "segment_kind": "EDITORIAL_NOTE_NOT_DECREE_SENTENCE",
            "boundary_basis": "EDITORIAL_NOTE",
        })
    return result


def summary_flags(record: dict[str, str], han: str, reading: str, translation: str, segments: dict[str, list]) -> list[str]:
    flags = []
    hc = han_count(han)
    rc = reading_syllables(reading)
    if hc < 10:
        flags.append("CRITICAL_HAN_TEXT_UNUSABLY_SHORT")
    if not rc:
        flags.append("MISSING_READING")
    if not translation.strip():
        flags.append("MISSING_TRANSLATION")
    if rc and hc < rc * 0.5:
        flags.append("HAN_READING_RATIO_BELOW_0_5")
    if rc and hc > rc * 1.6:
        flags.append("HAN_READING_RATIO_ABOVE_1_6")
    if repetitive_han(han):
        flags.append("REPEATED_HAN_OCR_TEXT")
    if any(len(s["text"]) >= 600 for name in ("PHIEN_AM", "DICH_NGHIA") for s in segments[name]):
        flags.append("EXCESSIVELY_LONG_VI_SEGMENT")
    if record.get("qa_priority", "").upper() in {"CRITICAL", "HIGH", "BLOCKED"}:
        flags.append("PREVIOUS_QA_" + record["qa_priority"].upper())
    if record.get("verified", "").strip().upper() == "YES":
        flags.append("UNEXPECTED_VERIFIED_INPUT_REQUIRES_REVIEW")
    return flags


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 4B: OCR-based sentence boundary proposals, NOT semantic alignment.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process all Step 4A document drafts")
    group.add_argument("--limit", type=int, default=5, help="How many documents to examine (default: 5)")
    group.add_argument("--ids", nargs="+", help="Specific document IDs to examine")
    parser.add_argument("--dry-run", action="store_true", help="Inspect input only; do not write outputs")
    parser.add_argument("--input", help="Optional explicit Step 4A document_triplets_draft.csv path")
    args = parser.parse_args(argv)

    try:
        input_path, records = find_input(args.input)
    except (FileNotFoundError, OSError) as exc:
        print("INPUT ERROR:", exc, file=sys.stderr)
        return 2

    required = {"doc_id", "pdf_page", "han", "phien_am", "dich_nghia"}
    if not records or not required.issubset(records[0]):
        print(f"INPUT ERROR: missing required Step 4A CSV columns {required}", file=sys.stderr)
        return 2

    if args.ids:
        wanted = set(args.ids)
        selected = [r for r in records if r["doc_id"] in wanted]
        missing = sorted(wanted - {r["doc_id"] for r in selected})
        if missing:
            print("IDs not found in eligible Step 4A triplets:", ", ".join(missing), file=sys.stderr)
    elif args.all:
        selected = records
    else:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        selected = records[:args.limit]

    print("STEP 4B — CONSERVATIVE OCR BOUNDARY PROPOSALS (UNVERIFIED)")
    print("Input:", input_path)
    print(f"Step 4A triplets: {len(records)}; selected: {len(selected)}")
    print("No document/translation pairing or semantic matching is performed.")
    if args.dry_run:
        print("DRY RUN COMPLETE — no output files written.")
        return 0

    if not selected:
        print("No documents selected; no output written.")
        return 2

    seen = set()
    for record in selected:
        id_ = record["doc_id"]
        if id_ in seen:
            print(f"INPUT ERROR: duplicate doc_id {id_}", file=sys.stderr)
            return 2
        seen.add(id_)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DEFAULT_OUTPUT_ROOT / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    segment_rows = []
    summary_rows = []
    review_rows = []
    count_by_type = Counter()
    document_flags = Counter()

    for seq, record in enumerate(selected, 1):
        id_ = record["doc_id"]
        streams = {
            "HAN": normalize(record["han"]),
            "PHIEN_AM": normalize(record["phien_am"]),
            "DICH_NGHIA": normalize(record["dich_nghia"]),
        }
        segments = {
            "HAN": split_han_lines(streams["HAN"]),
            "PHIEN_AM": split_vi_sentences(streams["PHIEN_AM"]),
            "DICH_NGHIA": split_vi_sentences(streams["DICH_NGHIA"]),
        }
        flags = summary_flags(record, streams["HAN"], streams["PHIEN_AM"], streams["DICH_NGHIA"], segments)
        for flag in flags:
            document_flags[flag] += 1
        for text_type, pieces in segments.items():
            count_by_type[text_type] += len(pieces)
            for position, piece in enumerate(pieces, 1):
                segment_rows.append({
                    "segment_id": f"{id_}__{text_type}__{position:03d}",
                    "doc_id": id_,
                    "pdf_page": record.get("pdf_page", ""),
                    "text_type": text_type,
                    "segment_index": position,
                    "segment_kind": piece["segment_kind"],
                    "boundary_basis": piece["boundary_basis"],
                    "segment_text": piece["text"],
                    "qa_priority_source": record.get("qa_priority", ""),
                    "verified": "NO",
                    "ready_for_training": "NO",
                })
        rcount = reading_syllables(streams["PHIEN_AM"])
        hcount = han_count(streams["HAN"])
        row = {
            "doc_id": id_,
            "pdf_page": record.get("pdf_page", ""),
            "han_characters": hcount,
            "phien_am_syllables_approx": rcount,
            "han_to_reading_ratio": f"{hcount/rcount:.3f}" if rcount else "",
            "han_ocr_fragments": len(segments["HAN"]),
            "han_layout_note": "OCR_LINES_ARE_NOT_SENTENCES",
            "phien_am_boundary_candidates": sum(s["segment_kind"] == "VI_SENTENCE_BOUNDARY_CANDIDATE" for s in segments["PHIEN_AM"]),
            "dich_nghia_boundary_candidates": sum(s["segment_kind"] == "VI_SENTENCE_BOUNDARY_CANDIDATE" for s in segments["DICH_NGHIA"]),
            "phien_am_unclosed_fragments": sum(s["segment_kind"] == "VI_UNCLOSED_FRAGMENT_NOT_SENTENCE" for s in segments["PHIEN_AM"]),
            "dich_nghia_unclosed_fragments": sum(s["segment_kind"] == "VI_UNCLOSED_FRAGMENT_NOT_SENTENCE" for s in segments["DICH_NGHIA"]),
            "editorial_notes_detected": sum(s["segment_kind"] == "EDITORIAL_NOTE_NOT_DECREE_SENTENCE" for n in segments for s in segments[n]),
            "qa_priority_source": record.get("qa_priority", ""),
            "flags": ";".join(flags),
            "verified": "NO",
            "ready_for_training": "NO",
        }
        summary_rows.append(row)
        if flags:
            review_rows.append(row)
        if seq % 50 == 0 or seq == len(selected):
            print(f"Processed {seq}/{len(selected)}")

    segment_fields = ["segment_id", "doc_id", "pdf_page", "text_type", "segment_index", "segment_kind", "boundary_basis", "segment_text", "qa_priority_source", "verified", "ready_for_training"]
    summary_fields = ["doc_id", "pdf_page", "han_characters", "phien_am_syllables_approx", "han_to_reading_ratio", "han_ocr_fragments", "han_layout_note", "phien_am_boundary_candidates", "dich_nghia_boundary_candidates", "phien_am_unclosed_fragments", "dich_nghia_unclosed_fragments", "editorial_notes_detected", "qa_priority_source", "flags", "verified", "ready_for_training"]

    write_csv(output_dir / "boundary_proposals.csv", segment_rows, segment_fields)
    write_csv(output_dir / "document_boundary_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "boundary_review_queue.csv", review_rows, summary_fields)

    excluded_path = input_path.parent / "excluded_documents.csv"
    excluded_count = 0
    if excluded_path.exists():
        excluded = read_csv(excluded_path)
        excluded_count = len(excluded)
        if excluded:
            write_csv(output_dir / "excluded_step4a_documents.csv", excluded, list(excluded[0]))

    report = {
        "stage": "Step 4B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_step4a_csv": str(input_path),
        "eligible_document_triplets_in_source": len(records),
        "documents_processed": len(selected),
        "source_step4a_exclusions_copied": excluded_count,
        "segment_rows_by_type": dict(count_by_type),
        "review_flag_counts": dict(document_flags),
        "documents_with_actionable_review_flags": len(review_rows),
        "han_ocr_lines_are_not_sentences": True,
        "verified_sentence_pairs": 0,
        "semantic_three_way_alignment_performed": False,
        "ready_for_training": False,
        "warnings": [
            "Hán OCR newlines are layout fragments, not established sentence boundaries.",
            "Vietnamese boundaries are punctuation proposals, not verified sentences.",
            "No Hán–Phiên âm–Dịch nghĩa sentence-level matching has been performed.",
            "No original data or Step 4A files were modified.",
        ],
    }
    (output_dir / "boundary_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Output:", output_dir)
    print("Segment rows:", len(segment_rows), dict(count_by_type))
    print("Actionable review flags:", len(review_rows), "of", len(selected))
    print("ALL RESULTS UNVERIFIED; ZERO SENTENCE PAIRS CREATED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
