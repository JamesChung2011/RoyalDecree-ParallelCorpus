r"""Step 4C: cautious three-way SENTENCE CANDIDATE proposals for Sac phong.

This code does not claim semantic alignment, transcription correctness, or a score
against the teacher's evaluation rubric. It projects Hán OCR characters onto
Phiên âm punctuation spans using the Step 4A dictionary-alignment DIAGNOSTICS,
and pairs Phiên âm / Dịch nghĩa *only when their candidate-boundary counts match*.
The resulting pairs are ordered suggestions for review, never verified data.

From the project root (Windows):
    .\.venv\Scripts\python.exe -m steps.step4c_candidate_triplets --dry-run
    .\.venv\Scripts\python.exe -m steps.step4c_candidate_triplets --limit 5
    .\.venv\Scripts\python.exe -m steps.step4c_candidate_triplets --all

Step 4A and Step 4B outputs are read-only. No Ollama or OCR services are used.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
OUTROOT = BOOK / "04_sentence_triplets_draft"
TOKEN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
ID_PATTERN = re.compile(r"^AUTO\.\d{3}\.\d{2}$")
HAN_RANGES = ((0x3400, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF))
TYPE_SENT = "VI_SENTENCE_BOUNDARY_CANDIDATE"
TYPE_NOTE = "EDITORIAL_NOTE_NOT_DECREE_SENTENCE"

CANDIDATE_FIELDS = [
    "sent_id", "doc_id", "doc_type", "han", "phien_am", "dich_nghia", "ghi_chu",
]
DETAIL_FIELDS = [
    "sent_id", "doc_id", "pdf_page", "phien_am_segment_id", "dich_nghia_segment_id",
    "han_characters", "phien_am_syllables", "dictionary_compatible", "dictionary_incompatible",
    "phonetic_only", "han_only", "dictionary_compatible_pct_diagnostic", "qa_priority_source",
    "flags", "pairing_method", "verified", "ready_for_training",
]
DOCUMENT_FIELDS = [
    "doc_id", "pdf_page", "status", "phien_am_sentence_candidates",
    "dich_nghia_sentence_candidates", "proposed_sentence_triplets", "unpaired_rows",
    "han_characters", "phien_am_syllables", "qa_priority_source", "flags", "verified",
]
UNPAIRED_FIELDS = [
    "doc_id", "pdf_page", "segment_id", "text_type", "segment_kind", "segment_text", "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_han(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in HAN_RANGES)


def choose_run(base: Path, required: list[str], explicit: str, desired_count: int | None = None) -> Path:
    if explicit:
        run = Path(explicit).expanduser().resolve()
        for name in required:
            if not (run / name).is_file():
                raise FileNotFoundError(f"Missing {name} in {run}")
        return run
    candidates = []
    for run in base.glob("run_*"):
        if all((run / name).is_file() for name in required):
            src = read_csv(run / required[0])
            identifiers = len({row.get("doc_id", "") for row in src})
            rank = (identifiers, run.name)
            if desired_count is not None:
                rank = (identifiers == desired_count, identifiers, run.name)
            candidates.append((rank, run))
    if not candidates:
        raise FileNotFoundError("No completed input run found in " + str(base))
    return max(candidates, key=lambda item: item[0])[1]


def validate_unique(rows: list[dict], key: str, label: str) -> None:
    ids = [r.get(key, "") for r in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError(f"Duplicate/missing {key} in {label}")


def count_tokens(text: str) -> int:
    return len(TOKEN.findall(text))


def group_by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[row["doc_id"]].append(row)
    return dict(result)


def get_segments(rows: list[dict], name: str, kind: str | None = None) -> list[dict]:
    filtered = [x for x in rows if x["text_type"] == name and (kind is None or x["segment_kind"] == kind)]
    return sorted(filtered, key=lambda r: int(r["segment_index"]))


def reconstruct_token_sequence(rows: list[dict], name: str) -> list[str]:
    return [word for seg in get_segments(rows, name) for word in TOKEN.findall(seg["segment_text"])]


def derive_han_to_phien_map(diagnostics: list[dict], han_length: int, reading_length: int):
    """Map each original Hán OCR character index (1-based) to an OCR reading index.

    Hán-only chars are associated with an adjacent reading index, but flagged;
    this is solely for proposing spans, not a linguistic correction.
    """
    seen_han = {}
    seen_reading = set()
    status_at_han = {}
    entries = []
    for row in diagnostics:
        status = row["dictionary_status"].upper()
        hi = int(row["han_index"]) if row["han_index"] else None
        pi = int(row["phien_am_index"]) if row["phien_am_index"] else None
        if hi is not None:
            if hi in seen_han or not 1 <= hi <= han_length:
                raise ValueError("Invalid or repeated Hán diagnostic index")
            seen_han[hi] = pi
            status_at_han[hi] = status
        if pi is not None:
            if pi in seen_reading or not 1 <= pi <= reading_length:
                raise ValueError("Invalid or repeated Phiên âm diagnostic index")
            seen_reading.add(pi)
        entries.append((hi, pi, status))
    if set(seen_han) != set(range(1, han_length + 1)) or seen_reading != set(range(1, reading_length + 1)):
        raise ValueError("Step 4A character mapping does not cover the source strings")
    mapped_indices = sorted(i for i, p in seen_han.items() if p is not None)
    if not mapped_indices:
        raise ValueError("No Hán-to-reading alignment anchors")
    mapping = {}
    for index in range(1, han_length + 1):
        phonetic = seen_han[index]
        if phonetic is None:
            pos = bisect.bisect_left(mapped_indices, index)
            previous = mapped_indices[pos - 1] if pos else None
            following = mapped_indices[pos] if pos < len(mapped_indices) else None
            anchor = previous if previous is not None else following
            phonetic = seen_han[anchor]
        mapping[index] = phonetic
    return mapping, entries


def unpaired_for_document(doc_id: str, page: str, segments: list[dict], reason: str) -> list[dict]:
    return [{
        "doc_id": doc_id, "pdf_page": page, "segment_id": row["segment_id"],
        "text_type": row["text_type"], "segment_kind": row["segment_kind"],
        "segment_text": row["segment_text"], "reason": reason,
    } for row in segments]


def inspect_doc(trip: dict, segments: list[dict], diag: list[dict], boundary: dict):
    """Return document summary, candidate rows, detail rows and unpaired rows."""
    doc_id = trip["doc_id"]
    page = trip.get("pdf_page", "")
    all_ph = get_segments(segments, "PHIEN_AM")
    all_dn = get_segments(segments, "DICH_NGHIA")
    ph = [x for x in all_ph if x["segment_kind"] == TYPE_SENT]
    dn = [x for x in all_dn if x["segment_kind"] == TYPE_SENT]
    source_qa = trip.get("qa_priority", "")
    chars = [c for c in trip["han"] if is_han(c)]
    syllables = TOKEN.findall(trip["phien_am"])
    flags = []
    review_flags = boundary.get("flags", "")
    flags.extend(f for f in review_flags.split(";") if f)
    statuses = [x.get("dictionary_status", "") for x in diag]
    compatible_count = statuses.count("COMPATIBLE")
    overall_pct = 100.0 * compatible_count / max(len(chars), len(syllables), 1)
    if overall_pct < 35:
        flags.append("VERY_LOW_DICTIONARY_COMPATIBILITY")
    elif overall_pct < 55:
        flags.append("LOW_DICTIONARY_COMPATIBILITY")
    if source_qa.upper() in {"HIGH", "CRITICAL", "BLOCKED"}:
        flags.append("SOURCE_QUALITY_" + source_qa.upper())
    if any(s["segment_kind"] != TYPE_SENT for s in all_ph + all_dn):
        flags.append("UNPAIRED_NOTES_OR_UNCLOSED_FRAGMENTS")
    if len(ph) != len(dn):
        flags.append("VI_BOUNDARY_COUNTS_DIFFER")
    if len(chars) < 10 or len(chars) < len(syllables) * 0.5:
        flags.append("HAN_OCR_UNUSABLY_INCOMPLETE")
    if len(chars) > len(syllables) * 1.6:
        flags.append("HAN_OCR_LENGTH_SUSPICIOUS")
    if "REPEATED_HAN_OCR_TEXT" in flags:
        flags.append("HAN_OCR_REPETITION_FLAG")
    if overall_pct < 25:
        flags.append("HAN_DICT_COMPATIBILITY_CRITICALLY_LOW")
    if source_qa.upper() in {"CRITICAL", "BLOCKED"}:
        flags.append("SOURCE_QA_BLOCKS_AUTO_PROPOSAL")
    if reconstruct_token_sequence(segments, "PHIEN_AM") != syllables:
        flags.append("READING_TOKEN_SEQUENCE_MISMATCH")
    if reconstruct_token_sequence(segments, "DICH_NGHIA") != TOKEN.findall(trip["dich_nghia"]):
        flags.append("TRANSLATION_TOKEN_SEQUENCE_MISMATCH")
    if not ph or not dn:
        flags.append("NO_PUNCTUATED_SENTENCE_CANDIDATES")
    flags = list(dict.fromkeys(flags))
    hard_blocks = {
        "VI_BOUNDARY_COUNTS_DIFFER", "HAN_OCR_UNUSABLY_INCOMPLETE", "HAN_OCR_LENGTH_SUSPICIOUS",
        "HAN_OCR_REPETITION_FLAG", "HAN_DICT_COMPATIBILITY_CRITICALLY_LOW", "SOURCE_QA_BLOCKS_AUTO_PROPOSAL",
        "READING_TOKEN_SEQUENCE_MISMATCH",
        "TRANSLATION_TOKEN_SEQUENCE_MISMATCH", "NO_PUNCTUATED_SENTENCE_CANDIDATES",
    }
    block_reasons = sorted(set(flags) & hard_blocks)
    mapped_han = {}
    if not block_reasons:
        try:
            mapped_han, _ = derive_han_to_phien_map(diag, len(chars), len(syllables))
        except ValueError as error:
            flags.append("STEP4A_DIAGNOSTIC_MAPPING_MISMATCH")
            block_reasons.append("STEP4A_DIAGNOSTIC_MAPPING_MISMATCH: " + str(error))

    result = []
    details = []
    unpaired = []
    if block_reasons:
        status = "WITHHELD_FROM_SENTENCE_PROPOSALS"
        unpaired.extend(unpaired_for_document(doc_id, page, segments, ";".join(block_reasons)))
    else:
        status = "ORDER_ONLY_SENTENCE_PROPOSALS_UNVERIFIED"
        # Gather each reading segment's inclusive index interval, preserving exact order.
        curr_syl = 0
        range_map = {}
        for seg in all_ph:
            num = count_tokens(seg["segment_text"])
            range_map[seg["segment_id"]] = (curr_syl + 1, curr_syl + num)
            curr_syl += num
        assert curr_syl == len(syllables)
        for seq, (p, d) in enumerate(zip(ph, dn), start=1):
            start, stop = range_map[p["segment_id"]]
            han_idx = [idx for idx in range(1, len(chars) + 1) if start <= mapped_han[idx] <= stop]
            han_text = "".join(chars[idx - 1] for idx in han_idx)
            related = [row for row in diag if row["phien_am_index"] and start <= int(row["phien_am_index"]) <= stop]
            counts = Counter(row["dictionary_status"] for row in related)
            comp = counts.get("COMPATIBLE", 0)
            incompat = counts.get("INCOMPATIBLE", 0)
            phon_only = counts.get("PHONETIC_ONLY", 0)
            han_only = sum(1 for idx in han_idx if not any(row["han_index"] == str(idx) and row["phien_am_index"] for row in related))
            syl_count = count_tokens(p["segment_text"])
            score = 100.0 * comp / max(syl_count, len(han_idx), 1)
            local_flags = []
            if not han_text:
                local_flags.append("NO_MAPPED_HAN_CHARS")
            if len(han_idx) < syl_count * 0.5:
                local_flags.append("LOCAL_HAN_COVERAGE_LOW")
            if score < 55:
                local_flags.append("LOCAL_DICTIONARY_MATCH_LOW")
            # All candidates are weakly inferred translations, even with equal counts.
            local_flags.extend(flags)
            local_flags = list(dict.fromkeys(local_flags))
            candidate_id = f"CAND_{doc_id}_{seq:03d}"
            notes = [
                "UNVERIFIED_CANDIDATE_ONLY", "VI_PUNCTUATION_BOUNDARY_PROPOSAL",
                "HAN_PROJECTED_FROM_STEP4A_DIAGNOSTICS", "PHIEN_AM_AND_DICH_NGHIA_PAIRED_BY_ORDER_NOT_SEMANTICS",
                "DO_NOT_USE_AS_GOLD_OR_TRAINING_DATA",
            ] + local_flags
            if not han_text or len(han_idx) < syl_count * 0.45 or score < 20:
                local_reason = "NO_MAPPED_HAN_CHARS" if not han_text else (
                    "LOCAL_HAN_COVERAGE_LOW" if len(han_idx) < syl_count * 0.45
                    else "LOCAL_DICTIONARY_COMPATIBILITY_CRITICALLY_LOW"
                )
                unpaired.extend(unpaired_for_document(doc_id, page, [p, d], local_reason))
            else:
                result.append({
                    "sent_id": candidate_id, "doc_id": doc_id,
                    "doc_type": trip.get("doc_type", "Sắc phong (provisional)"),
                    "han": han_text, "phien_am": p["segment_text"], "dich_nghia": d["segment_text"],
                    "ghi_chu": "; ".join(dict.fromkeys(notes)),
                })
                details.append({
                    "sent_id": candidate_id, "doc_id": doc_id, "pdf_page": page,
                    "phien_am_segment_id": p["segment_id"], "dich_nghia_segment_id": d["segment_id"],
                    "han_characters": len(han_idx), "phien_am_syllables": syl_count,
                    "dictionary_compatible": comp, "dictionary_incompatible": incompat,
                    "phonetic_only": phon_only, "han_only": han_only,
                    "dictionary_compatible_pct_diagnostic": f"{score:.1f}",
                    "qa_priority_source": source_qa, "flags": ";".join(local_flags),
                    "pairing_method": "ORDER_ONLY_NO_SEMANTIC_MATCH", "verified": "NO", "ready_for_training": "NO",
                })
        leftover = [s for s in segments if s["text_type"] in {"PHIEN_AM", "DICH_NGHIA"} and s["segment_kind"] != TYPE_SENT]
        unpaired.extend(unpaired_for_document(doc_id, page, leftover, "NOT_A_VERIFIED_SENTENCE"))
        if len(result) != len(ph):
            status = "PARTIAL_ORDER_PROPOSALS_UNVERIFIED"
    doc_row = {
        "doc_id": doc_id, "pdf_page": page, "status": status,
        "phien_am_sentence_candidates": len(ph), "dich_nghia_sentence_candidates": len(dn),
        "proposed_sentence_triplets": len(result), "unpaired_rows": len(unpaired),
        "han_characters": len(chars), "phien_am_syllables": len(syllables),
        "qa_priority_source": source_qa, "flags": ";".join(flags), "verified": "NO",
    }
    return doc_row, result, details, unpaired


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Step 4C: unverified order-based three-way SENTENCE CANDIDATES")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Process all documents")
    selection.add_argument("--limit", type=int, default=5, help="Test first N documents (default 5)")
    selection.add_argument("--ids", nargs="+", help="Select specific document IDs")
    parser.add_argument("--dry-run", action="store_true", help="Read inputs; do not write files")
    parser.add_argument("--source-4a", default="", help="Exact Step 4A run directory")
    parser.add_argument("--source-4b", default="", help="Exact Step 4B run directory")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        dir4b = choose_run(BOOK / "04_sentence_boundaries_draft", ["boundary_proposals.csv", "document_boundary_summary.csv", "boundary_report.json"], args.source_4b)
        seg_rows = read_csv(dir4b / "boundary_proposals.csv")
        summarize = read_csv(dir4b / "document_boundary_summary.csv")
        validate_unique(summarize, "doc_id", "Step 4B document_boundary_summary.csv")
        by_doc_segs = group_by_doc(seg_rows)
        needed = {r["doc_id"] for r in summarize}
        dir4a = choose_run(BOOK / "04_alignment_draft", ["document_triplets_draft.csv", "han_phien_character_diagnostics.csv"], args.source_4a, desired_count=len(needed))
        trip_rows = read_csv(dir4a / "document_triplets_draft.csv")
        validate_unique(trip_rows, "doc_id", "Step 4A document_triplets_draft.csv")
        trip_by_doc = {r["doc_id"]: r for r in trip_rows}
        if not needed.issubset(trip_by_doc):
            raise ValueError("Step 4A is missing some Step 4B document IDs: " + ",".join(sorted(needed - trip_by_doc)[:10]))
        missing_segments = needed - set(by_doc_segs)
        if missing_segments:
            raise ValueError("Step 4B boundary segments missing for: " + ",".join(sorted(missing_segments)[:10]))
        bad_ids = [r["doc_id"] for r in trip_rows if not ID_PATTERN.fullmatch(r["doc_id"])]
        if bad_ids:
            raise ValueError("Unexpected document IDs: " + str(bad_ids[:5]))
        diag_rows = read_csv(dir4a / "han_phien_character_diagnostics.csv")
        by_doc_diag = group_by_doc(diag_rows)
        if args.ids:
            wanted = set(args.ids)
            unknown = wanted - needed
            if unknown:
                raise ValueError("Unknown IDs: " + ", ".join(sorted(unknown)))
            chosen = sorted(wanted)
        else:
            chosen = sorted(needed) if args.all else sorted(needed)[:args.limit]
        print("STEP 4C — UNVERIFIED SENTENCE CANDIDATE ALIGNMENT")
        print("Step 4A:", dir4a)
        print("Step 4B:", dir4b)
        print(f"Documents available: {len(needed)}; selected: {len(chosen)}")
        print("Method: phonetic character projection + equal-count Vietnamese order pairing")
        if args.dry_run:
            print("DRY RUN COMPLETE — no files written or services called.")
            return 0
        output_root = OUTROOT / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
        output_root.mkdir(parents=True, exist_ok=False)
        summaries, candidates, detail_rows, unmatched = [], [], [], []
        for i, doc_id in enumerate(chosen, 1):
            row = trip_by_doc[doc_id]
            boundary = next(x for x in summarize if x["doc_id"] == doc_id)
            doc_row, candidate_rows, details, left = inspect_doc(row, by_doc_segs[doc_id], by_doc_diag.get(doc_id, []), boundary)
            summaries.append(doc_row)
            candidates.extend(candidate_rows)
            detail_rows.extend(details)
            unmatched.extend(left)
            if i % 50 == 0 or i == len(chosen):
                print(f"Processed {i}/{len(chosen)}")
        write_csv(output_root / "sentence_candidates_UNVERIFIED.csv", CANDIDATE_FIELDS, candidates)
        write_csv(output_root / "candidate_diagnostics.csv", DETAIL_FIELDS, detail_rows)
        write_csv(output_root / "document_pairing_status.csv", DOCUMENT_FIELDS, summaries)
        write_csv(output_root / "unpaired_segments.csv", UNPAIRED_FIELDS, unmatched)
        report = {
            "stage": "Step 4C", "created_at": datetime.now(timezone.utc).isoformat(),
            "source_4a": str(dir4a), "source_4b": str(dir4b),
            "documents_available": len(needed), "documents_processed": len(chosen),
            "documents_with_order_proposals": sum(int(r["proposed_sentence_triplets"]) > 0 for r in summaries),
            "documents_withheld_from_sentence_proposals": sum(int(r["proposed_sentence_triplets"]) == 0 for r in summaries),
            "order_only_sentence_candidates": len(candidates),
            "unpaired_segment_rows": len(unmatched),
            "status_counts": dict(Counter(r["status"] for r in summaries)),
            "verified_sentence_pairs": 0, "semantic_translation_alignment_performed": False,
            "teacher_95_percent_metric_measured": False, "ready_for_training": False,
            "warning": (
                "ALL OUTPUTS UNVERIFIED. Equal sentence counts and dictionary-assisted character projection "
                "do not establish semantic Hán–Phiên âm–Dịch nghĩa alignment. "
                "Never submit sentence_candidates_UNVERIFIED.csv as verified gold data."
            ),
        }
        (output_root / "alignment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Candidates:", len(candidates), "Unpaired segments:", len(unmatched))
        print("Withheld documents:", report["documents_withheld_from_sentence_proposals"])
        print("Output:", output_root)
        print("NO VERIFIED SENTENCE PAIRS OR SEMANTIC ALIGNMENT GENERATED.")
        return 0
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        print("INPUT/ALIGNMENT ERROR:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
