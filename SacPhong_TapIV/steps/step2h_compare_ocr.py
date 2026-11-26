"""Step 2H: non-destructive comparison of historical Chinese OCR candidates.

Run from the SacPhong project root:
    .venv\\Scripts\\python.exe -m steps.step2h_compare_ocr --dry-run
    .venv\\Scripts\\python.exe -m steps.step2h_compare_ocr

Uses the student's existing Step 2D dictionary reader and edit-distance
alignment. This is diagnostic ranking, NOT a verified Hán transcription,
CER score, 95% phonetic matching grade, or final corpus export.

Reads PPOCRLabel originals, GLM-OCR, Qwen (including archived runs), and
printed Phiên âm. Does NOT modify those files or invoke any AI models.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .step2d_phonetic_check import (
    extract_han, extract_phien_am, find_dictionary, load_ppocr_labels,
    normalize_reading, read_reading_dictionary, read_visual_similarity,
    sequence_alignment, suggest_similar,
)


CJK_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F), (0x30000, 0x323AF),
)
VALID_STATUSES = {"OCR_DRAFT_UNVERIFIED", "OCR_DRAFT", "DRAFT_UNVERIFIED"}
BAD_STATUSES = {"OCR_ERROR", "REPETITION_REJECTED", "EMPTY_OCR"}


def han_only(value: str) -> str:
    return "".join(
        char for char in str(value)
        if any(low <= ord(char) <= high for low, high in CJK_RANGES)
    )


def pathological_repetition(value: str) -> bool:
    """Reject obvious loops, NOT natural recurring phrases in royal decrees."""
    compact = re.sub(r"\s+", "", str(value))
    if "□" * 9 in compact:
        return True
    # Look for at least eight back-to-back repeats of a short unit.
    for size in range(1, 9):
        pattern = re.compile(r"(.{" + str(size) + r"})\1{7,}")
        if pattern.search(compact):
            return True
    return False


def get_candidate_paths(root: Path, include_archive: bool) -> list[Path]:
    base = root / "output" / "Sac_phong_Tap_IV" / "02_ocr_tho"
    containers = [base]
    if include_archive:
        containers.append(root / "_archive_OCR_experiments")
    found = []
    for container in containers:
        if not container.is_dir():
            continue
        for child in sorted(container.iterdir()):
            if not child.is_dir() or not child.name.startswith("han_"):
                continue
            if child.name in {"han", "han_region_probe"}:
                continue
            # Results from GLM and v2 Qwen (timestamped folders).
            if child.name in {"han_glm_candidates", "han_vlm_v2"}:
                found.extend(child.rglob("results.json"))
            # Old Qwen versions (one JSON per document).
            if child.name in {"han_vlm_candidates", "han_vlm_candidates_3b_backup"}:
                found.extend(child.glob("PILOT.*.json"))
    return sorted(set(path.resolve() for path in found))


def read_candidate_file(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError, OSError) as exc:
        print(f"WARNING: Could not parse candidate file {path}: {exc}")
        return []
    if not isinstance(payload, dict):
        print(f"WARNING: Invalid candidate structure: {path}")
        return []
    records = payload.get("attempts", [])
    if not isinstance(records, list):
        return []
    result = []
    for attempt in records:
        if not isinstance(attempt, dict):
            continue
        doc_id = str(attempt.get("doc_id") or payload.get("doc_id") or "").strip()
        if not doc_id:
            continue
        result.append({
            "doc_id": doc_id,
            "model": str(attempt.get("model") or payload.get("model") or "unknown"),
            "variant": str(attempt.get("variant") or "original"),
            "status": str(attempt.get("status") or "UNKNOWN"),
            "raw": str(attempt.get("raw_response") or ""),
            "text": str(attempt.get("chinese_candidate") or ""),
            "original_flags": attempt.get("flags") or [],
            "source_file": path,
            "elapsed": attempt.get("elapsed_seconds", ""),
        })
    return result


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_csv(destination: Path, rows: list[dict], columns: list[str]) -> None:
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_candidates(root: Path, include_archive: bool) -> tuple[list[dict], list[Path]]:
    base = root / "output" / "Sac_phong_Tap_IV" / "02_ocr_tho"
    han_path = base / "han" / "Label.txt"
    baseline = load_ppocr_labels(han_path)
    results = []
    for doc_id, records in sorted(baseline.items()):
        results.append({
            "doc_id": doc_id, "model": "RapidOCR_baseline", "variant": "original_ppocr_label",
            "status": "OCR_DRAFT_UNVERIFIED", "raw": "", "text": extract_han(records),
            "source_file": han_path, "original_flags": [], "elapsed": "",
        })
    files = get_candidate_paths(root, include_archive)
    for path in files:
        results.extend(read_candidate_file(path))
    # Multiple backups can contain the same candidate. Keep one copy,
    # but never combine different models or variants.
    unique = []
    seen = set()
    for record in results:
        signature = (
            record["doc_id"], record["model"], record["variant"],
            record["status"], record["text"],
        )
        if signature not in seen:
            seen.add(signature)
            unique.append(record)
    return unique, files


def evaluate(record: dict, syllables: list[str], dictionary: dict) -> tuple[dict, list[tuple]]:
    cleaned = han_only(record["text"])
    n, m = len(cleaned), len(syllables)
    status = record["status"]
    original_flags = record["original_flags"]
    if isinstance(original_flags, str):
        original_flags = [original_flags]
    flags = list(str(f) for f in original_flags)
    blocked = status in BAD_STATUSES or status not in VALID_STATUSES
    if pathological_repetition(record["raw"] or record["text"]):
        flags.append("REPETITION_DETECTED")
        blocked = True
    if n == 0:
        flags.append("NO_CHINESE_TEXT")
        blocked = True
    if m == 0:
        flags.append("NO_PHIEN_AM_OCR")
        blocked = True
    if m and n < 0.65 * m:
        flags.append("LOW_CHARACTER_COVERAGE")
    if m and n > 1.35 * m:
        flags.append("TOO_MANY_CHARACTERS")
    if "□" in record["text"]:
        flags.append("UNREADABLE_SYMBOLS")
    aligned: list[tuple] = []
    counts = Counter()
    if not blocked:
        aligned = sequence_alignment(list(cleaned), syllables, dictionary)
        counts = Counter(part[2] for part in aligned)
    compatible = counts["compatible"]
    incompatible = counts["incompatible"]
    missing = counts["phonetic_only"]
    extra = counts["han_only"]
    # Diagnostic coverage only. NOT independently established accuracy.
    score = round(100.0 * compatible / max(n, m, 1), 1) if not blocked else None
    result = {
        "doc_id": record["doc_id"],
        "model": record["model"],
        "variant": record["variant"],
        "source_file": record["source_file"],
        "source_status": status,
        "comparison_status": "REJECTED_FROM_RANKING" if blocked else "DIAGNOSTIC_ONLY",
        "han_chars": n,
        "phien_am_syllables": m,
        "compatible_pairs": compatible if not blocked else "",
        "incompatible_pairs": incompatible if not blocked else "",
        "han_only": extra if not blocked else "",
        "phien_am_only": missing if not blocked else "",
        "compatibility_diagnostic_pct": score if score is not None else "",
        "flags": "; ".join(sorted(set(flags))),
        "elapsed_seconds": record["elapsed"],
        "preview_han": cleaned[:96],
        "verified": "NO",
        "ready_for_training": "NO",
    }
    return result, aligned


def run(args: argparse.Namespace) -> Path | None:
    root = Path(args.project_root).resolve()
    base = root / "output" / "Sac_phong_Tap_IV" / "02_ocr_tho"
    viet_path = base / "viet" / "Label.txt"
    candidates, candidate_files = prepare_candidates(root, args.include_archive)
    viet = load_ppocr_labels(viet_path)
    doc_ids = set(viet) | {entry["doc_id"] for entry in candidates}
    if args.docs:
        doc_ids &= set(args.docs)
    doc_ids = sorted(doc_ids)
    if not doc_ids:
        raise RuntimeError("No pilot documents found. Confirm that Step 2 OCR output exists.")
    print("Step 2H: automatic candidate comparison (diagnostics only)")
    print(f"  Project: {root}")
    print(f"  Documents: {', '.join(doc_ids)}")
    print(f"  Candidate JSON files found: {len(candidate_files)}")
    print(f"  Total candidate records loaded: {len(candidates)}")
    if args.dry_run:
        for path in candidate_files:
            print("  Source:", safe_relative(path, root))
        print("DRY RUN: no dictionaries loaded or output created.")
        return None
    folder = root / args.dictionary_folder
    main = find_dictionary(folder, "QuocNgu_SinoNom_TongHop3.xlsx")
    extra = find_dictionary(folder, "QuocNgu_SinoNom_Dic.xlsx", required=False)
    similar_path = find_dictionary(folder, "SinoNom_Similar_Dic_v2.xlsx", required=False)
    readings, _, main_entries = read_reading_dictionary(main)
    supplemental = read_reading_dictionary(extra)[0] if extra else {}
    similarity = read_visual_similarity(similar_path)
    print(f"  Main Hán-Việt dictionary rows: {main_entries:,}")
    print(f"  Supplemental dictionary: {'YES' if extra else 'NO'}")
    print(f"  Visual similarity dictionary: {'YES' if similar_path else 'NO'}")
    by_doc = defaultdict(list)
    for rec in candidates:
        if rec["doc_id"] in doc_ids:
            by_doc[rec["doc_id"]].append(rec)
    comparisons: list[dict] = []
    summaries: list[dict] = []
    queue: list[dict] = []
    review_alignments: list[dict] = []
    for doc_id in doc_ids:
        phonetic, syllables, notes, heading_found = extract_phien_am(viet.get(doc_id, []))
        rows = []
        alns = {}
        for rec in by_doc[doc_id]:
            result, alignment = evaluate(rec, syllables, readings)
            result["source_file"] = safe_relative(rec["source_file"], root)
            result["dich_nghia_heading_found"] = "YES" if heading_found else "NO"
            comparisons.append(result)
            rows.append(result)
            alns[id(result)] = alignment
        scored = [row for row in rows if row["comparison_status"] == "DIAGNOSTIC_ONLY"]
        scored.sort(key=lambda r: (
            r["compatibility_diagnostic_pct"],
            -abs(r["han_chars"] - len(syllables)),
            r["compatible_pairs"],
        ), reverse=True)
        chosen = scored[0] if scored else None
        summary = {
            "doc_id": doc_id,
            "phien_am_ocr_syllables": len(syllables),
            "candidate_count": len(rows),
            "scored_candidate_count": len(scored),
            "leading_model_for_review": chosen["model"] if chosen else "NONE",
            "leading_variant_for_review": chosen["variant"] if chosen else "NONE",
            "leading_diagnostic_pct": chosen["compatibility_diagnostic_pct"] if chosen else "",
            "leading_chinese_chars": chosen["han_chars"] if chosen else "",
            "needs_review": "YES",
            "reason": (
                "NO_CANDIDATE_PASSED_BASIC_CHECKS" if chosen is None
                else "MODEL_TEXT_UNVERIFIED_AGAINST_IMAGE"
            ),
            "phien_am_ocr_excerpt": phonetic[:140],
        }
        summaries.append(summary)
        queue.append({
            **summary,
            "book_image": f"02_ocr_tho/han/{doc_id}_1.jpg",
            "action": "Verify image, printed Phiên âm and Chinese characters; do not auto-accept",
        })
        if chosen:
            for pos, (character, syllable, status) in enumerate(alns[id(chosen)], 1):
                normalized = normalize_reading(syllable) if syllable else ""
                similar_candidates = (
                    suggest_similar(character, syllable, similarity, readings)
                    if status == "incompatible" else []
                )
                review_alignments.append({
                    "doc_id": doc_id,
                    "leading_model_for_review": chosen["model"],
                    "leading_variant_for_review": chosen["variant"],
                    "position": pos,
                    "ocr_chinese": character,
                    "printed_phien_am_ocr": syllable,
                    "dictionary_status": status.upper(),
                    "supplementary_dictionary_match": (
                        "YES" if character and normalized in supplemental.get(character, set()) else "NO"
                    ),
                    "visually_similar_reading_candidates": " ".join(similar_candidates),
                    "confirmed_against_image": "NO",
                })
        print(f"  {doc_id}: {len(syllables)} syllables, {len(rows)} candidates; "
              f"leading diagnostic candidate = {summary['leading_model_for_review']} "
              f"({summary['leading_diagnostic_pct']}%) [UNVERIFIED]")
    target_root = base / "han_auto_compare_draft"
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    destination = target_root / stamp
    destination.mkdir(parents=True, exist_ok=False)
    comparison_cols = [
        "doc_id", "model", "variant", "source_file", "source_status",
        "comparison_status", "han_chars", "phien_am_syllables", "compatible_pairs",
        "incompatible_pairs", "han_only", "phien_am_only",
        "compatibility_diagnostic_pct", "flags", "elapsed_seconds", "preview_han",
        "dich_nghia_heading_found", "verified", "ready_for_training",
    ]
    summary_cols = [
        "doc_id", "phien_am_ocr_syllables", "candidate_count", "scored_candidate_count",
        "leading_model_for_review", "leading_variant_for_review", "leading_diagnostic_pct",
        "leading_chinese_chars", "needs_review", "reason", "phien_am_ocr_excerpt",
    ]
    write_csv(destination / "model_comparison.csv", comparisons, comparison_cols)
    write_csv(destination / "document_summary.csv", summaries, summary_cols)
    write_csv(destination / "review_queue.csv", queue, summary_cols + ["book_image", "action"])
    write_csv(destination / "leading_candidate_alignment.csv", review_alignments, [
        "doc_id", "leading_model_for_review", "leading_variant_for_review",
        "position", "ocr_chinese", "printed_phien_am_ocr", "dictionary_status",
        "supplementary_dictionary_match", "visually_similar_reading_candidates",
        "confirmed_against_image",
    ])
    report = {
        "status": "OCR_CANDIDATE_DIAGNOSTIC_ONLY",
        "verified": False,
        "ready_for_training": False,
        "warning": (
            "Dictionary-compatible alignment can be wrong even when its percentage is high. "
            "The printed Phiên âm is itself OCR. No manual ground truth was provided. "
            "NEVER report this as the teacher's >=95% matching score."
        ),
        "source_files": [safe_relative(p, root) for p in candidate_files],
        "input_ocr_labels": [safe_relative(base / "han" / "Label.txt", root),
                             safe_relative(viet_path, root)],
        "dictionary_main": safe_relative(main, root),
        "documents": summaries,
        "skipped_editorial_notes": {doc: extract_phien_am(viet.get(doc, []))[2] for doc in doc_ids},
    }
    (destination / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nReports saved (no source OCR changed):", destination)
    print("Open document_summary.csv and model_comparison.csv first.")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Step 2H: Compare automated Chinese OCR drafts")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dictionary-folder", default="dictionaries")
    parser.add_argument("--docs", nargs="+", help="Optional document IDs (e.g. PILOT.006.01)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-archive", dest="include_archive", action="store_false",
                        help="Only compare results inside the active output folder")
    parser.set_defaults(include_archive=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
