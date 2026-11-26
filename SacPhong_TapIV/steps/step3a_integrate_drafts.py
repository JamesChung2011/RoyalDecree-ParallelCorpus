r"""Step 3A: conservative cleaning and source-aware merging of OCR drafts.

Run from the project root:
    .\.venv\Scripts\python.exe -m steps.step3a_integrate_drafts --dry-run
    .\.venv\Scripts\python.exe -m steps.step3a_integrate_drafts --limit 5
    .\.venv\Scripts\python.exe -m steps.step3a_integrate_drafts --all

No OCR is run. No previous files are changed. No output is verified training data.
Python 3.11+; standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
DOC_ID_RE = re.compile(r"AUTO\.\d{3}\.\d{2}$")
FALLBACK_UNPAIRED = {"AUTO.082.01", "AUTO.082.02", "AUTO.090.01", "AUTO.090.02"}
COLUMNS = [
    "doc_id", "pdf_page", "han", "phien_am", "dich_nghia", "han_characters",
    "phien_am_syllables", "dich_nghia_words", "phien_am_source", "dich_nghia_source",
    "source_pair_status", "draft_alignment_eligible", "verified", "ready_for_training",
    "status", "flags", "han_source_json", "viet_source_json", "recovery_source_json"
]


def book_root() -> Path:
    config = ROOT / "config.yaml"
    if config.exists():
        try:
            import yaml
            values = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            if isinstance(values, dict):
                return ROOT / str(values.get("output_dir", "output")) / str(
                    values.get("book_name", "Sac_phong_Tap_IV")
                )
        except (ImportError, OSError, ValueError, TypeError):
            pass
    return DEFAULT_BOOK


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def chosen_run(parent: Path, *, kind: str, explicit: str = "") -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"{kind} run not found: {candidate}")
        return candidate
    if not parent.is_dir():
        return None
    runs = [p for p in parent.glob("run_*") if p.is_dir()]
    if not runs:
        return None
    if kind == "Step 2I":
        runs = [p for p in runs if (p / "vi").is_dir() and (p / "source_manifest.json").exists()]
        if not runs:
            return None
        # Prefer the most complete run; avoid accidentally choosing a newer pilot-only run.
        return max(runs, key=lambda p: (
            len(list((p / "vi").glob("AUTO.*.json"))),
            len(list((p / "han").glob("AUTO.*.json"))), p.name
        ))
    return max(runs, key=lambda p: (len(list((p / "records").glob("AUTO.*.json"))), p.name))


def most_recent_audit(parent: Path, explicit: str = "") -> Path | None:
    if explicit:
        found = Path(explicit).expanduser().resolve()
        if not found.is_dir():
            raise FileNotFoundError(f"Step 2L audit run not found: {found}")
        return found
    available = [p for p in parent.glob("run_*") if (p / "source_pair_audit.json").is_file()] if parent.is_dir() else []
    return max(available, key=lambda p: p.name) if available else None


def exception_map(audit_run: Path | None) -> dict[str, str]:
    result = {doc: "SOURCE_PAIR_UNRESOLVED" for doc in FALLBACK_UNPAIRED}
    if audit_run is None:
        return result
    audit = read_json(audit_run / "source_pair_audit.json")
    for entry in audit.get("source_coverage_exceptions", []):
        if not isinstance(entry, dict):
            continue
        upper, lower = entry.get("upper_image_doc_id"), entry.get("lower_image_doc_id")
        if upper and DOC_ID_RE.fullmatch(str(upper)):
            result[str(upper)] = str(entry.get("upper_status") or "SOURCE_PAIR_UNRESOLVED")
        if lower and DOC_ID_RE.fullmatch(str(lower)):
            result[str(lower)] = str(entry.get("lower_status") or "SOURCE_PAIR_UNRESOLVED")
    return result


def normalize_whitespace(text: object) -> str:
    """Safe cleanup only. Preserve diacritics, Hán variants, newlines and footnotes."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    value = value.replace("\u00a0", " ").replace("\u2009", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    trimmed: list[str] = []
    for line in lines:
        if line or (trimmed and trimmed[-1]):
            trimmed.append(line)
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    return "\n".join(trimmed)


def count_words(value: str) -> int:
    return len(re.findall(r"[^\W\d_]+", value, flags=re.UNICODE))


def count_han(value: str) -> int:
    return sum(0x3400 <= ord(ch) <= 0x9FFF or
               0xF900 <= ord(ch) <= 0xFAFF or
               0x20000 <= ord(ch) <= 0x323AF for ch in value)


def recovery_records(run: Path | None) -> dict[str, tuple[dict, Path]]:
    answer: dict[str, tuple[dict, Path]] = {}
    if run is None:
        return answer
    for path in (run / "records").glob("AUTO.*.json"):
        if DOC_ID_RE.fullmatch(path.stem):
            obj = read_json(path)
            if obj.get("doc_id") == path.stem:
                answer[path.stem] = (obj, path)
    return answer


def vi_draft(raw: dict) -> dict:
    return raw.get("result", {}) if isinstance(raw.get("result"), dict) else {}


def build_one(doc_id: str, vi_path: Path | None, han_path: Path | None,
              v2: dict, v3: dict, exceptions: dict[str, str], audit_exists: bool) -> dict:
    vi = read_json(vi_path) if vi_path else {}
    han = read_json(han_path) if han_path else {}
    v = vi_draft(vi)
    h = han.get("result", {}) if isinstance(han.get("result"), dict) else {}
    flags: set[str] = set()
    recovery_path: Path | None = None
    recovery_source = ""

    reading = normalize_whitespace(v.get("phien_am", ""))
    translation = normalize_whitespace(v.get("dich_nghia", ""))
    reading_source = "Step2I_vi" if reading else "MISSING"
    translation_source = "Step2I_vi" if translation else "MISSING"

    # Preserve all original OCR text. The recovery replaces *missing sections only*,
    # not populated (possibly incorrect) sections, and not the entire document.
    if not reading or not translation:
        for tag, collection in (("Step2K_v3", v3), ("Step2K_v2", v2)):
            pair = collection.get(doc_id)
            if pair is None:
                continue
            record, path = pair
            if not str(record.get("status", "")).startswith("RECOVERED"):
                continue
            proposed_reading = normalize_whitespace(record.get("phien_am", ""))
            proposed_translation = normalize_whitespace(record.get("dich_nghia", ""))
            changed = False
            if not reading and count_words(proposed_reading) >= 10:
                reading = proposed_reading
                reading_source = tag
                changed = True
                flags.add("UNVERIFIED_RECOVERED_PHIEN_AM")
            if not translation and count_words(proposed_translation) >= 10:
                translation = proposed_translation
                translation_source = tag
                changed = True
                flags.add("UNVERIFIED_RECOVERED_DICH_NGHIA")
            if changed:
                recovery_path = path
                recovery_source = tag
            if reading and translation:
                break

    status_h = str(h.get("status", "MISSING"))
    chinese_raw = normalize_whitespace(h.get("chinese_candidate", ""))
    if status_h in {"OCR_ERROR", "REPETITION_REJECTED"}:
        chinese = ""
        flags.add("HAN_OCR_REJECTED")
    else:
        chinese = chinese_raw
    if "REPETITION_DETECTED" in h.get("flags", []):
        chinese = ""
        flags.add("HAN_REPETITION_DETECTED")
    if not han_path:
        flags.add("MISSING_HAN_RECORD")
    if not vi_path:
        flags.add("MISSING_VI_RECORD")
    if not chinese:
        flags.add("MISSING_HAN_TEXT")
    if not reading:
        flags.add("MISSING_PHIEN_AM")
    if not translation:
        flags.add("MISSING_DICH_NGHIA")

    han_count, phien_count, dich_count = count_han(chinese), count_words(reading), count_words(translation)
    if han_count and phien_count and abs(han_count - phien_count) > max(2, .15 * phien_count):
        flags.add("HAN_PHIEN_COUNT_GAP_GT_15PCT")
    if "□" in chinese:
        flags.add("UNREADABLE_HAN_MARKERS")
    for flag in h.get("flags", []):
        if isinstance(flag, str):
            flags.add("HAN_SOURCE_" + flag)

    pair_issue = exceptions.get(doc_id, "")
    if pair_issue:
        flags.add("SOURCE_PAIRING_UNRESOLVED")
    elif not audit_exists:
        flags.add("STEP2L_AUDIT_NOT_FOUND")

    draft_ok = bool(han_count and phien_count and dich_count and not pair_issue)
    status = "BLOCKED_SOURCE_PAIRING" if pair_issue else (
        "INCOMPLETE_OCR_DRAFT" if not (han_count and phien_count and dich_count) else "UNVERIFIED_DOCUMENT_DRAFT"
    )
    if draft_ok:
        flags.add("DRAFT_ALIGNMENT_ONLY_NOT_VERIFIED")
    page = vi.get("pdf_page", han.get("pdf_page", int(doc_id[5:8])))
    return {
        "doc_id": doc_id, "pdf_page": int(page),
        "han": chinese, "phien_am": reading, "dich_nghia": translation,
        "han_characters": han_count, "phien_am_syllables": phien_count,
        "dich_nghia_words": dich_count,
        "phien_am_source": reading_source, "dich_nghia_source": translation_source,
        "source_pair_status": pair_issue or "PROVISIONAL_PAIR_UNVERIFIED",
        "draft_alignment_eligible": draft_ok, "verified": False,
        "ready_for_training": False, "status": status,
        "flags": sorted(flags),
        "han_source_json": relative(han_path), "viet_source_json": relative(vi_path),
        "recovery_source_json": relative(recovery_path),
        "metadata": {
            "chinese_raw_before_rejection": chinese_raw if chinese != chinese_raw else "",
            "selected_recovery_stage": recovery_source,
            "normalization": "NFC and whitespace only; no historical character corrections",
            "diagnostic_only": True,
        },
    }


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["flags"] = "; ".join(record["flags"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3A: safely combine OCR and recovery drafts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=5, help="Pilot size (default 5)")
    parser.add_argument("--all", action="store_true", help="Process all available document IDs")
    parser.add_argument("--ids", nargs="+", default=[], help="Specific AUTO.xxx.xx documents")
    parser.add_argument("--step2i-run", default="", help="Optional exact Step 2I run directory")
    parser.add_argument("--v2-run", default="", help="Optional exact Step 2K v2 run directory")
    parser.add_argument("--v3-run", default="", help="Optional exact Step 2K v3 run directory")
    parser.add_argument("--audit-run", default="", help="Optional exact Step 2L run directory")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    book = book_root()
    step2i = chosen_run(book / "02_ocr_tho" / "step2i_batch_draft", kind="Step 2I", explicit=args.step2i_run)
    if step2i is None:
        raise SystemExit("Step 2I run not found. Process OCR with Step 2I first.")
    v2run = chosen_run(book / "03_vi_section_recovery_v2", kind="Step 2K v2", explicit=args.v2_run)
    v3run = chosen_run(book / "03_vi_heading_recovery_v3", kind="Step 2K v3", explicit=args.v3_run)
    auditrun = most_recent_audit(book / "03_source_pair_audit_draft", args.audit_run)
    v2, v3 = recovery_records(v2run), recovery_records(v3run)
    exceptions = exception_map(auditrun)
    vi_files = {p.stem: p for p in (step2i / "vi").glob("AUTO.*.json") if DOC_ID_RE.fullmatch(p.stem)}
    han_files = {p.stem: p for p in (step2i / "han").glob("AUTO.*.json") if DOC_ID_RE.fullmatch(p.stem)}
    doc_ids = sorted(set(vi_files) | set(han_files))
    meta_path = step2i / "source_manifest.json"
    metadata = read_json(meta_path)
    manifest_target = int(metadata.get("manifest_count", len(doc_ids)))
    if not doc_ids:
        raise SystemExit("No OCR checkpoint records found.")
    if args.ids:
        bad = [doc for doc in args.ids if doc not in doc_ids]
        if bad:
            parser.error("Unknown document IDs: " + ", ".join(bad))
        selected = sorted(set(args.ids))
    else:
        selected = doc_ids if args.all else doc_ids[:args.limit]
    print("STEP 3A - CONSERVATIVE OCR DRAFT INTEGRATION")
    print("Step 2I:", step2i)
    print("Vietnamese OCR records:", len(vi_files), "Chinese OCR records:", len(han_files))
    print("Manifest target:", manifest_target)
    print("Recovery v2 records:", len(v2), "Recovery v3 records:", len(v3))
    print("Step 2L source-pair audit:", auditrun if auditrun else "MISSING (known four IDs remain blocked)")
    print("Selected:", len(selected))
    if args.dry_run:
        print("DRY RUN: no files written; no OCR performed.")
        return
    records = [build_one(doc, vi_files.get(doc), han_files.get(doc), v2, v3,
                         exceptions, auditrun is not None) for doc in selected]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = book / "03_clean_merge_draft" / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "document_drafts.csv", records)
    with (out / "document_drafts.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    review = [r for r in records if r["flags"] and (
        r["status"] != "UNVERIFIED_DOCUMENT_DRAFT" or
        any(f not in {"DRAFT_ALIGNMENT_ONLY_NOT_VERIFIED"} for f in r["flags"])
    )]
    write_csv(out / "review_queue.csv", review)
    states = Counter(r["status"] for r in records)
    reading_sources = Counter(r["phien_am_source"] for r in records)
    report = {
        "stage": "Step 3A", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_step2i": relative(step2i), "source_v2": relative(v2run),
        "source_v3": relative(v3run), "source_step2l": relative(auditrun),
        "manifest_target": manifest_target, "ocr_records_vi": len(vi_files),
        "ocr_records_han": len(han_files), "documents_selected": len(selected),
        "status_counts": dict(states), "phien_am_sources": dict(reading_sources),
        "draft_alignment_eligible": sum(r["draft_alignment_eligible"] for r in records),
        "unresolved_pairing_count": sum(bool(r["source_pair_status"] != "PROVISIONAL_PAIR_UNVERIFIED") for r in records),
        "missing_phien_am_count": sum(not r["phien_am"] for r in records),
        "review_queue_count": len(review),
        "verified_transcriptions": 0, "ready_for_training": False,
        "warnings": [
            "Draft merge only: not sentence aligned and not the teacher's 95% accuracy measurement.",
            "Original OCR, dictionaries, layout manifests, and recovery records have not been changed.",
            "Source-pair exceptions on PDF pages 82 and 90 remain blocked.",
            "No automatic spelling, Hán character, year/date, or historical-name substitution."
        ]
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Results:", out)
    print("Status counts:", dict(states))
    print("Phiên âm provenance:", dict(reading_sources))
    print("Draft-alignment eligible (NOT verified):", report["draft_alignment_eligible"])
    print("Review queue:", len(review))
    print("Source OCR and recovered files NOT modified.")


if __name__ == "__main__":
    main()
