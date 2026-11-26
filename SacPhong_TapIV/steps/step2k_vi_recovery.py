r"""Step 2K: selectively recover missing Vietnamese OCR sections from Step 2I.

Safe by default: reads existing Step 2I checkpoints and writes a separate
candidate folder. Does NOT modify original files or claim verification.

Commands from project root:
  .\.venv\Scripts\python.exe -m steps.step2k_vi_recovery --dry-run
  .\.venv\Scripts\python.exe -m steps.step2k_vi_recovery --limit 5
  .\.venv\Scripts\python.exe -m steps.step2k_vi_recovery --limit 5 --reocr
  .\.venv\Scripts\python.exe -m steps.step2k_vi_recovery --all --reocr
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from steps import step2i_batch_ocr as original

ROOT = Path(__file__).resolve().parents[1]
BOOK = original.load_book_root()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower()).replace("đ", "d")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", value)


def find_heading(line: str) -> tuple[str | None, str, bool]:
    """Return heading, remainder, fuzzy. Only match line starts; never infer content."""
    source = line.strip()
    if not source:
        return None, "", False
    for target, key in (("phienam", "phien_am"), ("dichnghia", "dich_nghia")):
        # Normalize just a short prefix; OCR may merge or break heading letters.
        prefix = source[: min(34, len(source))]
        short = re.split(r"[:：]", prefix, maxsplit=1)[0]
        token_matches = list(re.finditer(r"\S+", short))
        for n in (3, 2, 1):
            if len(token_matches) < n:
                continue
            end = token_matches[n - 1].end()
            head = normalize(short[:end])
            if not head or len(head) < len(target) - 1 or abs(len(head) - len(target)) > 2:
                continue
            distance = difflib.SequenceMatcher(None, head, target).ratio()
            if head == target or distance >= 0.78:
                remainder = source[end:].lstrip(" :：\t-–—")
                return key, remainder, head != target
    return None, "", False


def section_words(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def extract_sections(raw: str) -> dict:
    current = None
    lines = {"phien_am": [], "dich_nghia": []}
    headings: list[str] = []
    fuzzy_used = False
    for line in raw.splitlines():
        clean = unicodedata.normalize("NFC", line.strip())
        if not clean:
            continue
        heading, remainder, fuzzy = find_heading(clean)
        if heading:
            current = heading
            headings.append(heading)
            fuzzy_used = fuzzy_used or fuzzy
            if remainder:
                lines[current].append(remainder)
            continue
        if current:
            lines[current].append(clean)
    phien = "\n".join(lines["phien_am"]).strip()
    dich = "\n".join(lines["dich_nghia"]).strip()
    return {
        "phien_am": phien,
        "dich_nghia": dich,
        "phien_am_syllables": section_words(phien),
        "dich_nghia_words": section_words(dich),
        "headings_detected": headings,
        "fuzzy_heading_used": fuzzy_used,
    }


def score(candidate: dict) -> tuple[int, int, int, int]:
    """Ranking is for a review draft, not a correctness/accuracy measure."""
    p = candidate["phien_am_syllables"]
    d = candidate["dich_nghia_words"]
    return (
        int(p >= 8) + int(d >= 8),
        int(p > 0) + int(d > 0),
        min(p, 220) + min(d, 400),
        -int(candidate.get("fuzzy_heading_used", False)),
    )


def candidate_from_original(result: dict) -> dict:
    phien = unicodedata.normalize("NFC", str(result.get("phien_am", "")).strip())
    dich = unicodedata.normalize("NFC", str(result.get("dich_nghia", "")).strip())
    return {
        "source": "step2i_selected",
        "phien_am": phien,
        "dich_nghia": dich,
        "phien_am_syllables": section_words(phien),
        "dich_nghia_words": section_words(dich),
        "fuzzy_heading_used": False,
        "headings_detected": [],
    }


def candidate_from_raw(source: str, raw: str) -> dict:
    parsed = extract_sections(raw)
    parsed["source"] = source
    return parsed


def find_run(explicit: str) -> Path:
    if explicit:
        run = Path(explicit).expanduser().resolve()
        if not (run / "source_manifest.json").is_file():
            raise FileNotFoundError(f"Not a Step 2I run: {run}")
        return run
    parent = BOOK / "02_ocr_tho" / "step2i_batch_draft"
    possibilities = []
    for run in parent.glob("run_*") if parent.exists() else []:
        if (run / "source_manifest.json").is_file() and (run / "vi").is_dir():
            count = sum(1 for _ in (run / "vi").glob("AUTO.*.json"))
            possibilities.append((count, run.name, run))
    if not possibilities:
        raise FileNotFoundError("No Step 2I run with Vietnamese records found")
    # Prefer the most complete run, then latest identifier.
    return max(possibilities)[2]


def load_target_records(run: Path) -> list[tuple[Path, dict]]:
    targets = []
    for path in sorted((run / "vi").glob("AUTO.*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            print(f"WARNING: Cannot load {path.name}: {error}")
            continue
        result = item.get("result", {})
        flags = result.get("flags", [])
        if not isinstance(flags, list):
            flags = [str(flags)]
        missing = (
            section_words(str(result.get("phien_am", ""))) < 8
            or section_words(str(result.get("dich_nghia", ""))) < 8
            or any("MISSING_" in str(flag) for flag in flags)
        )
        if missing:
            targets.append((path, item))
    return targets


def resolve_crop(record: dict, source_run: Path) -> Path:
    rel = Path(str(record["source_image"]).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe crop path: {rel}")
    root = source_run.resolve()
    crop = (root / rel).resolve()
    if not crop.is_relative_to(root) or not crop.is_file():
        raise FileNotFoundError(f"Source crop not found: {crop}")
    return crop


def collect_candidates(item: dict, reocr: bool, exe: str, source_run: Path) -> tuple[list[dict], list[str]]:
    result = item["result"]
    candidates: list[dict] = [candidate_from_original(result)]
    errors: list[str] = []
    seen_raw: set[str] = set()

    for index, alternative in enumerate(result.get("alternatives", []), start=1):
        raw = str(alternative.get("raw_text", ""))
        if not raw or raw in seen_raw:
            continue
        seen_raw.add(raw)
        candidates.append(candidate_from_raw(
            f"saved_psm_{alternative.get('psm', index)}", raw
        ))
    raw_selected = str(result.get("raw_text", ""))
    if raw_selected and raw_selected not in seen_raw:
        candidates.append(candidate_from_raw("saved_selected_raw", raw_selected))

    if reocr:
        try:
            image = resolve_crop(item, source_run)
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"CROP_ERROR: {error}")
        else:
            for psm in (3, 4, 12):
                try:
                    raw = original.tesseract_text(exe, image, psm)
                except Exception as error:
                    errors.append(f"TESSERACT_PSM_{psm}: {error}")
                    continue
                if raw and raw not in seen_raw:
                    seen_raw.add(raw)
                    candidates.append(candidate_from_raw(f"reocr_psm_{psm}", raw))
    return candidates, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover missing Vietnamese OCR sections (drafts only)")
    parser.add_argument("--source-run", default="", help="Explicit Step 2I run directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reocr", action="store_true", help="Retry missing sections with Tesseract PSM 3,4,12")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Process all missing-section documents")
    parser.add_argument("--ids", nargs="*", default=[], help="Specific AUTO.xxx.xx IDs")
    parser.add_argument("--tesseract", default="", help="Tesseract executable path")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    run = find_run(args.source_run)
    manifest = json.loads((run / "source_manifest.json").read_text(encoding="utf-8-sig"))
    source_run = Path(manifest["source_step1c_run"])
    targets = load_target_records(run)
    if args.ids:
        selected = [(path, item) for path, item in targets if item["doc_id"] in set(args.ids)]
    elif args.all:
        selected = targets
    else:
        selected = targets[:args.limit]
    print("STEP 2K — VIETNAMESE SECTION RECOVERY (DRAFT)")
    print("Step 2I source:", run)
    print("Missing-section targets:", len(targets))
    print("Selected:", len(selected))
    print("Re-OCR enabled:", args.reocr)
    print("Document IDs:", ", ".join(item["doc_id"] for _, item in selected))
    if args.dry_run:
        print("DRY RUN: no OCR or new output files.")
        return
    if not selected:
        print("No missing-section documents selected. Nothing to do.")
        return

    exe = original.tesseract_exe(args.tesseract) if args.reocr else ""
    timestamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    output = BOOK / "03_vi_section_recovery_draft" / timestamp
    (output / "records").mkdir(parents=True)
    summary = []
    for path, item in selected:
        doc = item["doc_id"]
        candidates, errors = collect_candidates(item, args.reocr, exe, source_run)
        best = max(candidates, key=score)
        prior = candidates[0]
        improved = score(best) > score(prior)
        source = best["source"]
        p = best["phien_am_syllables"]
        d = best["dich_nghia_words"]
        status = "IMPROVED_DRAFT_NEEDS_VERIFICATION" if improved else "NO_IMPROVEMENT"
        if p < 8 or d < 8:
            status = "UNRESOLVED_MISSING_SECTION" if not improved else "PARTIAL_RECOVERY_NEEDS_REVIEW"
        record = {
            "doc_id": doc,
            "source_step2i_record": str(path),
            "verified": False,
            "ready_for_training": False,
            "status": status,
            "selected_source": source,
            "phien_am": best["phien_am"],
            "dich_nghia": best["dich_nghia"],
            "phien_am_syllables": p,
            "dich_nghia_words": d,
            "fuzzy_heading_used": best.get("fuzzy_heading_used", False),
            "original_phien_am_syllables": prior["phien_am_syllables"],
            "original_dich_nghia_words": prior["dich_nghia_words"],
            "available_candidates": [
                {"source": c["source"], "phien_am_syllables": c["phien_am_syllables"],
                 "dich_nghia_words": c["dich_nghia_words"], "fuzzy": c.get("fuzzy_heading_used", False)}
                for c in candidates
            ],
            "errors": errors,
            "warning": "Output is unverified OCR, not a corrected translation or a gold corpus."
        }
        (output / "records" / f"{doc}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary.append({
            "doc_id": doc, "status": status, "selected_source": source,
            "phien_am_before": prior["phien_am_syllables"], "phien_am_after": p,
            "dich_nghia_before": prior["dich_nghia_words"], "dich_nghia_after": d,
            "fuzzy_heading": best.get("fuzzy_heading_used", False),
            "errors": "; ".join(errors)
        })
        print(f"{doc}: {status}; Phiên âm {prior['phien_am_syllables']}→{p}; Dịch nghĩa {prior['dich_nghia_words']}→{d}")

    with (output / "recovery_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    report = {
        "stage": "Step 2K", "source_step2i": str(run), "selected": len(selected),
        "original_missing_section_targets": len(targets),
        "status_counts": dict(Counter(row["status"] for row in summary)),
        "verified": False, "new_source_ocr_modified": False,
        "warning": "Any recovered text still requires checking against the book."
    }
    (output / "recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Output:", output)
    print("No Step 2I files were modified.")


if __name__ == "__main__":
    main()
