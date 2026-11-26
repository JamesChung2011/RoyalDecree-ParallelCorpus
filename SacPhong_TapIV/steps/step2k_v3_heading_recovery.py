"""Step 2K v3: non-destructive recovery from already saved Step 2K v2 OCR text.

No OCR is performed. This deliberately does not invent text or change previous results.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "output" / "Sac_phong_Tap_IV" / "03_vi_section_recovery_v2"
OUTPUT_ROOT = ROOT / "output" / "Sac_phong_Tap_IV" / "03_vi_heading_recovery_v3"


def norm(value: str) -> str:
    value = value.strip().lower().replace("đ", "d")
    value = "".join(c for c in unicodedata.normalize("NFD", value)
                    if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z]+", value))


def classify_heading(line: str) -> str | None:
    stripped = line.strip().strip("'\"`?.,:;!()[] ")
    n = norm(stripped)
    if not n or len(n) > 22 or len(n.split()) > 3:
        return None
    if difflib.SequenceMatcher(a=n, b="phien am").ratio() >= 0.67:
        return "phien_am"
    if difflib.SequenceMatcher(a=n, b="dich nghia").ratio() >= 0.70:
        return "dich_nghia"
    return None


def count_words(value: str) -> int:
    return len(re.findall(r"[^\W\d_]+", value, re.UNICODE))


def parse_ocr(raw: str) -> list[dict]:
    lines = [x.strip() for x in raw.splitlines()]
    labels = [(i, classify_heading(line)) for i, line in enumerate(lines)]
    starts = [i for i, label in labels if label == "phien_am"]
    translations = [i for i, label in labels if label == "dich_nghia"]
    candidates: list[dict] = []
    for start in starts:
        endings = [i for i in translations if i > start]
        if not endings:
            continue
        end = endings[0]
        reading = "\n".join(v for v in lines[start + 1:end] if v)
        next_header = next((i for i, label in labels if i > end and label == "phien_am"), len(lines))
        translation = "\n".join(v for v in lines[end + 1:next_header] if v)
        r_count, t_count = count_words(reading), count_words(translation)
        if r_count < 25 or t_count < 15:
            continue
        candidates.append({
            "phien_am": reading, "dich_nghia": translation,
            "phien_am_syllables": r_count, "dich_nghia_words": t_count,
            "heading_lines": [lines[start], lines[end]],
        })
    return candidates


def find_latest_run() -> Path:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing Step 2K v2 results: {SOURCE_ROOT}")
    runs = sorted((p for p in SOURCE_ROOT.iterdir()
                   if p.is_dir() and (p / "recovery_summary.csv").is_file()
                   and (p / "diagnostics").is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError("No Step 2K v2 run with recovery_summary.csv and diagnostics found")
    return runs[0]


def process_doc(run: Path, docid: str) -> dict:
    previous_file = run / "records" / f"{docid}.json"
    if not previous_file.is_file():
        return {"doc_id": docid, "status": "MISSING_V2_RECORD", "phien_am_syllables": 0,
                "dich_nghia_words": 0, "selected_pad": "", "source_text": "", "flags": ["NO_V2_RECORD"]}
    previous = json.loads(previous_file.read_text(encoding="utf-8-sig"))
    files = sorted((run / "diagnostics").glob(f"{docid}_up_*.txt"))
    tried: list[dict] = []
    for path in files:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        for candidate in parse_ocr(raw):
            candidate["source_text"] = path.name
            try:
                candidate["pad"] = int(path.stem.rsplit("_", 1)[-1]) / 1000
            except ValueError:
                candidate["pad"] = 1.0
            tried.append(candidate)
    # A smaller crop minimizes contamination from neighboring documents.
    tried.sort(key=lambda c: (c["pad"], -c["phien_am_syllables"]))
    if tried:
        best = tried[0]
        return {
            "doc_id": docid, "pdf_page": previous.get("pdf_page"),
            "status": "RECOVERED_PARSER_CANDIDATE_NEEDS_REVIEW", "verified": False,
            "ready_for_training": False,
            "phien_am": best["phien_am"], "dich_nghia": best["dich_nghia"],
            "phien_am_syllables": best["phien_am_syllables"],
            "dich_nghia_words": best["dich_nghia_words"],
            "selected_pad": best["pad"], "source_text": best["source_text"],
            "matched_headings": best["heading_lines"],
            "flags": ["FUZZY_HEADING", "OCR_DRAFT_UNVERIFIED", "REVIEW_NEIGHBORING_TEXT"],
            "source_v2": str(previous_file),
        }
    all_empty = bool(files) and all(not f.read_text(encoding="utf-8-sig", errors="replace").strip() for f in files)
    return {
        "doc_id": docid, "pdf_page": previous.get("pdf_page"),
        "status": "UNRESOLVED_BLANK_CROP" if all_empty else "UNRESOLVED_SECTION_PAIR",
        "verified": False, "ready_for_training": False,
        "phien_am": "", "dich_nghia": "", "phien_am_syllables": 0,
        "dich_nghia_words": 0, "selected_pad": None, "source_text": "",
        "flags": ["NEEDS_LAYOUT_FIX" if all_empty else "NEEDS_FOCUSED_REVIEW"],
        "source_v2": str(previous_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2K v3: fuzzy heading recovery from v2 OCR texts")
    parser.add_argument("--source-run", type=Path, help="Optional specific Step 2K v2 run directory")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only, do not create output")
    args = parser.parse_args()
    run = args.source_run.resolve() if args.source_run else find_latest_run()
    source = run / "recovery_summary.csv"
    if not source.is_file():
        raise SystemExit(f"Missing {source}")
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["doc_id"] for r in rows if r.get("status") == "UNRESOLVED_MISSING_PHIEN_AM"]
    print("STEP 2K v3 - OCR HEADING RECOVERY (UNVERIFIED)")
    print("Source v2 run:", run)
    print("Unresolved documents:", len(ids), ", ".join(ids))
    print("No new OCR, AI, or original file changes.")
    if args.dry_run:
        print("DRY RUN COMPLETE")
        return
    results = [process_doc(run, id_) for id_ in ids]
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    target = OUTPUT_ROOT / stamp
    (target / "records").mkdir(parents=True, exist_ok=False)
    for record in results:
        (target / "records" / f"{record['doc_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(record["doc_id"], record["status"], record["phien_am_syllables"], "syllables")
    columns = ["doc_id", "status", "phien_am_syllables", "dich_nghia_words", "selected_pad", "source_text", "flags"]
    with (target / "recovery_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for record in results:
            writer.writerow({c: ";".join(record.get(c, [])) if c == "flags" else record.get(c, "") for c in columns})
    report = {"stage": "Step 2K v3", "source_v2_run": str(run), "processed": len(results),
              "status_counts": dict(Counter(record["status"] for record in results)),
              "verified": False, "original_files_changed": False,
              "warning": "Fuzzy heading recognition only; candidate text must be checked against printed source."}
    (target / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", target)


if __name__ == "__main__":
    main()
