"""Step 2K v2 - recover clipped Vietnamese Phiên âm through source-PDF recropping.

Adds NEW draft output. Never modifies Step 1C crops, Step 2I checkpoints, or
any verified data. Requires pymupdf and Tesseract (vie+eng), already installed.

Run from project root:
    python -m steps.step2k_v2_crop_recovery --dry-run
    python -m steps.step2k_v2_crop_recovery --limit 5
    python -m steps.step2k_v2_crop_recovery --all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
DEFAULT_PADS = [0.035, 0.095, 0.155]


def count_words(text: str) -> int:
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def normalize_heading(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", text)


def heading(line: str) -> tuple[str | None, str]:
    """Match only at beginning of OCR line, tolerate small punctuation noise."""
    s = line.strip()
    if not s:
        return None, ""
    for pattern, name in (
        (r"^\s*(?:phi[eê]n|phien)\s*(?:a|â)m\s*[:：.\-]?", "phien_am"),
        (r"^\s*(?:d[iị]ch|dich)\s*(?:ngh[iĩ]a|nghia)\s*[:：.\-]?", "dich_nghia"),
    ):
        match = re.match(pattern, s, re.I)
        if match:
            return name, s[match.end():].strip()
    # Treat short heading prefixes as a full heading, not arbitrary prose.
    for words in (3, 2, 1):
        tokens = list(re.finditer(r"\S+", s[:36]))
        if len(tokens) < words:
            continue
        end = tokens[words - 1].end()
        value = normalize_heading(s[:end])
        if value in ("phienam", "dichnghia"):
            return ("phien_am" if value == "phienam" else "dich_nghia"), s[end:].lstrip(" :：.\t-")
    return None, ""


def parse_last_complete_pair(raw: str) -> dict:
    """Extract anchored Phiên âm -> Dịch nghĩa section without mixing documents."""
    lines = [unicodedata.normalize("NFC", x.strip()) for x in raw.splitlines()]
    markers = []
    for pos, text in enumerate(lines):
        section, remainder = heading(text)
        if section is not None:
            markers.append((pos, section, remainder))
    pairs = []
    for i, (start, section, remainder) in enumerate(markers):
        if section != "phien_am":
            continue
        following = markers[i + 1:]
        if not following or following[0][1] != "dich_nghia":
            continue
        end, _, dich_rem = following[0]
        stop = following[1][0] if len(following) > 1 else len(lines)
        phien_lines = ([remainder] if remainder else []) + lines[start + 1:end]
        dich_lines = ([dich_rem] if dich_rem else []) + lines[end + 1:stop]
        phien = "\n".join(x for x in phien_lines if x).strip()
        dich = "\n".join(x for x in dich_lines if x).strip()
        if phien and dich:
            pairs.append((start, phien, dich))
    if not pairs:
        return {"phien_am": "", "dich_nghia": "", "phien_am_syllables": 0,
                "dich_nghia_words": 0, "complete_pairs": 0}
    # Last complete section pair is nearest the target crop; earlier text is
    # from the preceding decree when the top boundary was expanded.
    _, phien, dich = pairs[-1]
    return {"phien_am": phien, "dich_nghia": dich,
            "phien_am_syllables": count_words(phien),
            "dich_nghia_words": count_words(dich), "complete_pairs": len(pairs)}


def find_run(explicit: str) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "source_manifest.json").is_file():
            raise FileNotFoundError(f"Not a Step 2I run: {root}")
        return root
    parent = BOOK / "02_ocr_tho" / "step2i_batch_draft"
    choices = []
    for root in parent.glob("run_*") if parent.exists() else []:
        if (root / "source_manifest.json").is_file() and (root / "vi").is_dir():
            choices.append((len(list((root / "vi").glob("AUTO.*.json"))), root.name, root))
    if not choices:
        raise FileNotFoundError("No Step 2I run with Vietnamese results found")
    return max(choices)[2]


def find_pdf(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        value = "source.pdf"
        config = ROOT / "config.yaml"
        if config.exists():
            try:
                import yaml
                value = str((yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("pdf_path", value))
            except (ImportError, OSError, ValueError, TypeError):
                pass
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original PDF missing: {path}")
    return path


def tesseract_exe(explicit: str) -> str:
    options = [explicit, shutil.which("tesseract") or ""]
    if sys.platform == "win32":
        options.append(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    for option in options:
        if option and Path(option).is_file():
            return option
    raise FileNotFoundError("Tesseract executable not found")


def get_targets(run: Path) -> list[tuple[str, dict]]:
    result = []
    for path in sorted((run / "vi").glob("AUTO.*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            value = record.get("result", {})
            if count_words(str(value.get("phien_am", ""))) < 8 or count_words(str(value.get("dich_nghia", ""))) < 8:
                result.append((record["doc_id"], record))
        except (ValueError, OSError, KeyError) as error:
            print(f"WARNING: Skipping unreadable record {path}: {error}")
    return result


def load_geometry(source: Path) -> dict[str, dict]:
    filename = source / "document_manifest.csv"
    if not filename.is_file():
        raise FileNotFoundError(f"Missing Step 1C manifest: {filename}")
    with filename.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    ids = [x["auto_id"] for x in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate AUTO IDs in Step 1C manifest")
    return {x["auto_id"]: x for x in records}


def make_crop(page, box: list[float], pad: float, dpi: int, filename: Path) -> None:
    if not (0 <= pad <= 0.28):
        raise ValueError("Invalid upward expansion")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Invalid normalized bounding box: {box}")
    rect = page.rect
    clip = pymupdf.Rect(x0 * rect.width, max(0, y0 - pad) * rect.height,
                        x1 * rect.width, y1 * rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), clip=clip, alpha=False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(filename), jpg_quality=94)


def ocr(image: Path, exe: str, psm: int) -> str:
    args = [exe, str(image), "stdout", "-l", "vie+eng", "--psm", str(psm)]
    result = subprocess.run(args, capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-600:])
    return result.stdout.decode("utf-8-sig", "replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2K v2: recover clipped Phiên âm by recropping source PDF")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--source-run", default="")
    parser.add_argument("--manifest", default="", help="Optional Step 1C export folder")
    parser.add_argument("--pdf", default="", help="Optional PDF path")
    parser.add_argument("--tesseract", default="")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--pads", type=float, nargs="+", default=DEFAULT_PADS,
                        help="Top expansions as fraction of PDF page height")
    args = parser.parse_args()
    if args.limit < 1 or not 140 <= args.dpi <= 360 or not args.pads or any(not 0.0 < x <= 0.28 for x in args.pads):
        parser.error("Invalid --limit, --dpi or --pads")

    run = find_run(args.source_run)
    manifest = json.loads((run / "source_manifest.json").read_text(encoding="utf-8-sig"))
    source = Path(args.manifest).resolve() if args.manifest else Path(manifest["source_step1c_run"])
    rows = load_geometry(source)
    targets = get_targets(run)
    if args.ids:
        selected = [(doc, item) for doc, item in targets if doc in set(args.ids)]
    elif args.all:
        selected = targets
    else:
        selected = targets[:args.limit]

    print("STEP 2K v2 - CROP-BASED VIETNAMESE RECOVERY (UNVERIFIED)")
    print("Step 2I:", run)
    print("Step 1C:", source)
    print("Missing-section targets:", len(targets))
    print("Selected:", len(selected))
    print("IDs:", ", ".join(d for d, _ in selected))
    print("Top expansions:", args.pads)
    if args.dry_run:
        print("DRY RUN: no PDF rendering, OCR, or new files")
        return
    if not selected:
        print("Nothing selected")
        return
    pdf_path = find_pdf(args.pdf)
    exe = tesseract_exe(args.tesseract)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    out = BOOK / "03_vi_section_recovery_v2" / stamp
    (out / "records").mkdir(parents=True)
    (out / "candidate_crops").mkdir()
    (out / "diagnostics").mkdir()
    summaries = []
    counts = Counter()
    with pymupdf.open(pdf_path) as pdf:
        for doc, item in selected:
            errs = []
            existing = item.get("result", {})
            old_p = count_words(str(existing.get("phien_am", "")))
            old_d = count_words(str(existing.get("dich_nghia", "")))
            candidates = []
            source_row = rows.get(doc)
            if source_row is None:
                errs.append("NO_LAYOUT_RECORD")
            else:
                page_idx = int(source_row["pdf_page"]) - 1
                if not 0 <= page_idx < len(pdf):
                    errs.append("INVALID_PAGE_INDEX")
                else:
                    box = json.loads(source_row["viet_bbox_normalized"])
                    # Each candidate gets its own image so the original crop stays unchanged.
                    for pad in sorted(set(args.pads)):
                        filename = out / "candidate_crops" / f"{doc}_up_{int(pad * 1000):03d}.jpg"
                        try:
                            make_crop(pdf[page_idx], box, pad, args.dpi, filename)
                            raw = ocr(filename, exe, psm=6)
                            parsed = parse_last_complete_pair(raw)
                            parsed.update({"pad": pad, "psm": 6, "source_image": filename.name, "raw_text": raw})
                            candidates.append(parsed)
                        except Exception as exc:
                            errs.append(f"pad {pad:.3f}: {exc}")
            # An anchored complete pair is a prerequisite; word count alone is NOT sufficient.
            valid = [c for c in candidates if c["phien_am_syllables"] >= 12 and c["dich_nghia_words"] >= 20]
            # Larger text is not always better; avoid selecting accidental large merged passages.
            best = max(valid, key=lambda c: (min(c["phien_am_syllables"], 200),
                                                -abs(c["dich_nghia_words"] - old_d),
                                                -c["pad"])) if valid else None
            phien = best["phien_am"] if best else str(existing.get("phien_am", ""))
            # Preserve existing Dịch nghĩa if already present; no silent translation replacement.
            dich = str(existing.get("dich_nghia", "")) if old_d >= 20 else (best["dich_nghia"] if best else "")
            flags = ["CROP_GEOMETRY_ADJUSTED", "TRANSCRIPTION_UNVERIFIED"] if best else ["UNRESOLVED_NO_COMPLETE_PAIR"]
            if best and best["complete_pairs"] > 1:
                flags.append("MULTIPLE_SECTION_PAIRS_IN_EXPANDED_CROP")
            if best and old_d >= 20 and abs(best["dich_nghia_words"] - old_d) / max(old_d, 1) > .30:
                flags.append("TRANSLATION_LENGTH_DIFFERS_FROM_EXISTING")
            if best and best["phien_am_syllables"] >= 220:
                flags.append("READING_POSSIBLY_MERGED")
            status = "RECOVERED_CANDIDATE_NEEDS_REVIEW" if best else "UNRESOLVED_MISSING_PHIEN_AM"
            record = {
                "doc_id": doc, "pdf_page": int(source_row["pdf_page"]) if source_row else None,
                "status": status, "verified": False, "ready_for_training": False,
                "phien_am": phien, "dich_nghia": dich,
                "phien_am_syllables": count_words(phien), "dich_nghia_words": count_words(dich),
                "original_phien_am_syllables": old_p, "original_dich_nghia_words": old_d,
                "selected_top_expansion": best["pad"] if best else None,
                "selected_crop": best["source_image"] if best else None,
                "candidate_counts": [{"pad": c["pad"], "psm": c["psm"],
                                      "phien_am_syllables": c["phien_am_syllables"],
                                      "dich_nghia_words": c["dich_nghia_words"],
                                      "complete_pairs": c["complete_pairs"]} for c in candidates],
                "flags": flags, "errors": errs,
                "source_step2i": str(run / "vi" / f"{doc}.json"),
                "warning": "Unverified OCR; inspect crop and printed reading before accepting"
            }
            (out / "records" / f"{doc}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            for c in candidates:
                # Candidate text may reveal heading errors without rerunning OCR.
                f = out / "diagnostics" / f"{doc}_up_{int(c['pad']*1000):03d}.txt"
                f.write_text(c["raw_text"], encoding="utf-8")
            summary = {"doc_id": doc, "status": status, "phien_am_before": old_p,
                       "phien_am_after": record["phien_am_syllables"], "dich_nghia_before": old_d,
                       "dich_nghia_after": record["dich_nghia_words"],
                       "selected_top_expansion": record["selected_top_expansion"],
                       "flags": "; ".join(flags), "errors": "; ".join(errs)}
            summaries.append(summary)
            counts[status] += 1
            print(f"{doc}: {status}, Phiên âm {old_p}->{record['phien_am_syllables']}, "
                  f"Dịch nghĩa {old_d}->{record['dich_nghia_words']}")

    with (out / "recovery_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["doc_id", "status", "phien_am_before", "phien_am_after", "dich_nghia_before",
                "dich_nghia_after", "selected_top_expansion", "flags", "errors"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summaries)
    meta = {"stage": "Step 2K v2", "source_step2i": str(run), "source_pdf": str(pdf_path),
            "source_step1c": str(source), "processed": len(summaries), "status_counts": dict(counts),
            "verified": False, "original_ocr_changed": False,
            "warning": "A recovered Phiên âm is still an unverified OCR draft."}
    (out / "recovery_report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out)
    print("SOURCE CROPS AND OCR CHECKPOINTS WERE NOT MODIFIED")


if __name__ == "__main__":
    main()
