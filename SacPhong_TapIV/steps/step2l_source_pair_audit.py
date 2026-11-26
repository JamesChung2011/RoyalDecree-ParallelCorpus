"""Step 2L — audit pages with two decree photographs but only one printed Vietnamese block.

This is a SOURCE-COVERAGE/PAIRING AUDIT only; it NEVER transcribes or edits the
OCR, never moves files, and never makes one translation serve two photographs.

Run:
    python -m steps.step2l_source_pair_audit --dry-run
    python -m steps.step2l_source_pair_audit

Requires PyMuPDF (pymupdf), already used in Steps 1C/2K.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
PAGE_CASES = {
    82: {
        "upper": "AUTO.082.01", "lower": "AUTO.082.02",
        "observation": "Two Hán photographs on the left; one printed Phiên âm/Dịch nghĩa block at lower right.",
    },
    90: {
        "upper": "AUTO.090.01", "lower": "AUTO.090.02",
        "observation": "Two Hán photographs on the left; one printed Phiên âm/Dịch nghĩa block at lower right.",
    },
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def find_step1c(explicit: str) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not (source / "document_manifest.csv").is_file():
            raise FileNotFoundError(f"Missing document_manifest.csv: {source}")
        return source
    root = BOOK / "01_layout_resolved_draft"
    choices = []
    for candidate in root.glob("run_*") if root.exists() else []:
        if (candidate / "document_manifest.csv").is_file():
            # Prefer the run where the 291 image pairs were actually exported.
            has_crops = (
                (candidate / "crops" / "han" / "AUTO.082.01_1.jpg").is_file()
                and (candidate / "crops" / "viet" / "AUTO.090.02_1.jpg").is_file()
            )
            choices.append((int(has_crops), candidate.name, candidate))
    if not choices:
        raise FileNotFoundError(f"No Step 1C manifest in {root}")
    return max(choices)[2]


def find_pdf(explicit: str) -> Path:
    if explicit:
        file = Path(explicit).expanduser().resolve()
    else:
        file = ROOT / "source.pdf"
        config = ROOT / "config.yaml"
        if config.is_file():
            try:
                import yaml
                cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
                relative = Path(str(cfg.get("pdf_path", "source.pdf")))
                file = relative if relative.is_absolute() else ROOT / relative
            except (ImportError, ValueError, OSError, TypeError):
                pass
        file = file.resolve()
    if not file.is_file():
        raise FileNotFoundError(f"PDF not found: {file}")
    return file


def bbox(row: dict, name: str) -> tuple[float, float, float, float]:
    coords = json.loads(row[name]) if isinstance(row[name], str) else row[name]
    if len(coords) != 4:
        raise ValueError(f"Bad {name} for {row.get('auto_id')}: {coords}")
    x0, y0, x1, y1 = map(float, coords)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Out-of-range {name} for {row.get('auto_id')}: {coords}")
    return x0, y0, x1, y1


def render_audit(page: pymupdf.Page, upper: dict, lower: dict, dest: Path) -> None:
    """Save an ORIGINAL-page visualization with blue photo boxes and text regions."""
    zoom = 1.65
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    # Annotate an in-memory page copy so the original PDF remains untouched.
    scratch = pymupdf.open()
    try:
        tmp_page = scratch.new_page(width=page.rect.width, height=page.rect.height)
        tmp_page.show_pdf_page(tmp_page.rect, page.parent, page.number)
        for row, color, label in ((upper, (0.05, 0.3, 0.9), "UPPER HAN"),
                                  (lower, (0.05, 0.3, 0.9), "LOWER HAN")):
            box = bbox(row, "han_bbox_normalized")
            rect = pymupdf.Rect(box[0] * page.rect.width, box[1] * page.rect.height,
                                box[2] * page.rect.width, box[3] * page.rect.height)
            tmp_page.draw_rect(rect, color=color, width=2.4)
            tmp_page.insert_text((rect.x0 + 2, max(15, rect.y0 - 3)),
                                 label + " " + row["auto_id"], fontsize=9, color=color)
        for row, color, label in ((upper, (0.85, 0.12, 0.12), "NO PRINTED TEXT IN THIS REGION"),
                                  (lower, (0.0, 0.58, 0.23), "ONE PRINTED VIET SECTION")):
            box = bbox(row, "viet_bbox_normalized")
            rect = pymupdf.Rect(box[0] * page.rect.width, box[1] * page.rect.height,
                                box[2] * page.rect.width, box[3] * page.rect.height)
            tmp_page.draw_rect(rect, color=color, width=2.4)
            tmp_page.insert_text((rect.x0 + 2, max(15, rect.y0 - 3)), label, fontsize=8, color=color)
        output_pix = tmp_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        output_pix.save(str(dest), jpg_quality=89)
    finally:
        scratch.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 2-photo / 1-translation source pages")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="", help="Explicit Step 1C exported run folder")
    parser.add_argument("--pdf", default="", help="Explicit source PDF path")
    args = parser.parse_args()

    source = find_step1c(args.manifest)
    rows = read_csv(source / "document_manifest.csv")
    lookup = {row.get("auto_id"): row for row in rows}
    if len(lookup) != len(rows):
        raise RuntimeError("Duplicate document IDs in source manifest")
    if len(rows) != 291:
        print(f"WARNING: Manifest has {len(rows)} rows, not the expected 291")

    print("STEP 2L — SOURCE IMAGE / PRINTED TEXT AUDIT")
    print("Step 1C source:", source)
    print("Pairs currently proposed:", len(rows))
    for number, case in PAGE_CASES.items():
        for key in ("upper", "lower"):
            if case[key] not in lookup:
                raise RuntimeError(f"Required source ID missing: {case[key]}")
            if int(lookup[case[key]]["pdf_page"]) != number:
                raise RuntimeError(f"Unexpected PDF page for {case[key]}")
        print(f"PDF {number}: {case['upper']} and {case['lower']}: 2 photos / 1 printed Vietnamese section")

    if args.dry_run:
        print("DRY RUN: no OCR, model calls, output, or file changes")
        return

    pdf_path = find_pdf(args.pdf)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    dest = BOOK / "03_source_pair_audit_draft" / stamp
    dest.mkdir(parents=True, exist_ok=False)

    exceptions = []
    try:
        with pymupdf.open(pdf_path) as book:
            for pdf_number, case in PAGE_CASES.items():
                if pdf_number < 1 or pdf_number > len(book):
                    raise ValueError(f"PDF page {pdf_number} out of range")
                upper = lookup[case["upper"]]
                lower = lookup[case["lower"]]
                render_audit(book[pdf_number - 1], upper, lower,
                             dest / "previews" / f"page_{pdf_number:03d}_annotated.jpg")
                exceptions.append({
                    "pdf_page": pdf_number,
                    "upper_image_doc_id": case["upper"],
                    "lower_image_doc_id": case["lower"],
                    "printed_vietnamese_blocks_visible": 1,
                    "han_photographs_visible": 2,
                    "upper_status": "SOURCE_PAIRING_UNRESOLVED_NO_PRINTED_TEXT_IN_UPPER_REGION",
                    "lower_status": "PRINTED_TEXT_PRESENT_BUT_DOCUMENT_MAPPING_UNVERIFIED",
                    "reason": case["observation"],
                    "do_not": "Do not copy the lower text to the upper photo or count it twice",
                    "next_check": "Determine whether photos are two documents, or two views of one document",
                })
    except Exception:
        # Keep diagnostics rather than erasing existing folders, if a PDF fails midway.
        raise

    fields = ["pdf_page", "upper_image_doc_id", "lower_image_doc_id",
              "printed_vietnamese_blocks_visible", "han_photographs_visible",
              "upper_status", "lower_status", "reason", "do_not", "next_check"]
    write_csv(dest / "source_pair_exceptions.csv", exceptions, fields)
    payload = {
        "stage": "Step 2L", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(pdf_path), "source_manifest": str(source),
        "provisional_pairs_in_original_manifest": len(rows),
        "pages_audited": list(PAGE_CASES), "source_coverage_exceptions": exceptions,
        "verified_alignments": 0, "original_files_modified": False,
        "warning": "This source audit identifies ambiguous image-to-text pairing only. No text was generated, changed or transferred. Keep original 291-pair manifest for traceability."
    }
    (dest / "source_pair_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Source-pair exceptions:", len(exceptions))
    print("Saved to:", dest)
    print("No OCR records or existing manifests were changed.")


if __name__ == "__main__":
    main()
