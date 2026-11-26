"""Step 1C: merge the Step1B-v2 175-page layout manifest with audited exception rules.

Produces a separate *provisional* geometry manifest and annotated exception previews.
Optionally exports crops, without changing Step1B, existing OCR, or PPOCRLabel files.

Run from project root:
    python -m steps.step1c_resolve_layout
    python -m steps.step1c_resolve_layout --export-crops

Nothing here verifies Chinese transcriptions or the Vietnamese translations.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import yaml
from PIL import Image, ImageDraw


DOCUMENT_COLS = [
    "auto_id", "pdf_page", "document_number", "layout", "status",
    "han_bbox_normalized", "viet_bbox_normalized", "han_image", "viet_image",
    "origin", "notes"
]
PAGE_COLS = ["pdf_page", "layout", "estimated_documents", "status", "flags", "resolution"]
SKIP_COLS = ["pdf_page", "reason"]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step1B-v2 file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_box(value: object, kind: str, doc_id: str) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{doc_id}: invalid {kind}: {value}")
    try:
        x0, y0, x1, y1 = map(float, value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{doc_id}: non-numeric {kind}: {value}") from exc
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"{doc_id}: out-of-range {kind}: {value}")
    if (x1 - x0) < 0.08 or (y1 - y0) < 0.07:
        raise ValueError(f"{doc_id}: unexpectedly small {kind}: {value}")
    return [round(v, 5) for v in (x0, y0, x1, y1)]


def validate_document(row: dict) -> dict:
    id_ = str(row["auto_id"])
    for kind in ("han_bbox_normalized", "viet_bbox_normalized"):
        row[kind] = validate_box(row[kind], kind, id_)
    a, b, c, d = row["han_bbox_normalized"]
    e, f, g, h = row["viet_bbox_normalized"]
    overlap_w = max(0.0, min(c, g) - max(a, e))
    overlap_h = max(0.0, min(d, h) - max(b, f))
    if overlap_w * overlap_h > 0.02:
        raise ValueError(f"{id_}: Hán/Viet crop intersection is too large")
    return row


def render_page(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def annotate_page(page, rows: list[dict], page_number: int, dest: Path) -> None:
    image = render_page(page, dpi=120)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for row in rows:
        for key, color in (("han_bbox_normalized", (15, 90, 235)),
                           ("viet_bbox_normalized", (0, 155, 75))):
            x0, y0, x1, y1 = row[key]
            draw.rectangle((x0*w, y0*h, x1*w, y1*h), outline=color, width=4)
        x0, y0 = row["han_bbox_normalized"][:2]
        draw.text((x0*w, max(5, y0*h - 18)), row["auto_id"], fill=(0, 45, 170))
    draw.text((12, 8), f"PDF {page_number} | Step1C | DRAFT: not verified", fill=(0, 0, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, quality=90)


def export_crop(page, box: list[float], dpi: int, target: Path) -> None:
    page_rect = page.rect
    x0, y0, x1, y1 = box
    clip = pymupdf.Rect(x0*page_rect.width, y0*page_rect.height,
                        x1*page_rect.width, y1*page_rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72),
                          clip=clip, alpha=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    pix.save(target, jpg_quality=93)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe whole-book crop manifest merger (no OCR)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overrides", default="step1c_page_overrides.json")
    parser.add_argument("--v2-manifest", default="", help="Step1B-v2 full_book folder override")
    parser.add_argument("--export-crops", action="store_true", help="Export all provisional Hán/Viet crops")
    parser.add_argument("--crop-dpi", type=int, default=220)
    parser.add_argument("--preview-pages", type=int, nargs="*", default=[7, 62, 86, 95, 107])
    args = parser.parse_args()
    if not 90 <= args.crop_dpi <= 360:
        parser.error("--crop-dpi must be between 90 and 360")

    root = Path.cwd()
    cfg_path = (root / args.config).resolve()
    override_path = (root / args.overrides).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    options = json.loads(override_path.read_text(encoding="utf-8"))
    pdf_path = Path(cfg.get("pdf_path", "source.pdf"))
    if not pdf_path.is_absolute():
        pdf_path = cfg_path.parent / pdf_path
    book_root = root / cfg.get("output_dir", "output") / cfg.get("book_name", "Sac_phong_Tap_IV")
    v2 = Path(args.v2_manifest).resolve() if args.v2_manifest else book_root / "01_layout_auto_draft_v2" / "full_book"

    source_pages = read_csv(v2 / "page_manifest.csv")
    source_docs = read_csv(v2 / "document_manifest.csv")
    pdf = pymupdf.open(pdf_path)
    try:
        expected = int(options["pdf_pages_expected"])
        if len(pdf) != expected or len(source_pages) != expected:
            raise RuntimeError(f"PDF/manifests should have {expected} pages, got {len(pdf)}/{len(source_pages)}")
        page_index = {int(r["pdf_page"]): r for r in source_pages}
        if set(page_index) != set(range(1, expected + 1)):
            raise RuntimeError("Step1B-v2 page manifest has missing or repeated page IDs")

        skipped = {int(k): v for k, v in options["non_content_pages"].items()}
        overridden = {int(k): v for k, v in options["decree_page_overrides"].items()}
        if set(skipped) & set(overridden):
            raise RuntimeError("Same page appears in both skip and override rules")
        flagged = {int(r["pdf_page"]) for r in source_pages if r["status"] != "PROVISIONAL_CROPS"}
        unexpected = flagged - set(skipped) - set(overridden)
        if unexpected:
            raise RuntimeError(f"Unresolved flagged pages: {sorted(unexpected)}")
        still_skipping_content = {page for page in skipped if page_index[page]["status"] == "PROVISIONAL_CROPS"}
        if still_skipping_content:
            raise RuntimeError(f"Non-content pages unexpectedly have successful crops: {sorted(still_skipping_content)}")

        all_docs: list[dict] = []
        pages: list[dict] = []
        removed: list[dict] = []
        for page_number in range(1, expected + 1):
            original_page = page_index[page_number]
            if page_number in skipped:
                pages.append({"pdf_page": page_number, "layout": "non_content", "estimated_documents": 0,
                              "status": "SKIP_NON_CONTENT", "flags": "", "resolution": skipped[page_number]})
                removed.append({"pdf_page": page_number, "reason": skipped[page_number]})
                continue
            if page_number in overridden:
                override = overridden[page_number]
                examples = override["documents"]
                for i, entry in enumerate(examples, start=1):
                    id_ = f"AUTO.{page_number:03d}.{i:02d}"
                    all_docs.append(validate_document({
                        "auto_id": id_, "pdf_page": page_number, "document_number": i,
                        "layout": override["layout"], "status": "PROVISIONAL_UNVERIFIED",
                        "han_bbox_normalized": entry["han_bbox_normalized"],
                        "viet_bbox_normalized": entry["viet_bbox_normalized"],
                        "han_image": "", "viet_image": "", "origin": "AUDITED_EXCEPTION_GEOMETRY",
                        "notes": override["reason"]
                    }))
                pages.append({"pdf_page": page_number, "layout": override["layout"],
                              "estimated_documents": len(examples), "status": "PROVISIONAL_EXCEPTION",
                              "flags": original_page["flags"], "resolution": override["reason"]})
                continue

            matching = [r for r in source_docs if int(r["pdf_page"]) == page_number]
            if original_page["status"] != "PROVISIONAL_CROPS" or not matching:
                raise RuntimeError(f"Page {page_number} is neither skipped, overridden, nor safely detected")
            for row in matching:
                all_docs.append(validate_document({
                    **row, "origin": "STEP1B_V2", "notes": "Unverified geometry from Step1B v2"
                }))
            pages.append({"pdf_page": page_number, "layout": original_page["layout"],
                          "estimated_documents": len(matching), "status": "PROVISIONAL_CROPS",
                          "flags": "", "resolution": "Step1B v2"})

        ids = [r["auto_id"] for r in all_docs]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate AUTO document IDs")
        if sum(int(p["estimated_documents"]) for p in pages) != len(all_docs):
            raise RuntimeError("Page/document count mismatch")
        if any(int(p["pdf_page"]) in skipped for p in all_docs):
            raise RuntimeError("Skipped page has an output document")
        by_page: dict[int, list[dict]] = {}
        for row in all_docs:
            by_page.setdefault(int(row["pdf_page"]), []).append(row)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = book_root / "01_layout_resolved_draft" / f"run_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        preview_pages = sorted(set(args.preview_pages) | set(overridden))
        for page_number in preview_pages:
            if page_number not in by_page:
                continue
            annotate_page(pdf[page_number - 1], by_page[page_number], page_number,
                          output / "previews" / f"page_{page_number:03d}_annotated.jpg")
        if args.export_crops:
            for idx, row in enumerate(all_docs, start=1):
                id_ = row["auto_id"]
                page = pdf[int(row["pdf_page"]) - 1]
                name = id_ + "_1.jpg"
                for field, dirname in (("han_bbox_normalized", "han"), ("viet_bbox_normalized", "viet")):
                    dest = output / "crops" / dirname / name
                    export_crop(page, row[field], args.crop_dpi, dest)
                    row["han_image" if dirname == "han" else "viet_image"] = str(dest.relative_to(output).as_posix())
                if idx % 50 == 0 or idx == len(all_docs):
                    print(f"Exported provisional crop pairs: {idx}/{len(all_docs)}")
        write_csv(output / "page_manifest.csv", pages, PAGE_COLS)
        write_csv(output / "document_manifest.csv", all_docs, DOCUMENT_COLS)
        write_csv(output / "non_content_pages.csv", removed, SKIP_COLS)
        summary = {
            "source_pdf": str(pdf_path),
            "source_step1b_v2": str(v2),
            "pdf_pages": len(pdf),
            "non_content_pages": len(removed),
            "body_pages": len(pdf) - len(removed),
            "exception_pages": len(overridden),
            "provisional_decree_pairs": len(all_docs),
            "exported_crop_pairs": len(all_docs) if args.export_crops else 0,
            "unverified_geometry": True,
            "verified_ocr_transcriptions": 0,
            "checked_image_pages": preview_pages,
            "warning": "All pairs are provisional; these figures are not OCR or translation accuracy."
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Step 1C complete. PDF pages: {len(pdf)}; excluded non-content: {len(removed)}")
        print(f"Body pages: {len(pdf) - len(removed)}; exception pages: {len(overridden)}")
        print(f"Provisional Hán/Viet pairs: {len(all_docs)}")
        print(f"Saved: {output}")
        print("No OCR labels, Step1B manifests, or model outputs were modified.")
    finally:
        pdf.close()


if __name__ == "__main__":
    main()
