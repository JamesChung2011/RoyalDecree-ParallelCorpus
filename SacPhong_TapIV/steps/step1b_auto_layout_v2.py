"""Step 1B v2: corrected page-layout preflight for Sắc phong (Tập IV).

Only PROPOSES regions. No OCR or verified transcript is produced.
Does not modify Step 1 crops, OCR labels, dictionaries, or candidate results.

Run from project root:
    python -m steps.step1b_auto_layout_v2
    python -m steps.step1b_auto_layout_v2 --all
    python -m steps.step1b_auto_layout_v2 --all --export-crops

The six built-in sample PDF pages were chosen from the current project:
6,7,8 (original pilot), 83 (printed page 82), 92 (printed page 91),
129 (printed page 128). PDF page numbers are 1-based.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import yaml
from PIL import Image, ImageDraw

SAMPLE_PAGES = (6, 7, 8, 83, 92, 129)
PAGE_COLUMNS = [
    "pdf_page", "layout", "estimated_documents", "status", "flags",
    "right_ink_fraction", "image_runs", "render_width", "render_height",
]
DOCUMENT_COLUMNS = [
    "auto_id", "pdf_page", "document_number", "layout", "status",
    "han_bbox_normalized", "viet_bbox_normalized", "han_image", "viet_image",
    "notes",
]


def _csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pdf_to_bgr(page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)


def _detect_runs(mask: np.ndarray, x0: float, x1: float) -> list[tuple[float, float]]:
    h, w = mask.shape
    left, right = round(x0 * w), round(x1 * w)
    row_fraction = mask[:, left:right].mean(axis=1)
    kernel_size = max(5, int(round(h * 0.022)) | 1)
    row_fraction = np.convolve(row_fraction, np.ones(kernel_size) / kernel_size, mode="same")
    active = row_fraction > 0.38
    active[:round(0.105 * h)] = False
    active[round(0.875 * h):] = False
    min_height = round(h * 0.075)
    result = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_height:
                result.append((start / h, i / h))
            start = None
    if start is not None and h - start >= min_height:
        result.append((start / h, 1.0))
    return result


def _clamp_box(rect: tuple[float, float, float, float]) -> list[float]:
    x0, y0, x1, y1 = rect
    x0, x1 = max(0.0, x0), min(1.0, x1)
    y0, y1 = max(0.0, y0), min(1.0, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid rectangle {rect}")
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def _horizontal_bounds(mask: np.ndarray, y0: float, y1: float, layout: str) -> tuple[float, float]:
    h, w = mask.shape
    xa, xb = (0.035, 0.615) if layout == "side_by_side" else (0.035, 0.975)
    left, right = round(xa * w), round(xb * w)
    top, bottom = round(y0 * h), round(y1 * h)
    if bottom <= top + 3:
        return xa, xb
    col_density = mask[top:bottom, left:right].mean(axis=0)
    k = max(5, int(w * 0.014) | 1)
    col_density = np.convolve(col_density, np.ones(k) / k, mode="same")
    present = col_density > 0.42
    # Find the widest dense image span. Thin Vietnamese text near the
    # boundary must not extend the historical document's crop.
    spans = []
    start = None
    for pos, occupied in enumerate(present):
        if occupied and start is None:
            start = pos
        elif not occupied and start is not None:
            spans.append((start, pos))
            start = None
    if start is not None:
        spans.append((start, len(present)))
    spans = [pair for pair in spans if pair[1] - pair[0] > 0.22 * (right - left)]
    if not spans:
        return xa, xb
    a, b = max(spans, key=lambda span: span[1] - span[0])
    derived_start = (left + a) / w
    derived_end = (left + b) / w
    if layout == "side_by_side":
        # Keep the left margin of torn documents; never intrude into the
        # printed Vietnamese section beginning around x=0.61.
        return max(0.035, min(0.060, derived_start - .010)), min(.595, max(.555, derived_end + .012))
    # V2 FIX: stacked decree photographs span almost the whole page width.
    # A dense-span estimate cropped out the right half on PDF page 7.
    return 0.035, 0.970


def _text_region_bottom(image: np.ndarray) -> float:
    """Keep page-number/footer bars out of Viet provisional crops.

    Many pages have a saturated red strip near the bottom. Use its
    first visible row as a conservative bound, not as OCR text.
    If none is detected, fall back to a 94% page-height ceiling.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    hue, saturation, value = cv2.split(hsv)
    red = (((hue < 12) | (hue > 173)) & (saturation > 130) & (value > 95))
    row_pixels = red.sum(axis=1)
    min_pixels = max(30, round(width * 0.018))
    start = round(height * 0.85)
    footer_rows = np.where(row_pixels[start:] > min_pixels)[0]
    if not len(footer_rows):
        return 0.940
    red_top = (start + int(footer_rows[0])) / height
    return max(0.86, min(0.940, red_top - 0.012))


def analyze_page(image: np.ndarray, pdf_page: int) -> tuple[dict, list[dict]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Strong color OR gray-on-white ink. Small printed letters on the right
    # are usually too sparse to pass the row-density test.
    mask = ((hsv[:, :, 1] > 25) & (hsv[:, :, 2] < 246)) | (gray < 205)
    right_fraction = float(mask[round(h * .15):round(h * .38),
                                round(w * .70):round(w * .93)].mean())
    layout = "stacked" if right_fraction >= .47 else "side_by_side"
    runs = _detect_runs(mask, 0.09, 0.44)
    flags = []
    if .35 < right_fraction < .61:
        flags.append("LAYOUT_BORDERLINE")
    if not runs:
        flags.append("NO_CLEAR_DECREE_REGION")
    if len(runs) > 2:
        flags.append("MORE_THAN_TWO_REGIONS")
    if layout == "stacked" and len(runs) != 1:
        flags.append("STACKED_REGION_COUNT_UNEXPECTED")
    if layout == "stacked" and runs and runs[-1][1] > .82:
        flags.append("NO_ROOM_FOR_TRANSLATION")
    if layout == "side_by_side" and len(runs) not in (1, 2):
        flags.append("SIDE_LAYOUT_REGION_COUNT_UNEXPECTED")
    if any((end - start) < .12 for start, end in runs):
        flags.append("REGION_TOO_SHORT")
    status = "REVIEW_LAYOUT" if flags else "PROVISIONAL_CROPS"
    page_record = {
        "pdf_page": pdf_page, "layout": layout,
        "estimated_documents": len(runs), "status": status,
        "flags": "; ".join(flags),
        "right_ink_fraction": round(right_fraction, 3),
        "image_runs": json.dumps([[round(a, 3), round(b, 3)] for a,b in runs]),
        "render_width": w, "render_height": h,
    }
    docs = []
    viet_bottom = _text_region_bottom(image)
    if status != "PROVISIONAL_CROPS":
        return page_record, docs
    for i, (start, end) in enumerate(runs):
        x0, x1 = _horizontal_bounds(mask, start, end, layout)
        han_bbox = _clamp_box((x0, start - .008, x1, end + (.030 if layout == "stacked" else .008)))
        if layout == "side_by_side":
            # Keep printed footnotes inside the draft (separate them later).
            # The old .875 limit clipped lower notes on some pages.
            next_start = runs[i + 1][0] if i + 1 < len(runs) else .950
            viet_bbox = _clamp_box((.585, start - .010, .985,
                                    min(viet_bottom, next_start - .010)))
        else:
            # On long stacked pages, translations extend below .885.
            viet_bbox = _clamp_box((.050, end + .040, .985, viet_bottom))
        docs.append({
            "auto_id": f"AUTO.{pdf_page:03d}.{i + 1:02d}",
            "pdf_page": pdf_page, "document_number": i + 1,
            "layout": layout, "status": "PROVISIONAL_UNVERIFIED",
            "han_bbox_normalized": han_bbox,
            "viet_bbox_normalized": viet_bbox,
            "han_image": "", "viet_image": "",
            "notes": ("V2 provisional bounds: full-width Hán on stacked pages; "
                      "Vietnamese region may contain footnotes; verify original "
                      "text, reading order, and pair count"),
        })
    return page_record, docs


def _draw_preview(image: np.ndarray, docs: list[dict], page_record: dict, destination: Path) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(rgb)
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    w, h = preview.size
    for doc in docs:
        for field, color in (("han_bbox_normalized", (20, 105, 250)),
                             ("viet_bbox_normalized", (10, 160, 60))):
            a, b, c, d = doc[field]
            draw.rectangle((a*w,b*h,c*w,d*h), outline=color, width=3)
        x, y = doc["han_bbox_normalized"][:2]
        draw.text((x*w+3, max(0,y*h-15)), doc["auto_id"], fill=(0,0,180))
    draw.text((10, 8), f"PDF {page_record['pdf_page']} | {page_record['layout']} | {page_record['status']}",
              fill=(0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination, quality=88)


def _crop_save(page, bbox: list[float], dpi: int, destination: Path) -> None:
    width, height = page.rect.width, page.rect.height
    x0, y0, x1, y1 = bbox
    rect = pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72), clip=rect, alpha=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pix.save(destination)


def main() -> None:
    p = argparse.ArgumentParser(description="Propose page layout/crop regions without OCR or label overwrite")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--pages", type=int, nargs="+", default=None, help="PDF pages, 1-based")
    p.add_argument("--all", action="store_true", help="Scan every PDF page; non-content pages may be flagged")
    p.add_argument("--dpi", type=int, default=96, help="Low-resolution scan dpi, default 96")
    p.add_argument("--crop-dpi", type=int, default=200, help="Crop export dpi")
    p.add_argument("--export-crops", action="store_true", help="Save draft crops only for unflagged page layouts")
    p.add_argument("--force", action="store_true", help="Replace earlier Step1B results for this scope")
    args = p.parse_args()
    if args.all and args.pages:
        p.error("Choose either --all or --pages")
    if args.dpi < 60 or args.dpi > 220:
        p.error("--dpi must be between 60 and 220")
    root = Path.cwd()
    cfg_file = (root / args.config).resolve()
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        p.error("Invalid config.yaml")
    source = Path(cfg.get("pdf_path", "source.pdf"))
    if not source.is_absolute(): source = cfg_file.parent / source
    if not source.is_file():
        p.error(f"PDF not found: {source}")
    output_root = root / cfg.get("output_dir", "output") / cfg.get("book_name", "Sac_phong_Tap_IV")
    scope = "full_book" if args.all else ("custom" if args.pages else "six_page_pilot")
    directory = output_root / "01_layout_auto_draft_v2" / scope
    if directory.exists() and any(directory.iterdir()) and not args.force:
        p.error(f"Existing results at {directory}. Move them aside or add --force")
    directory.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(source)
    pages = list(range(1, len(doc) + 1)) if args.all else (args.pages or list(SAMPLE_PAGES))
    if not pages or any(page < 1 or page > len(doc) for page in pages):
        p.error(f"Page number must be between 1 and {len(doc)}")
    page_rows = []
    doc_rows = []
    review_rows = []
    selected_previews = set(args.pages or SAMPLE_PAGES)
    for number in pages:
        page = doc[number - 1]
        image = _pdf_to_bgr(page, args.dpi)
        page_row, docs = analyze_page(image, number)
        page_rows.append(page_row)
        if page_row["status"] == "REVIEW_LAYOUT":
            review_rows.append(page_row)
        if number in selected_previews or not args.all or page_row["status"] == "REVIEW_LAYOUT":
            _draw_preview(image, docs, page_row, directory / "previews" / f"page_{number:03d}_annotated.jpg")
        if args.export_crops:
            for item in docs:
                name = item["auto_id"]
                han_file = directory / "draft_crops" / "han" / f"{name}_1.png"
                viet_file = directory / "draft_crops" / "viet" / f"{name}_1.png"
                _crop_save(page, item["han_bbox_normalized"], args.crop_dpi, han_file)
                _crop_save(page, item["viet_bbox_normalized"], args.crop_dpi, viet_file)
                item["han_image"] = han_file.relative_to(directory).as_posix()
                item["viet_image"] = viet_file.relative_to(directory).as_posix()
        doc_rows.extend(docs)
        if len(pages) < 15 or number % 25 == 0:
            print(f"PDF page {number:3d}: {page_row['layout']:12s} {page_row['status']:17s} {len(docs)} draft pair(s)")
    _csv_write(directory / "page_manifest.csv", page_rows, PAGE_COLUMNS)
    _csv_write(directory / "document_manifest.csv", doc_rows, DOCUMENT_COLUMNS)
    _csv_write(directory / "review_queue.csv", review_rows, PAGE_COLUMNS)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": source.name,
        "pdf_pages_total": len(doc), "pdf_pages_scanned": len(page_rows),
        "provisional_document_pairs": len(doc_rows),
        "pages_needing_layout_review": len(review_rows),
        "no_verified_transcriptions": True,
        "no_ocr_performed": True,
        "notes": "Layout-only preflight. Draft crops are not confirmed Hán–Phiên âm–Dịch nghĩa pairs. Manual review of flags and representative pages required.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nLAYOUT PREFLIGHT DONE")
    for key, value in summary.items():
        if key != "notes": print(f"  {key}: {value}")
    print("Output:", directory)
    print("Source OCR labels, Step 1 crops, and prior model results NOT changed.")
    print("V2 conservative crops are UNVERIFIED; inspect before batch OCR.")


if __name__ == "__main__":
    main()
