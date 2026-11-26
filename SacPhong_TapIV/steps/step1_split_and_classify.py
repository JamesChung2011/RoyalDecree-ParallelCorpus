"""Step 1 for Sắc phong Tập IV: split mixed-layout PDF into document crops.

Run from the project root:
    python pipeline.py --step 1

This prototype processes only the pages listed under page_regions in config.yaml.
It does NOT perform OCR or certify that crops/alignments are correct.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pymupdf
import yaml
from PIL import Image, ImageDraw, ImageFont


def _load_config(config_path: str) -> dict:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must contain a YAML mapping.")
    if cfg.get("layout_type") != "SAC_PHONG_MIXED":
        raise ValueError("This Step 1 patch expects layout_type: SAC_PHONG_MIXED.")
    if "page_regions" not in cfg or not cfg["page_regions"]:
        raise ValueError("Set page_regions in config.yaml before running Step 1.")
    cfg["_project_dir"] = path.parent
    return cfg


def _resolve_path(project_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_dir / p


def _bbox_pixels(rect: list, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        raise ValueError(f"Expected four normalized bbox coordinates, got: {rect}")
    a, b, c, d = [float(v) for v in rect]
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        raise ValueError(f"Invalid normalized bbox {rect}: need 0 <= x0 < x1 <= 1")
    x0, y0 = round(a * width), round(b * height)
    x1, y1 = round(c * width), round(d * height)
    return x0, y0, x1, y1


def _save_crop(im: Image.Image, bbox: tuple[int, int, int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = im.crop(bbox)
    if crop.width < 40 or crop.height < 40:
        raise ValueError(f"Crop too small: {path}, size={crop.size}")
    crop.save(path, "JPEG", quality=94, optimize=True)


def _make_preview(im: Image.Image, annotated: list[dict], path: Path) -> None:
    # Preview is intentionally resized for quick, visual manual verification.
    preview = im.copy()
    preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    sx = preview.width / im.width
    sy = preview.height / im.height
    draw = ImageDraw.Draw(preview)
    for item in annotated:
        scale_rect = tuple(round(x * sx) if i % 2 == 0 else round(x * sy)
                           for i, x in enumerate(item["bbox"]))
        outline = "#1a53ff" if item["kind"] == "han" else "#0a9c48"
        draw.rectangle(scale_rect, outline=outline, width=4)
        tx = scale_rect[0] + 4
        ty = max(2, scale_rect[1] - 21)
        label = f'{item["doc_id"]} {item["kind"]}'
        draw.rectangle((tx, ty, tx + 130, ty + 19), fill="white")
        draw.text((tx + 2, ty + 1), label, fill=outline)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, "JPEG", quality=90)


def split_and_classify(config_path: str = "config.yaml") -> dict:
    """Return processing counts; compatibility with the existing pipeline.py."""
    cfg = _load_config(config_path)
    project_dir: Path = cfg["_project_dir"]
    pdf_path = _resolve_path(project_dir, str(cfg.get("pdf_path", "source.pdf")))
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Copy 'Bản sao của tap 4.pdf' into the project folder and name the COPY 'source.pdf'."
        )

    output_root = _resolve_path(project_dir, str(cfg.get("output_dir", "output")))
    book_output = output_root / str(cfg.get("book_name", "Sac_phong_Tap_IV"))
    han_dir = book_output / "02_ocr_tho" / "han"
    vn_dir = book_output / "02_ocr_tho" / "viet"
    previews = book_output / "00_preview"
    for d in (han_dir, vn_dir, previews):
        d.mkdir(parents=True, exist_ok=True)

    regions = cfg["page_regions"]
    pages = sorted((int(p), blocks) for p, blocks in regions.items())
    dpi = int(cfg.get("ocr", {}).get("dpi", 220))
    if not (72 <= dpi <= 600):
        raise ValueError(f"OCR dpi must be 72-600. Got {dpi}.")
    prefix = str(cfg.get("book_code", "PILOT")).strip()
    if not prefix or "." in prefix:
        raise ValueError("book_code must be a non-empty identifier without dots")

    summary = {"pages": 0, "documents": 0, "han_crops": 0, "viet_crops": 0}
    manifest_rows = []
    with pymupdf.open(pdf_path) as pdf:
        print(f"Sắc phong PDF pages: {len(pdf)}")
        print(f"Step 1 pilot pages: {[page for page, _ in pages]}")
        for page_num, items in pages:
            if not (1 <= page_num <= len(pdf)):
                raise ValueError(f"PDF page {page_num} out of range 1..{len(pdf)}")
            if not isinstance(items, list) or not items:
                raise ValueError(f"page_regions[{page_num}] needs one or more document entries")
            pdf_page = pdf[page_num - 1]
            pix = pdf_page.get_pixmap(dpi=dpi, alpha=False)
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            full_page_path = previews / f"page_{page_num:03d}.jpg"
            im.save(full_page_path, "JPEG", quality=88)
            drawn = []
            for entry_idx, entry in enumerate(items, start=1):
                doc_id = f"{prefix}.{page_num:03d}.{entry_idx:02d}"
                for required in ("han", "viet"):
                    if required not in entry:
                        raise ValueError(f"Missing '{required}' crop in page {page_num}, entry {entry_idx}")
                    bbox = _bbox_pixels(entry[required], im.width, im.height)
                    outdir = han_dir if required == "han" else vn_dir
                    filename = f"{doc_id}_1.jpg"
                    _save_crop(im, bbox, outdir / filename)
                    drawn.append({"bbox": bbox, "doc_id": doc_id, "kind": required})
                    summary["han_crops" if required == "han" else "viet_crops"] += 1
                manifest_rows.append({
                    "doc_id": doc_id,
                    "pdf_page": page_num,
                    "decree_on_page": entry_idx,
                    "han_image": f"02_ocr_tho/han/{doc_id}_1.jpg",
                    "viet_image": f"02_ocr_tho/viet/{doc_id}_1.jpg",
                    "status": "NEEDS_VISUAL_REVIEW",
                    "notes": "Pilot region; verify image-to-text pairing and region borders",
                    "han_bbox_normalized": json.dumps(entry["han"]),
                    "viet_bbox_normalized": json.dumps(entry["viet"]),
                })
                summary["documents"] += 1
            _make_preview(im, drawn, previews / f"page_{page_num:03d}_annotated.jpg")
            summary["pages"] += 1
            print(f"  page {page_num}: {len(items)} document(s) cropped")

    manifest_path = book_output / "00_preview" / "crop_manifest.csv"
    fieldnames = ["doc_id", "pdf_page", "decree_on_page", "han_image", "viet_image",
                  "status", "notes", "han_bbox_normalized", "viet_bbox_normalized"]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nSTEP 1 COMPLETED (PILOT ONLY)")
    print(f"  {summary['pages']} PDF pages; {summary['documents']} provisional documents")
    print(f"  Inspect previews: {previews}")
    print(f"  Manifest:         {manifest_path}")
    print("  No OCR, translation or alignment has been performed.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split mixed-layout royal decrees into image crops")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    split_and_classify(args.config)


if __name__ == "__main__":
    main()
