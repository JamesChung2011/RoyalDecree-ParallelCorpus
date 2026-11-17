"""
Step 1: Split PDF into individual page images.
Step 2: Classify images into folders (ch/, vi/, blank/, misc/, intro/, toc/, ref/).

Usage:
    python -m steps.step1_split_and_classify
"""

import os
import sys
import yaml
import pymupdf
from tqdm import tqdm
from pathlib import Path


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_page(page_num: int, cfg: dict) -> str:
    """Determine which folder a page belongs to (1-indexed page_num)."""
    pages = cfg["pages"]

    if page_num in pages.get("blank", []):
        return "blank"
    if page_num in pages.get("misc", []):
        return "misc"
    if page_num in pages.get("intro", []):
        return "intro"
    if page_num in pages.get("toc", []):
        return "toc"
    if page_num in pages.get("ref", []):
        return "ref"
    if pages["vi_start"] <= page_num <= pages["vi_end"]:
        return "vi"
    if pages["ch_start"] <= page_num <= pages["ch_end"]:
        return "ch"

    # Auto-detect blank pages for unclassified pages
    return "blank"


def is_blank_page(page: pymupdf.Page, threshold: int = 10) -> bool:
    """Check if a page is blank (very little text and mostly white pixels)."""
    text = page.get_text().strip()
    if len(text) <= threshold:
        # Also check pixel content
        pix = page.get_pixmap(dpi=72)  # Low DPI for speed
        samples = pix.samples
        # Count non-white pixels
        total = len(samples) // pix.n
        white_count = sum(
            1 for i in range(0, len(samples), pix.n)
            if all(samples[i + c] > 240 for c in range(min(pix.n, 3)))
        )
        white_ratio = white_count / total if total > 0 else 1.0
        return white_ratio > 0.98
    return False


def split_and_classify(config_path="config.yaml"):
    cfg = load_config(config_path)
    book_name = cfg["book_name"]
    pdf_path = cfg["pdf_path"]
    dpi = cfg["ocr"]["dpi"]
    output_dir = Path(cfg["output_dir"]) / book_name

    # Create all category folders
    categories = ["ch", "vi", "intro", "ref", "toc", "blank", "misc"]
    for cat in categories:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count
    print(f"Book: {book_name}")
    print(f"Total pages: {total_pages}")
    print(f"Output: {output_dir}")
    print(f"DPI: {dpi}")
    print()

    stats = {cat: 0 for cat in categories}

    for page_idx in tqdm(range(total_pages), desc="Splitting & classifying"):
        page_num = page_idx + 1  # 1-indexed
        page = doc[page_idx]

        # Classify
        category = classify_page(page_num, cfg)

        # For unclassified pages that fell through to "blank", verify
        if category == "blank" and page_num not in cfg["pages"].get("blank", []):
            if not is_blank_page(page):
                # If not actually blank, it's probably misc
                category = "misc"

        # Generate image filename: <book_name>.<NNN>.jpg
        img_name = f"{book_name}.{page_num:03d}.jpg"
        img_path = output_dir / category / img_name

        # Skip if already exists (resume support)
        if img_path.exists():
            stats[category] += 1
            continue

        # Render page to image with memory-safe approach
        try:
            pix = page.get_pixmap(dpi=dpi)
        except Exception:
            # Fall back to lower DPI if memory fails
            try:
                pix = page.get_pixmap(dpi=200)
                print(f"\n  WARNING: Page {page_num} rendered at 200 DPI (memory limit)")
            except Exception:
                pix = page.get_pixmap(dpi=150)
                print(f"\n  WARNING: Page {page_num} rendered at 150 DPI (memory limit)")
        pix.save(str(img_path))
        pix = None  # Free memory immediately

        stats[category] += 1

    doc.close()

    print("\n=== Classification Summary ===")
    for cat, count in sorted(stats.items()):
        print(f"  {cat:8s}: {count:4d} pages")
    print(f"  {'TOTAL':8s}: {sum(stats.values()):4d} pages")

    return stats


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    split_and_classify(config_path)
