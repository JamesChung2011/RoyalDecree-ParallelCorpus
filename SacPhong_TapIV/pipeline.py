"""
Main pipeline runner for Hán Nôm ↔ Quốc Ngữ parallel corpus building.

Pipeline Steps:
    1. Split PDF → 300 DPI images & classify into structured folders (ch, vi, intro, toc, ref, blank)
    2. OCR / Text Extraction:
       - Hán Nôm (ch): RapidOCR (PP-OCRv4 ONNX), right-to-left vertical columns
       - Quốc Ngữ (vi, intro, toc, ref): VietOCR (VGG-Transformer) or direct PDF text extraction
    3. Historical Nôm Spellcheck & Ink-Bleed Correction (NOM_CONFUSION_MAP)
    4. Woodblock Anchor Parallel Alignment & Dual Export (Leaf-level & Sentence-level CSV/XLSX)
    5. Quality & Similarity Evaluation (≥95% threshold check)

Usage:
    python pipeline.py                     # Run pipeline (steps 1, 2, 3, 4, 5)
    python pipeline.py --step 2            # Run specific step
    python pipeline.py --step 2 --folder vi --force  # Force re-run specific folder
    python pipeline.py --config config.yaml
"""

import sys
import argparse
import time
from pathlib import Path


def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner(book_name: str):
    width = 64
    print("╔" + "═" * width + "╗")
    print(f"║  Hán Nôm ↔ Quốc Ngữ Parallel Corpus Pipeline".center(width) + "║")
    print(f"║  Book: {book_name}".center(width) + "║")
    print("╚" + "═" * width + "╝")


def run_step(step: int, cfg: dict, config_path: str = "config.yaml", **kwargs):
    """Run a single pipeline step with error handling and timing."""
    print(f"\n{'#'*66}")
    print(f"# STEP {step}".ljust(64) + " #")
    print(f"{'#'*66}")
    start = time.time()

    if step == 1:
        from steps.step1_split_and_classify import split_and_classify
        split_and_classify(config_path)

    elif step == 2:
        from steps.step2_ocr import process_folder
        folder = kwargs.get("folder", "all")
        force = kwargs.get("force", False)
        folders = ["ch", "vi", "intro", "toc", "ref"] if folder == "all" else [folder]
        for f in folders:
            process_folder(f, cfg, force=force)

    elif step == 3:
        from steps.step3_spellcheck import run_spellcheck
        run_spellcheck(cfg)

    elif step == 4:
        from steps.step4_align import build_alignment, save_alignment
        alignments = build_alignment(cfg)
        save_alignment(alignments, cfg)

    elif step == 5:
        from steps.step5_evaluate import evaluate_alignment
        results = evaluate_alignment(cfg)
        if results:
            sim = results.get("similarity_pct", 0)
            status = "PASS ✓" if results.get("passes") else "FAIL ✗"
            print(f"\n  Final Alignment Similarity: {sim}%")
            print(f"  Quality Status:             {status}")

    else:
        print(f"Error: Unknown step {step}")
        return

    elapsed = time.time() - start
    print(f"\n  Step {step} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Hán Nôm ↔ Quốc Ngữ Parallel Corpus Pipeline"
    )
    parser.add_argument("--step", type=int, default=0,
                        help="Run specific step (1, 2, 3, 4, 5). Default 0 runs all steps.")
    parser.add_argument("--folder", default="all",
                        choices=["ch", "vi", "intro", "toc", "ref", "all"],
                        help="For Step 2 OCR: which folder to process")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running and overwrite existing label caches")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    book_name = cfg.get("book_name", "Unknown Book")

    print_banner(book_name)

    if args.step > 0:
        run_step(args.step, cfg, args.config, folder=args.folder, force=args.force)
    else:
        all_steps = [1, 2, 3, 4, 5]
        total_start = time.time()
        for step in all_steps:
            run_step(step, cfg, args.config, folder=args.folder, force=args.force)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*66}")
        print(f"  Pipeline execution completed in {total_elapsed/60:.2f} minutes!")
        print(f"{'='*66}")


if __name__ == "__main__":
    main()
