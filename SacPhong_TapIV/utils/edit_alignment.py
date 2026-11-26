"""
Interactive Alignment Inspector & Manual Adjustment Tool.

Allows inspecting, adjusting boundaries, and overriding Vietnamese translations
for any woodblock page where automated OCR/regex marker detection was slightly off.

Usage:
    # 1. Inspect a specific page and its neighbors
    python -m utils.edit_alignment --inspect 240

    # 2. Set/override text for a specific page
    python -m utils.edit_alignment --set 240 "Văn bản dịch tiếng Việt chính xác..."

    # 3. Shift words across page boundaries (e.g. shift 5 words from 240 to 241)
    python -m utils.edit_alignment --shift 240 --words 5

    # 4. List all active manual overrides
    python -m utils.edit_alignment --list-overrides

    # 5. Remove an override
    python -m utils.edit_alignment --clear 240

    # 6. Apply all overrides and update CSV, XLSX, and evaluation
    python -m utils.edit_alignment --apply
"""

import os
import sys
import re
import yaml
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from steps.step4_align import (
    load_config, build_alignment, save_alignment,
    extract_page_num_from_filename
)
from steps.step5_evaluate import evaluate_alignment

OVERRIDES_FILE = "alignment_overrides.yaml"


def load_overrides_dict() -> dict:
    if os.path.exists(OVERRIDES_FILE):
        try:
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def save_overrides_dict(d: dict):
    with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
        f.write("# ==============================================================================\n")
        f.write("# MANUAL ALIGNMENT OVERRIDES\n")
        f.write("# Edit this file directly or use `python -m utils.edit_alignment`\n")
        f.write("# ==============================================================================\n\n")
        yaml.safe_dump(d, f, allow_unicode=True, default_flow_style=False)


def inspect_page(page_num: int, cfg: dict):
    book_name = cfg["book_name"]
    csv_path = Path(cfg["output_dir"]) / book_name / f"{book_name}.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Please run pipeline.py --step 5 first.")
        return

    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    target_img = f"{book_name}.{page_num:03d}.jpg"
    idx = df[df["img_id"] == target_img].index

    if len(idx) == 0:
        print(f"Page {page_num} ({target_img}) not found in Nôm aligned pages.")
        return

    i = idx[0]
    row = df.iloc[i]
    nom = str(row["SinoNom_Char"])
    vn = str(row["ChuQN_txt"])

    print(f"\n{'='*70}")
    print(f"PAGE INSPECTION: {target_img} (Row {i+1} / {len(df)})")
    print(f"{'='*70}")
    print(f"Nôm characters : {len(nom)}")
    print(f"Quốc ngữ chars : {len(vn)} (approx {len(vn.split())} words)")
    print(f"Length Ratio   : {len(vn)/max(len(nom),1):.2f}")

    overrides = load_overrides_dict()
    is_overridden = (page_num in overrides or str(page_num) in overrides or target_img in overrides)
    print(f"Status         : {'[MANUALLY OVERRIDDEN]' if is_overridden else '[AUTOMATIC ALIGNMENT]'}")

    print(f"\n--- HÁN NÔM TEXT ---")
    print(nom)

    print(f"\n--- QUỐC NGỮ TRANSLATION ---")
    print(vn)

    if i > 0:
        prev_row = df.iloc[i - 1]
        prev_vn = str(prev_row["ChuQN_txt"])
        print(f"\n--- PREVIOUS PAGE ({prev_row['img_id']}) ENDING ---")
        print("..." + prev_vn[-180:] if len(prev_vn) > 180 else prev_vn)

    if i < len(df) - 1:
        next_row = df.iloc[i + 1]
        next_vn = str(next_row["ChuQN_txt"])
        print(f"\n--- NEXT PAGE ({next_row['img_id']}) BEGINNING ---")
        print(next_vn[:180] + "..." if len(next_vn) > 180 else next_vn)

    print(f"{'='*70}\n")


def set_page_text(page_num: int, text: str, cfg: dict):
    text = re.sub(r'\s+', ' ', text).strip()
    overrides = load_overrides_dict()
    overrides[page_num] = text
    save_overrides_dict(overrides)
    print(f"✓ Saved manual override for page {page_num} ({len(text)} chars)")
    apply_overrides(cfg)


def shift_words(page_num: int, word_count: int, cfg: dict):
    """
    word_count > 0: moves word_count words from page_num to page_num + 1.
    word_count < 0: moves |word_count| words from page_num - 1 to page_num.
    """
    book_name = cfg["book_name"]
    csv_path = Path(cfg["output_dir"]) / book_name / f"{book_name}.csv"
    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")

    target_img = f"{book_name}.{page_num:03d}.jpg"
    idx = df[df["img_id"] == target_img].index
    if len(idx) == 0:
        print(f"Page {page_num} not found.")
        return

    i = idx[0]
    overrides = load_overrides_dict()

    if word_count > 0:
        # Move words to next page
        if i >= len(df) - 1:
            print("Cannot shift from last page to next.")
            return
        curr_words = str(df.iloc[i]["ChuQN_txt"]).split()
        next_words = str(df.iloc[i + 1]["ChuQN_txt"]).split()

        if len(curr_words) <= word_count:
            print("Not enough words on current page to shift.")
            return

        shifted = curr_words[-word_count:]
        new_curr = " ".join(curr_words[:-word_count])
        new_next = " ".join(shifted + next_words)

        p_curr = page_num
        p_next = extract_page_num_from_filename(df.iloc[i + 1]["img_id"])

        overrides[p_curr] = new_curr
        overrides[p_next] = new_next

        print(f"✓ Shifted {word_count} words from Page {p_curr} -> Page {p_next}")

    else:
        # Move words from prev page
        if i == 0:
            print("Cannot shift from page before first page.")
            return
        k = abs(word_count)
        prev_words = str(df.iloc[i - 1]["ChuQN_txt"]).split()
        curr_words = str(df.iloc[i]["ChuQN_txt"]).split()

        if len(prev_words) <= k:
            print("Not enough words on previous page to shift.")
            return

        shifted = prev_words[-k:]
        new_prev = " ".join(prev_words[:-k])
        new_curr = " ".join(shifted + curr_words)

        p_prev = extract_page_num_from_filename(df.iloc[i - 1]["img_id"])
        p_curr = page_num

        overrides[p_prev] = new_prev
        overrides[p_curr] = new_curr

        print(f"✓ Shifted {k} words from Page {p_prev} -> Page {p_curr}")

    save_overrides_dict(overrides)
    apply_overrides(cfg)


def clear_page_override(page_num: int, cfg: dict):
    overrides = load_overrides_dict()
    changed = False
    for k in [page_num, str(page_num), f"{cfg['book_name']}.{page_num:03d}.jpg"]:
        if k in overrides:
            del overrides[k]
            changed = True

    if changed:
        save_overrides_dict(overrides)
        print(f"✓ Cleared manual override for page {page_num}")
        apply_overrides(cfg)
    else:
        print(f"No manual override found for page {page_num}")


def list_overrides():
    overrides = load_overrides_dict()
    print(f"\n{'='*60}")
    print(f"ACTIVE MANUAL OVERRIDES ({len(overrides)} pages)")
    print(f"{'='*60}")
    if not overrides:
        print("  (No manual overrides configured. System using automated alignment)")
    for k, v in overrides.items():
        val_str = v if isinstance(v, str) else v.get("ChuQN_txt", "")
        print(f"  • Page {k}: {len(val_str)} chars -> {val_str[:70]}...")
    print(f"{'='*60}\n")


def apply_overrides(cfg: dict):
    print("Re-aligning and updating output files...")
    alignments = build_alignment(cfg)
    save_alignment(alignments, cfg)
    res = evaluate_alignment(cfg)
    if res:
        status_str = "PASS ✓" if res.get('passes') else "FAIL ✗"
        print(f"\n✓ Evaluation result: {res['similarity_pct']}% similarity ({status_str})")


def main():
    parser = argparse.ArgumentParser(description="Inspect & Manually Adjust Sentence Alignment")
    parser.add_argument("--inspect", type=int, help="Inspect page by number (e.g. 240)")
    parser.add_argument("--set", type=int, help="Page number to set text for")
    parser.add_argument("--text", type=str, help="Text to assign with --set")
    parser.add_argument("--shift", type=int, help="Page number to shift boundary for")
    parser.add_argument("--words", type=int, default=5, help="Number of words to shift (>0: to next, <0: from prev)")
    parser.add_argument("--clear", type=int, help="Clear override for page number")
    parser.add_argument("--list-overrides", action="store_true", help="List all active overrides")
    parser.add_argument("--apply", action="store_true", help="Re-apply overrides to CSV, XLSX and evaluate")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.inspect is not None:
        inspect_page(args.inspect, cfg)
    elif args.set is not None:
        if not args.text:
            print("Error: --set requires --text \"...\"")
            return
        set_page_text(args.set, args.text, cfg)
    elif args.shift is not None:
        shift_words(args.shift, args.words, cfg)
    elif args.clear is not None:
        clear_page_override(args.clear, cfg)
    elif args.list_overrides:
        list_overrides()
    elif args.apply:
        apply_overrides(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
