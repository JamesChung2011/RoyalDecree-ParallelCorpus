"""
Step 3: Spellcheck and Clean Annotations for Hán Nôm (ch/).

Applies historical Nôm dictionary corrections for ch/ using NOM_CONFUSION_MAP:
- Corrects woodblock ink bleed and wood grain confusion errors (e.g. 秦 -> 奏, 栈 -> 機, 豆 -> 臣).
- Vietnamese spellchecking is completely removed: Vietnamese text is extracted pure
  and verbatim directly from the original PDF text layer.

Usage:
    python -m steps.step3_spellcheck [--config config.yaml]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.spellcheck import spellcheck_nom_text
from utils.text_cleaning import is_noise_box


def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spellcheck_folder(folder_name: str, cfg: dict):
    book_name = cfg["book_name"]
    folder_path = Path(cfg["output_dir"]) / book_name / folder_name
    label_path = folder_path / "Label.txt"

    if not label_path.exists():
        print(f"Label.txt not found in {folder_path}")
        return 0, 0, 0

    print(f"\n{'='*60}")
    print(f"Hán Nôm Woodblock Spellchecking: {folder_name}/Label.txt")
    print(f"{'='*60}")

    updated_lines = []
    total_boxes = 0
    total_corrections = 0
    removed_noise_boxes = 0

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip()]

    for line in tqdm(lines, desc=f"Processing {folder_name}"):
        parts = line.split("\t", 1)
        if len(parts) < 2:
            updated_lines.append(line)
            continue

        img_ref = parts[0]
        try:
            boxes = json.loads(parts[1])
        except json.JSONDecodeError:
            updated_lines.append(line)
            continue

        kept_boxes = []

        for box in boxes:
            total_boxes += 1
            raw_text = box.get("transcription", "")

            if is_noise_box(raw_text):
                removed_noise_boxes += 1
                continue

            corrected_text, fixes = spellcheck_nom_text(raw_text)
            total_corrections += fixes

            box["transcription"] = corrected_text
            kept_boxes.append(box)

        new_json = json.dumps(kept_boxes, ensure_ascii=False)
        updated_lines.append(f"{img_ref}\t{new_json}")

    # Write back cleaned Label.txt
    with open(label_path, "w", encoding="utf-8") as f:
        for l in updated_lines:
            f.write(l + "\n")

    print(f"✓ {folder_name}/: {total_corrections} Hán Nôm character fixes, {removed_noise_boxes} noise boxes removed.")
    return total_boxes, total_corrections, removed_noise_boxes


def run_spellcheck(cfg: dict):
    # Only spellcheck Hán Nôm (ch/)
    # Vietnamese spellchecking is removed: Vietnamese text is extracted pure from PDF.
    folders = ["ch"]
    summary = {}
    for f in folders:
        boxes, fixes, removed = spellcheck_folder(f, cfg)
        summary[f] = {"boxes": boxes, "fixes": fixes, "removed": removed}

    print("\n" + "=" * 60)
    print("HÁN NÔM SPELLCHECK SUMMARY")
    print("=" * 60)
    for folder, data in summary.items():
        print(f"  {folder:8s}: {data['fixes']:4d} character fixes across {data['boxes']:5d} boxes")
    print("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Hán Nôm woodblock spellcheck")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_spellcheck(cfg)


if __name__ == "__main__":
    main()
