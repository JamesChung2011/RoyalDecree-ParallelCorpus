"""
Step 5: Evaluate alignment quality.

Computes similarity percentage between Nôm characters and Quốc Ngữ text
per the project spec formula:

    similarity (%) = (1 - total_red_chars / total_chars) × 100

Target: ≥ 95%.

Usage:
    python -m steps.step5_evaluate [--config config.yaml]
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_meaningful_chars(text: str) -> int:
    if not text:
        return 0
    return len(text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", ""))


def evaluate_df(df: pd.DataFrame) -> dict:
    total_pairs = len(df)
    total_nom_chars = 0
    total_vn_chars = 0
    red_chars = 0
    matched_pairs = 0
    empty_nom = 0
    empty_vn = 0

    for _, row in df.iterrows():
        nom = str(row.get("SinoNom_Char", "")) if pd.notna(row.get("SinoNom_Char")) else ""
        vn = str(row.get("ChuQN_txt", "")) if pd.notna(row.get("ChuQN_txt")) else ""

        nom = nom.strip()
        vn = vn.strip()

        nom_chars = count_meaningful_chars(nom)
        vn_chars = count_meaningful_chars(vn)
        total_nom_chars += nom_chars
        total_vn_chars += vn_chars

        if not nom:
            empty_nom += 1
        if not vn:
            empty_vn += 1

        if nom and vn:
            matched_pairs += 1
        else:
            red_chars += max(nom_chars, vn_chars)

    total_chars = total_nom_chars + total_vn_chars
    similarity = (1 - red_chars / total_chars) * 100 if total_chars > 0 else 0

    return {
        "total_pairs": total_pairs,
        "matched_pairs": matched_pairs,
        "empty_nom": empty_nom,
        "empty_vn": empty_vn,
        "total_nom_chars": total_nom_chars,
        "total_vn_chars": total_vn_chars,
        "total_chars": total_chars,
        "red_chars": red_chars,
        "similarity_pct": round(similarity, 2),
        "passes": similarity >= 95.0,
    }


def evaluate_alignment(cfg: dict) -> dict:
    book_name = cfg["book_name"]
    output_dir = Path(cfg["output_dir"]) / book_name
    csv_path = output_dir / f"{book_name}.csv"

    if not csv_path.exists():
        print(f"Alignment file not found: {csv_path}")
        return {}

    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    results = evaluate_df(df)

    # Check if sentence-level alignment exists and evaluate it too
    sent_csv_path = output_dir / f"{book_name}_sentences.csv"
    if sent_csv_path.exists():
        sent_df = pd.read_csv(str(sent_csv_path), encoding="utf-8-sig")
        sent_results = evaluate_df(sent_df)
        results["sentences"] = sent_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment quality")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate_alignment(cfg)

    if not results:
        return

    report_text = f"""==================================================
ALIGNMENT EVALUATION REPORT (WOODBLOCK LEVEL)
==================================================
  Total pairs:       {results['total_pairs']}
  Matched pairs:     {results['matched_pairs']}
  Empty Nôm:         {results['empty_nom']}
  Empty Quốc Ngữ:    {results['empty_vn']}
  Total Nôm chars:   {results['total_nom_chars']}
  Total QN chars:    {results['total_vn_chars']}
  Total chars:       {results['total_chars']}
  Red chars:         {results['red_chars']}
  Similarity:        {results['similarity_pct']}%
  Passes (≥95%):     {'✓ YES' if results['passes'] else '✗ NO'}
==================================================
"""
    if "sentences" in results:
        sr = results["sentences"]
        report_text += f"""
==================================================
SENTENCE-LEVEL SUB-DIVIDED ALIGNMENT REPORT
==================================================
  Total pairs:       {sr['total_pairs']}
  Matched pairs:     {sr['matched_pairs']}
  Empty Nôm:         {sr['empty_nom']}
  Empty Quốc Ngữ:    {sr['empty_vn']}
  Total Nôm chars:   {sr['total_nom_chars']}
  Total QN chars:    {sr['total_vn_chars']}
  Total chars:       {sr['total_chars']}
  Red chars:         {sr['red_chars']}
  Similarity:        {sr['similarity_pct']}%
  Passes (≥95%):     {'✓ YES' if sr['passes'] else '✗ NO'}
==================================================
"""
    print("\n" + report_text)

    book_name = cfg["book_name"]
    output_dir = Path(cfg["output_dir"]) / book_name
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "evaluation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  Report saved to:")
    print(f"    - {txt_path}")
    print(f"    - {json_path}")


if __name__ == "__main__":
    main()
