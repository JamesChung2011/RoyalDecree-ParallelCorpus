r"""Optional, isolated Chinese OCR column-level pilot for Sắc phong Tập IV.

No changes to existing pipeline, source.pdf, Label.txt, or Step 2 outputs.
Run at project root:
    .\.venv\Scripts\python.exe -m steps.step2c_region_probe
    .\.venv\Scripts\python.exe -m steps.step2c_region_probe --visual-only

This is a *research experiment*: bounding boxes are tentative and machine text
is unverified. No predicted Chinese character is converted into a gold label.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_DOCS = ("PILOT.006.01", "PILOT.008.02")


def _read_settings(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "documents" not in data or not isinstance(data["documents"], dict):
        raise ValueError("step2c_regions.json is missing 'documents'")
    return data


def _project_root(cfg_path: Path) -> tuple[Path, Path]:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.resolve().parent
    out = base / str(cfg.get("output_dir", "output")) / str(cfg["book_name"])
    return base, out


def _find_image(out: Path, doc_id: str) -> Path:
    manifest = out / "00_preview" / "crop_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing crop manifest: {manifest}. Run Step 1 first.")
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        matches = [r for r in csv.DictReader(f) if (r.get("doc_id") or "").strip() == doc_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest row for {doc_id}, found {len(matches)}")
    rel = Path(matches[0]["han_image"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe path in crop manifest: {rel}")
    image_path = out / rel
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing Chinese crop: {image_path}")
    return image_path


def _orient(im: Image.Image, direction: str) -> Image.Image:
    if direction == "original":
        return im.copy()
    if direction == "rotate_90_cw":
        return im.transpose(Image.Transpose.ROTATE_270)
    if direction == "rotate_90_ccw":
        return im.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(f"Unsupported orientation: {direction!r}")


def _rect(box: list[int], ref_size: tuple[int, int], new_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"Expected four rectangle coordinates, got: {box}")
    rw, rh = ref_size
    nw, nh = new_size
    if min(rw, rh, nw, nh) <= 0:
        raise ValueError("Image size must be positive")
    x0 = max(0, min(nw - 1, round(float(box[0]) * nw / rw)))
    y0 = max(0, min(nh - 1, round(float(box[1]) * nh / rh)))
    x1 = max(1, min(nw, round(float(box[2]) * nw / rw)))
    y1 = max(1, min(nh, round(float(box[3]) * nh / rh)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate region {box!r} after scaling")
    return x0, y0, x1, y1


def _enhance(src: Image.Image, variant: str) -> Image.Image:
    import cv2
    rgb = np.asarray(src.convert("RGB"))
    if variant == "original":
        return src.convert("RGB")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if variant == "clahe":
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8)).apply(gray)
        return Image.fromarray(gray).convert("RGB")
    if variant == "black_ink":
        # Saturated silk creates false positives: this is a candidate only.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0]
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8)).apply(l)
        binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 14)
        return Image.fromarray(binary).convert("RGB")
    raise ValueError(f"Unknown variant {variant}")


def _rapid_init():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print("RapidOCR unavailable; testing Tesseract only.")
        return None
    try:
        return RapidOCR()
    except Exception as e:
        print(f"RapidOCR initialization failed; skipping: {e}")
        return None


def _rapid_on_single_column(engine: Any, region: Image.Image, variant: str) -> tuple[str, int]:
    # RapidOCR is a horizontal-line engine; rotate a vertical column clockwise.
    source = _enhance(region, variant)
    horizontal = source.transpose(Image.Transpose.ROTATE_270)
    results, _ = engine(np.asarray(horizontal))
    entries = []
    for r in results or []:
        if len(r) < 3 or not r[1]:
            continue
        try:
            x = min(float(p[0]) for p in r[0])
        except Exception:
            x = 0.0
        entries.append((x, str(r[1]).strip()))
    entries.sort(key=lambda e: e[0])
    return "".join(v for _, v in entries), len(entries)


def _tess_exe() -> str | None:
    for candidate in (
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def _tess_langs(exe: str | None) -> set[str]:
    if not exe:
        return set()
    try:
        proc = subprocess.run([exe, "--list-langs"], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=15)
        return set((proc.stdout + "\n" + proc.stderr).split())
    except Exception:
        return set()


def _tess_column(exe: str, region: Image.Image, variant: str,
                 lang: str, strategy: str, tmp_file: Path) -> str:
    prepared = _enhance(region, variant)
    if strategy == "rotate_then_horizontal":
        prepared = prepared.transpose(Image.Transpose.ROTATE_270)
        psm = "7"
    else:
        psm = "5"
    prepared.save(tmp_file)
    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")
    res = subprocess.run([exe, str(tmp_file), "stdout", "-l", lang,
                          "--psm", psm], text=True, capture_output=True,
                         encoding="utf-8", errors="replace", timeout=50, env=env)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[-250:])
    return re.sub(r"\s+", "", res.stdout)


def _n_han(text: str) -> int:
    return sum(1 for c in text if (0x3400 <= ord(c) <= 0x9FFF or
                                   0xF900 <= ord(c) <= 0xFAFF or
                                   0x20000 <= ord(c) <= 0x323AF))


def _viet_syllables(out: Path, doc_id: str) -> tuple[int | None, str]:
    """Very rough *count only*, NOT Hán-Việt pronunciation verification."""
    source = out / "02_ocr_tho" / "viet" / (doc_id + ".phien_am.txt")
    if not source.exists():
        return None, "No printed Phiên âm draft found"
    raw = source.read_text(encoding="utf-8")
    # Avoid zero if the segmentation file was empty. Count words, not alignment.
    words = re.findall(r"[^\W\d_]+", raw, flags=re.UNICODE)
    return len(words), "Syllable count from uncorrected Vietnamese OCR (NOT a reference label)"


def run(cfg_path: Path, regions_path: Path, docs: list[str], visual_only: bool) -> Path:
    _, out = _project_root(cfg_path)
    config = _read_settings(regions_path)["documents"]
    dst = out / "02_ocr_tho" / "han_region_probe"
    dst.mkdir(parents=True, exist_ok=True)
    rapid = None if visual_only else _rapid_init()
    tess = None if visual_only else _tess_exe()
    langs = _tess_langs(tess)
    vertical_lang = next((n for n in ("chi_tra_vert", "HanT_vert") if n in langs), None)
    horizontal_lang = next((n for n in ("chi_tra", "HanT") if n in langs), None)
    print(f"Step 2C column probe: RapidOCR={'yes' if rapid else 'no'}, "
          f"Tesseract={'yes' if tess else 'no'}, chi_tra={'yes' if horizontal_lang else 'no'}")

    all_records = []
    for doc in docs:
        if doc not in config:
            raise ValueError(f"No configured regions for {doc}")
        spec = config[doc]
        src = _find_image(out, doc)
        with Image.open(src) as im:
            original = ImageOps.exif_transpose(im).convert("RGB")
        oriented = _orient(original, spec["orientation"])
        doc_out = dst / doc
        image_dir = doc_out / "regions"
        candidate_dir = doc_out / "candidates"
        image_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        oriented.save(doc_out / "oriented_page.jpg", quality=94)
        overlay = oriented.copy()
        draw = ImageDraw.Draw(overlay)
        ref_size = (int(spec["reference_width"]), int(spec["reference_height"]))
        phonetics, note = _viet_syllables(out, doc)
        doc_results = []
        for i, r in enumerate(spec["regions"], start=1):
            name = r["name"]
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(f"Unsafe region name: {name!r}")
            box = _rect(r["xyxy"], ref_size, oriented.size)
            image = oriented.crop(box)
            region_file = image_dir / f"{i:02d}_{name}.png"
            image.save(region_file)
            # Paint preview only; never modify underlying source crop.
            draw.rectangle(box, outline=(30, 100, 245), width=3)
            label_y = max(0, box[1] - 17)
            draw.rectangle((box[0], label_y, box[0] + 19, label_y + 14), fill=(255, 255, 255))
            draw.text((box[0] + 2, label_y), f"{i:02d}", fill=(10, 25, 200))
            for variant in ("original", "clahe", "black_ink"):
                cand = _enhance(image, variant)
                cand.save(image_dir / f"{i:02d}_{name}_{variant}.png")
                methods = []
                if not visual_only and rapid is not None:
                    methods.append(("rapid_rotated", lambda: _rapid_on_single_column(rapid, image, variant)[0]))
                if not visual_only and tess and horizontal_lang:
                    methods.append(("tess_rotated", lambda: _tess_column(
                        tess, image, variant, horizontal_lang, "rotate_then_horizontal",
                        candidate_dir / "_tesseract_tmp.png")))
                if not visual_only and tess and vertical_lang:
                    methods.append(("tess_vertical", lambda: _tess_column(
                        tess, image, variant, vertical_lang, "vertical",
                        candidate_dir / "_tesseract_tmp.png")))
                for method, fn in methods:
                    error = ""
                    try:
                        text = fn()
                    except Exception as e:
                        text = ""
                        error = f"{type(e).__name__}: {str(e)[:180]}"
                    name_out = f"{i:02d}_{name}__{method}__{variant}.txt"
                    (candidate_dir / name_out).write_text(text, encoding="utf-8")
                    entry = {
                        "doc_id": doc, "region": f"{i:02d}_{name}",
                        "orientation": spec["orientation"], "method": method,
                        "variant": variant, "cjk_characters": _n_han(text),
                        "text": text, "error": error,
                        "verified": "NO", "important": "CANDIDATE ONLY — NOT GROUND TRUTH",
                    }
                    all_records.append(entry)
                    doc_results.append(entry)
        temp_tess = candidate_dir / "_tesseract_tmp.png"
        if temp_tess.exists():
            temp_tess.unlink()
        overlay.save(doc_out / "region_overlay.jpg", quality=95)
        report = {
            "doc_id": doc,
            "original_file": str(src),
            "orientation": spec["orientation"],
            "region_count": len(spec["regions"]),
            "region_definitions": spec["regions"],
            "rough_phien_am_word_count": phonetics,
            "word_count_note": note,
            "warning": (
                "Raw OCR character counts are coverage hints ONLY. "
                "Printed Phiên âm OCR may contain errors; counts do not establish "
                "character correctness or pronunciation agreement. Reading order and "
                "manual region boundaries need review."
            ),
            "best_candidate_per_region": "NONE_AUTOMATICALLY_SELECTED",
            "num_candidate_attempts": len(doc_results),
            "needs_manual_verification": True,
        }
        (doc_out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {doc}: {len(spec['regions'])} tentative regions, {len(doc_results)} OCR attempts")
        print(f"    Preview: {doc_out / 'region_overlay.jpg'}")
    csv_path = dst / "candidate_comparison.csv"
    cols = ["doc_id", "region", "orientation", "method", "variant",
            "cjk_characters", "text", "error", "verified", "important"]
    if not visual_only:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=cols)
            wr.writeheader()
            wr.writerows(all_records)
        print(f"Candidate table: {csv_path}")
    if visual_only:
        print("VISUAL-ONLY mode: no OCR executed. Review region_overlay.jpg before recognition.")
    else:
        print("All recognized characters are UNVERIFIED; DO NOT substitute them into Label.txt.")
    return dst


def main():
    parser = argparse.ArgumentParser(description="Sắc phong Step 2C Chinese column OCR pilot")
    parser.add_argument("--config", default="config.yaml", help="Existing project config")
    parser.add_argument("--regions", default="step2c_regions.json", help="Tentative pilot column ROIs")
    parser.add_argument("--doc", nargs="+", default=list(DEFAULT_DOCS),
                        help="Pilot document IDs; defaults to 006.01 and 008.02")
    parser.add_argument("--visual-only", action="store_true", help="Save overlays without running OCR")
    args = parser.parse_args()
    run(Path(args.config), Path(args.regions), args.doc, args.visual_only)


if __name__ == "__main__":
    main()
