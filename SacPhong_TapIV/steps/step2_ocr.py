"""Step 2 pilot: OCR Hán decree crops and printed Phiên âm/Dịch nghĩa crops.

Designed for Step 1's output/Sac_phong_Tap_IV/00_preview/crop_manifest.csv.

Usage from project root:
    python pipeline.py --step 2 --folder ch
    python pipeline.py --step 2 --folder vi
    python pipeline.py --step 2
    python pipeline.py --step 2 --folder vi --force

No translated or corrected characters are invented. OCR is unverified and must be
compared to the original page image before inclusion in a training corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def _root(cfg: dict) -> Path:
    """For the original pipeline runner, paths are relative to project root."""
    return Path(str(cfg.get("output_dir", "output"))) / str(cfg["book_name"])


def _read_manifest(root: Path) -> list[dict]:
    path = root / "00_preview" / "crop_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find {path}. Run 'python pipeline.py --step 1' first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty crop manifest: {path}")
    seen: set[str] = set()
    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        if not re.fullmatch(r"[\w-]+\.\d{3}\.\d{2}", doc_id):
            raise ValueError(f"Unexpected doc_id in crop manifest: {doc_id!r}")
        if doc_id in seen:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        seen.add(doc_id)
    return rows


def _ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError(
            "RapidOCR is missing. Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from e
    return RapidOCR()


def _rapid_results(engine: Any, img: np.ndarray) -> list[dict]:
    results, _ = engine(img)
    boxes = []
    for entry in results or []:
        if len(entry) < 3:
            continue
        pts, txt, conf = entry[0], entry[1], entry[2]
        if not txt or len(pts) != 4:
            continue
        coords = [[int(round(x)), int(round(y))] for x, y in pts]
        boxes.append({
            "transcription": str(txt).strip(),
            "points": coords,
            "confident": round(float(conf), 4),
        })
    return boxes


def _rotate_back(points: list[list[int]], original_width: int,
                 original_height: int, direction: str) -> list[list[int]]:
    """Map OCR boxes from 90-degree rotated crop into original image coords."""
    out: list[list[int]] = []
    for xr, yr in points:
        if direction == "clockwise":
            x, y = yr, original_height - 1 - xr
        elif direction == "counterclockwise":
            x, y = original_width - 1 - yr, xr
        else:
            x, y = xr, yr
        out.append([max(0, min(original_width - 1, x)),
                    max(0, min(original_height - 1, y))])
    return out


def _recognize_han_baseline(engine: Any, path: Path, cfg: dict, doc_id: str) -> tuple[list[dict], str]:
    """Rotate vertical Hán text to horizontal for RapidOCR.

    Convention: traditional columns are read right-to-left and within each
    column top-to-bottom. A 90-degree clockwise rotation makes characters
    horizontal, but places the rightmost column at the *bottom*. Consequently
    OCR rows must be assembled from bottom to top.
    """
    ocr_cfg = cfg.get("ocr", {})
    overrides = ocr_cfg.get("han_rotation_overrides", {}) or {}
    mode = overrides.get(doc_id, ocr_cfg.get("han_rotation", "clockwise"))
    if mode not in ("clockwise", "counterclockwise", "none"):
        raise ValueError("ocr.han_rotation must be clockwise/counterclockwise/none")
    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
        w, h = original.size
        if mode == "clockwise":
            processed = original.transpose(Image.Transpose.ROTATE_270)
        elif mode == "counterclockwise":
            processed = original.transpose(Image.Transpose.ROTATE_90)
        else:
            processed = original
        # Mild contrast normalization only. Preserve the archival photo itself.
        items = _rapid_results(engine, np.asarray(processed))
        def mid(box):
            xs = [p[0] for p in box["points"]]
            ys = [p[1] for p in box["points"]]
            return sum(xs) / 4.0, sum(ys) / 4.0
        if mode == "clockwise":
            # bottom-to-top rows = original right-to-left columns
            items.sort(key=lambda b: (-round(mid(b)[1] / 25), mid(b)[0]))
        elif mode == "counterclockwise":
            items.sort(key=lambda b: (round(mid(b)[1] / 25), -mid(b)[0]))
        else:
            items.sort(key=lambda b: (-round(mid(b)[0] / 25), mid(b)[1]))
        text = "\n".join(item["transcription"] for item in items)
        for item in items:
            item["points"] = _rotate_back(item["points"], w, h, mode)
        return items, text


def _cjk_count(text: str) -> int:
    """Count CJK ideographs without mistaking symbols for readable Chinese."""
    return sum(1 for c in text if (
        0x3400 <= ord(c) <= 0x9FFF
        or 0xF900 <= ord(c) <= 0xFAFF
        or 0x20000 <= ord(c) <= 0x323AF
    ))


def _han_preprocess(img: Image.Image, variant: str) -> tuple[np.ndarray, float]:
    """Make an OCR input, never modifying the original archival crop.

    The scale factor maps OCR box positions back to the source crop.
    These are *candidate* images; visual enhancement does not prove accuracy.
    """
    import cv2

    rgb = np.array(img.convert("RGB"))
    if variant == "original":
        return rgb, 1.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if variant == "clahe":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        adjusted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return adjusted, 1.0
    if variant == "gray_upscaled":
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(cv2.resize(enhanced, None, fx=1.6, fy=1.6,
                           interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB), 1.6
    if variant == "ink_threshold":
        # Local-contrast binarization is particularly relevant to yellow silk.
        # It can also magnify background decoration: review output carefully.
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 35, 12)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), 1.0
    raise ValueError(f"Unsupported Chinese OCR variant: {variant!r}")


def _han_sort(boxes: list[dict], rotation: str) -> list[dict]:
    def center(b):
        pts = b["points"]
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    if rotation == "clockwise":
        # After clockwise rotation: bottommost row is rightmost source column.
        return sorted(boxes, key=lambda b: (-round(center(b)[1] / 25), center(b)[0]))
    if rotation == "counterclockwise":
        return sorted(boxes, key=lambda b: (round(center(b)[1] / 25), -center(b)[0]))
    # Original horizontal writing is top-down and left-right; original vertical
    # writing is NOT reliably ordered by this branch. Use vertical rotation for it.
    return sorted(boxes, key=lambda b: (round(center(b)[1] / 20), center(b)[0]))


def _recognize_han_variant(engine: Any, path: Path, cfg: dict,
                           doc_id: str, variant: str) -> tuple[list[dict], str]:
    """OCR a Chinese candidate and restore all box points to source pixels."""
    ocr_cfg = cfg.get("ocr", {})
    overrides = ocr_cfg.get("han_rotation_overrides", {}) or {}
    mode = overrides.get(doc_id, ocr_cfg.get("han_rotation", "clockwise"))
    if mode not in ("clockwise", "counterclockwise", "none"):
        raise ValueError(f"Unsupported rotation: {mode!r}")
    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
        w, h = original.size
        processed, scale = _han_preprocess(original, variant)

    # cv2 ndarray rotations are exactly equivalent to PIL ROTATE_270/90.
    if mode == "clockwise":
        processed = np.rot90(processed, k=3).copy()
    elif mode == "counterclockwise":
        processed = np.rot90(processed, k=1).copy()
    items = _han_sort(_rapid_results(engine, processed), mode)
    # Scale back before reverse rotation. Clamp coordinates to actual image.
    for item in items:
        pts = [[int(round(x / scale)), int(round(y / scale))]
               for x, y in item["points"]]
        item["points"] = _rotate_back(pts, w, h, mode)
    return items, "\n".join(it["transcription"] for it in items)


def _chinese_tesseract_languages(exe: str) -> tuple[str | None, str | None]:
    """Find installed Traditional Chinese horizontal/vertical tessdata."""
    run = subprocess.run([exe, "--list-langs"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=20)
    if run.returncode != 0:
        return None, None
    names = set((run.stdout + "\n" + run.stderr).split())
    horizontal = next((x for x in ("chi_tra", "HanT") if x in names), None)
    vertical = next((x for x in ("chi_tra_vert", "HanT_vert") if x in names), None)
    return horizontal, vertical


def _tesseract_chinese_candidate(exe: str, path: Path, lang: str,
                                 psm: int) -> tuple[list[dict], str]:
    """Independent *unverified* OCR candidate for inspection, not a gold label."""
    import os
    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")
    cmd = [exe, str(path), "stdout", "-l", lang, "--psm", str(psm), "tsv"]
    run = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env, timeout=90)
    if run.returncode != 0:
        raise RuntimeError(f"Tesseract Chinese comparison failed: {run.stderr[:300]}")
    boxes, groups = [], {}
    for word in csv.DictReader(run.stdout.splitlines(), delimiter="\t"):
        value = (word.get("text") or "").strip()
        if not value:
            continue
        try:
            conf = float(word.get("conf") or -1)
            x, y, wd, ht = (int(word[k]) for k in ("left", "top", "width", "height"))
            key = tuple(int(word[k]) for k in ("block_num", "par_num", "line_num"))
        except (ValueError, KeyError):
            continue
        if conf < 0 or min(wd, ht) < 1:
            continue
        groups.setdefault(key, []).append((value, conf, x, y, x + wd, y + ht))
    for pieces in groups.values():
        x0 = min(w[2] for w in pieces)
        y0 = min(w[3] for w in pieces)
        x1 = max(w[4] for w in pieces)
        y1 = max(w[5] for w in pieces)
        boxes.append({
            "transcription": "".join(w[0] for w in pieces),
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "confident": round(sum(w[1] for w in pieces) / len(pieces) / 100, 4),
        })
    if lang.endswith("_vert"):
        boxes.sort(key=lambda b: (-b["points"][0][0], b["points"][0][1]))
    else:
        boxes.sort(key=lambda b: (b["points"][0][1], b["points"][0][0]))
    return boxes, "\n".join(b["transcription"] for b in boxes)


def _save_han_candidate(folder: Path, name: str, items: list[dict], text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.txt").write_text(text, encoding="utf-8")
    (folder / f"{name}.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_metric(items: list[dict], text: str, engine_name: str, variant: str) -> dict:
    confidence = [float(b.get("confident", 0)) for b in items]
    return {
        "engine": engine_name,
        "variant": variant,
        "text_boxes": len(items),
        "cjk_characters": _cjk_count(text),
        "mean_engine_confidence": round(sum(confidence) / len(confidence), 4)
                                   if confidence else 0.0,
        "review_status": "UNVERIFIED; DO NOT USE AS GOLD LABEL",
    }


def _recognize_han_candidates(engine: Any, path: Path, cfg: dict, doc_id: str,
                               candidate_dir: Path) -> tuple[list[dict], str, list[dict]]:
    """Collect independent OCR drafts without silently selecting hallucinated text.

    The primary output is original RapidOCR so previous runs remain comparable.
    Additional RapidOCR preprocessing and optional Tesseract candidates are
    reviewable, not automatically merged into the source transcription.
    """
    ocr_cfg = cfg.get("ocr", {})
    requested = ocr_cfg.get("han_variants", ["original", "clahe", "gray_upscaled", "ink_threshold"])
    variants = list(dict.fromkeys(str(v) for v in requested))
    if "original" not in variants:
        variants.insert(0, "original")
    if any(v not in ("original", "clahe", "gray_upscaled", "ink_threshold") for v in variants):
        raise ValueError("ocr.han_variants contains an unknown variant")

    metrics = []
    primary_items, primary_text = [], ""
    for variant in variants:
        try:
            items, recognized = _recognize_han_variant(engine, path, cfg, doc_id, variant)
        except Exception as exc:
            metrics.append({"engine": "RapidOCR", "variant": variant,
                            "error": str(exc), "review_status": "FAILED"})
            continue
        _save_han_candidate(candidate_dir, f"rapid_{variant}", items, recognized)
        m = _candidate_metric(items, recognized, "RapidOCR", variant)
        m["is_primary"] = (variant == "original")
        metrics.append(m)
        if variant == "original":
            primary_items, primary_text = items, recognized

    if ocr_cfg.get("han_compare_tesseract", True):
        try:
            exe = _find_tesseract(cfg)
            horizontal, vertical = _chinese_tesseract_languages(exe)
            rotation = (ocr_cfg.get("han_rotation_overrides", {}) or {}).get(
                doc_id, ocr_cfg.get("han_rotation", "clockwise"))
            lang = horizontal if rotation == "none" else vertical
            psm = 6 if rotation == "none" else 5
            if lang:
                for variant in ("original", "clahe"):
                    with Image.open(path) as source:
                        image, scale = _han_preprocess(ImageOps.exif_transpose(source).convert("RGB"), variant)
                    # Tesseract uses the original upright writing direction.
                    # Both chosen comparison variants preserve size 1:1.
                    if scale != 1.0:
                        continue
                    import tempfile
                    with tempfile.TemporaryDirectory() as td:
                        tmp = Path(td) / "candidate.png"
                        Image.fromarray(image).save(tmp)
                        try:
                            items, recognized = _tesseract_chinese_candidate(exe, tmp, lang, psm)
                        except Exception as exc:
                            metrics.append({"engine": "Tesseract", "variant": variant,
                                            "error": str(exc), "review_status": "FAILED"})
                            continue
                    _save_han_candidate(candidate_dir, f"tesseract_{variant}", items, recognized)
                    m = _candidate_metric(items, recognized, "Tesseract", variant)
                    m["is_primary"] = False
                    metrics.append(m)
            else:
                metrics.append({"engine": "Tesseract", "variant": "not_installed",
                                "error": "Traditional Chinese language data missing (chi_tra/chi_tra_vert)",
                                "review_status": "SKIPPED"})
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            metrics.append({"engine": "Tesseract", "variant": "unavailable",
                            "error": str(exc), "review_status": "SKIPPED"})

    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "comparison.json").write_text(json.dumps({
        "doc_id": doc_id,
        "primary": "rapid_original",
        "note": "Character counts and confidence are NOT accuracy measurements.",
        "candidates": metrics,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return primary_items, primary_text, metrics


def _recognize_viet_rapid(engine: Any, path: Path) -> tuple[list[dict], str]:
    with Image.open(path) as im:
        items = _rapid_results(engine, np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
    items.sort(key=lambda b: (round(sum(pt[1] for pt in b["points"]) / 4 / 15),
                              sum(pt[0] for pt in b["points"]) / 4))
    text = "\n".join(item["transcription"] for item in items)
    return items, text


def _find_tesseract(cfg: dict) -> str:
    """Use PATH, a configured executable, or the default Windows location."""
    ocr_cfg = cfg.get("ocr", {})
    configured = str(ocr_cfg.get("tesseract_path") or "").strip()
    choices = [configured, shutil.which("tesseract") or "",
               r"C:\Program Files\Tesseract-OCR\tesseract.exe",
               r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for candidate in choices:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "Tesseract executable not found. Install the Windows Tesseract OCR app "
        "with Vietnamese (vie) language data, then set ocr.tesseract_path "
        "in config.yaml if it is not on PATH."
    )


def _check_tesseract_languages(exe: str) -> None:
    probe = subprocess.run([exe, "--list-langs"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=20)
    langs = set((probe.stdout + "\n" + probe.stderr).split())
    if "vie" not in langs:
        raise RuntimeError("Missing 'vie' traineddata. Install Vietnamese OCR language in Tesseract.")
    if "eng" not in langs:
        raise RuntimeError("Missing 'eng' traineddata. Install English OCR language in Tesseract.")


def _tesseract_tsv(exe: str, path: Path, psm: int) -> tuple[list[dict], str]:
    """OCR using Vietnamese traineddata; one PPOCRLabel box per printed line."""
    import os
    env = os.environ.copy()
    # Prevent Tesseract from starting excessive OpenMP threads for tiny crops.
    env.setdefault("OMP_THREAD_LIMIT", "1")
    cmd = [exe, str(path), "stdout", "-l", "vie+eng", "--psm", str(psm), "tsv"]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", env=env, timeout=90)
    if run.returncode != 0:
        raise RuntimeError(f"Tesseract failed on {path.name}: {run.stderr[:500]}")
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for word in csv.DictReader(run.stdout.splitlines(), delimiter="\t"):
        value = (word.get("text") or "").strip()
        if not value:
            continue
        try:
            confidence = float(word.get("conf") or -1)
            x, y, width, height = (int(word[k]) for k in ("left", "top", "width", "height"))
            key = tuple(int(word[k]) for k in ("block_num", "par_num", "line_num"))
        except (KeyError, ValueError):
            continue
        if confidence < 0 or width < 1 or height < 1:
            continue
        groups.setdefault(key, []).append({
            "text": value, "conf": confidence / 100,
            "bbox": (x, y, x + width, y + height),
        })
    boxes = []
    for words in groups.values():
        x0 = min(w["bbox"][0] for w in words)
        y0 = min(w["bbox"][1] for w in words)
        x1 = max(w["bbox"][2] for w in words)
        y1 = max(w["bbox"][3] for w in words)
        boxes.append({
            "transcription": " ".join(w["text"] for w in words),
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "confident": round(sum(w["conf"] for w in words) / len(words), 4),
        })
    boxes.sort(key=lambda b: (b["points"][0][1], b["points"][0][0]))
    return boxes, "\n".join(b["transcription"] for b in boxes)


def _recognize_viet_tesseract(path: Path, cfg: dict) -> tuple[list[dict], str, int]:
    """Use PSM 6 for ordinary printed blocks; PSM 11 for wide/sparse pages.

    If a mode fails to find either heading, try the other mode and prefer the
    candidate with both printed headings. NO model-generated translations.
    """
    exe = _find_tesseract(cfg)
    _check_tesseract_languages(exe)
    with Image.open(path) as im:
        ratio = im.width / max(im.height, 1)
    modes = [11, 6] if ratio > 4.0 else [6, 11]
    candidates = []
    for mode in modes:
        boxes, text = _tesseract_tsv(exe, path, mode)
        reading, meaning, _ = _split_vn_blocks(text)
        headings_found = int(bool(reading)) + int(bool(meaning))
        # Prefer two recognized sections; then prefer more meaningful text.
        score = (headings_found, min(len(reading) + len(meaning), 10000))
        candidates.append((score, boxes, text, mode))
        if headings_found == 2 and len(reading) > 40 and len(meaning) > 40:
            break
    _, boxes, text, chosen_psm = max(candidates, key=lambda v: v[0])
    return boxes, text, chosen_psm


def _split_vn_blocks(text: str) -> tuple[str, str, str]:
    """Split only on recognized *printed* section headings; no translation added."""
    lines = [s.strip() for s in text.replace("\r\n", "\n").split("\n") if s.strip()]
    reading, meaning, other = [], [], []
    active = "other"
    phien = re.compile(r"^\s*(?:phi[eê]n\s*[aâ]m|phien\s*am)\s*[:：.]?\s*", re.I)
    dich = re.compile(r"^\s*(?:d[iị]ch\s*ngh[iĩ]a|dich\s*nghia)\s*[:：.]?\s*", re.I)
    for line in lines:
        m1 = phien.match(line)
        m2 = dich.match(line)
        if m1:
            active = "reading"
            remainder = line[m1.end():].strip()
            if remainder:
                reading.append(remainder)
            continue
        if m2:
            active = "meaning"
            remainder = line[m2.end():].strip()
            if remainder:
                meaning.append(remainder)
            continue
        if active == "reading":
            reading.append(line)
        elif active == "meaning":
            meaning.append(line)
        else:
            other.append(line)
    return "\n".join(reading), "\n".join(meaning), "\n".join(other)


def _rewrite_label_file(folder: Path, image_names: list[str]) -> None:
    """Rebuild idempotently from one JSON sidecar per OCR image."""
    with (folder / "Label.txt").open("w", encoding="utf-8") as label, \
         (folder / "fileState.txt").open("w", encoding="utf-8") as state:
        for name in image_names:
            json_path = folder / f"{Path(name).stem}.ocr.json"
            if not json_path.exists():
                continue
            boxes = json.loads(json_path.read_text(encoding="utf-8"))
            label.write(f"{folder.name}/{name}\t{json.dumps(boxes, ensure_ascii=False)}\n")
            # 0 = labeling NOT complete; raw OCR needs human verification.
            state.write(f"{folder.name}/{name}\t0\n")


def _aggregate_text(root: Path, rows: list[dict]) -> None:
    # Stage 3 is reserved for CLEAN, checked text. Never put raw OCR there.
    target = root / "02_ocr_tho" / "aggregate_draft"
    target.mkdir(parents=True, exist_ok=True)
    inputs = {
        "han.txt": ("han", "txt"),
        "phien_am.txt": ("viet", "phien_am"),
        "dich_nghia.txt": ("viet", "dich_nghia"),
    }
    for filename, (folder_name, suffix) in inputs.items():
        pieces = []
        for row in rows:
            doc_id = row["doc_id"]
            f = root / "02_ocr_tho" / folder_name / (
                f"{doc_id}.txt" if suffix == "txt" else f"{doc_id}.{suffix}.txt"
            )
            if f.is_file():
                pieces.append(f"### {doc_id}\n{f.read_text(encoding='utf-8').strip()}\n")
        (target / filename).write_text("\n".join(pieces), encoding="utf-8")


def process_folder(folder_name: str, cfg: dict, force: bool = False) -> dict:
    """Compatible with original pipeline.py's Step 2 function call."""
    if folder_name not in ("ch", "vi"):
        print(f"Skipping '{folder_name}': mixed-layout pilot has only ch/vi cropped regions.")
        return {"processed": 0, "skipped": True}
    root = _root(cfg)
    rows = _read_manifest(root)
    data_folder = "han" if folder_name == "ch" else "viet"
    image_key = "han_image" if folder_name == "ch" else "viet_image"
    folder = root / "02_ocr_tho" / data_folder
    folder.mkdir(parents=True, exist_ok=True)
    engine_type = cfg.get("ocr", {}).get("vi_engine", "rapidocr").lower()
    if folder_name == "vi" and engine_type not in ("rapidocr", "tesseract"):
        raise ValueError("For this pilot, ocr.vi_engine must be rapidocr or tesseract")
    if folder_name == "vi" and engine_type == "tesseract":
        exe = _find_tesseract(cfg)
        _check_tesseract_languages(exe)
        engine = None
    else:
        engine = _ocr_engine()
    print(f"\nStep 2 OCR — {data_folder}: {len(rows)} pilot images; "
          f"engine={'RapidOCR' if engine is not None else 'Tesseract'}")
    results = []
    image_names = []
    processed = 0
    for row in rows:
        doc_id = row["doc_id"]
        image_rel = Path(row[image_key])
        img_path = root / image_rel
        if not img_path.exists():
            raise FileNotFoundError(f"Step 1 crop missing: {img_path}")
        image_names.append(img_path.name)
        text_path = folder / f"{doc_id}.txt"
        json_path = folder / f"{img_path.stem}.ocr.json"
        status = "OCR_DRAFT_UNVERIFIED"
        used_psm = None
        candidate_metrics = []
        candidate_dir = root / "02_ocr_tho" / "han_candidates" / doc_id
        if not force and text_path.exists() and json_path.exists():
            text = text_path.read_text(encoding="utf-8")
            items = json.loads(json_path.read_text(encoding="utf-8"))
            comparison_path = candidate_dir / "comparison.json"
            if folder_name == "ch" and comparison_path.is_file():
                candidate_metrics = json.loads(comparison_path.read_text(encoding="utf-8")).get("candidates", [])
        else:
            if folder_name == "ch":
                items, text, candidate_metrics = _recognize_han_candidates(
                    engine, img_path, cfg, doc_id, candidate_dir)
            elif engine_type == "tesseract":
                items, text, used_psm = _recognize_viet_tesseract(img_path, cfg)
            else:
                items, text = _recognize_viet_rapid(engine, img_path)
            text_path.write_text(text, encoding="utf-8")
            json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            processed += 1
        if folder_name == "vi":
            reading, meaning, other = _split_vn_blocks(text)
            (folder / f"{doc_id}.phien_am.txt").write_text(reading, encoding="utf-8")
            (folder / f"{doc_id}.dich_nghia.txt").write_text(meaning, encoding="utf-8")
            if not reading or not meaning:
                status = "MISSING_SECTION_HEADING_REVIEW_NEEDED"
            results.append({"doc_id": doc_id, "region_count": len(items),
                            "engine": engine_type, "selected_psm": used_psm,
                            "phien_am_detected": bool(reading),
                            "dich_nghia_detected": bool(meaning),
                            "extra_before_headings": len(other), "status": status})
        else:
            status = status if text.strip() else "EMPTY_OCR_REVIEW_NEEDED"
            results.append({"doc_id": doc_id, "region_count": len(items),
                            "han_char_count": _cjk_count(text),
                            "candidate_count": len(candidate_metrics),
                            "candidate_directory": str(candidate_dir),
                            "status": status})
        if folder_name == "ch" and not text.strip():
            status = "EMPTY_OCR_REVIEW_NEEDED"
        extra = f", Tesseract PSM {used_psm}" if used_psm is not None else ""
        print(f"  {doc_id}: {len(items)} text boxes ({status}{extra})")
    _rewrite_label_file(folder, image_names)
    _aggregate_text(root, rows)
    summary_path = root / "02_ocr_tho" / f"pilot_{data_folder}_summary.json"
    summary_path.write_text(json.dumps({"scope": "pilot_only", "source": "ocr_not_verified",
                                         "newly_processed": processed, "documents": results},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    if folder_name == "ch":
        print(f"  Chinese comparison candidates: {root / '02_ocr_tho' / 'han_candidates'}")
        print("  Primary OCR was NOT replaced by unverified alternative text.")
    print(f"  Saved OCR results in: {folder}")
    print(f"  Raw text preview:     {root / '02_ocr_tho' / 'aggregate_draft'}")
    print("  WARNING: OCR text is unverified. Do NOT use it as gold training data.")
    return {"processed": processed, "documents": len(results)}


def main():
    import yaml
    parser = argparse.ArgumentParser(description="Sắc phong Step 2 OCR pilot")
    parser.add_argument("--folder", default="all", choices=["ch", "vi", "all"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chosen = ["ch", "vi"] if args.folder == "all" else [args.folder]
    for folder in chosen:
        process_folder(folder, cfg, force=args.force)


if __name__ == "__main__":
    main()
