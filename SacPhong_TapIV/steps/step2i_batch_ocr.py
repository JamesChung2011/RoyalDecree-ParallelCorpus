r"""Step 2I: resumable OCR over Step 1C provisional paired crops.

Use from the HanNom_Project_SacPhong project root:

    .\.venv\Scripts\python.exe -m steps.step2i_batch_ocr --dry-run
    .\.venv\Scripts\python.exe -m steps.step2i_batch_ocr --engine vi
    .\.venv\Scripts\python.exe -m steps.step2i_batch_ocr --engine han
    .\.venv\Scripts\python.exe -m steps.step2i_batch_ocr --engine vi --all
    .\.venv\Scripts\python.exe -m steps.step2i_batch_ocr --engine han --all

All text is unverified. No existing OCR labels or earlier experiments are changed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
OLLAMA_API = "http://127.0.0.1:11434"
VI = "vi"
HAN = "han"
CSV_FIELDS = [
    "doc_id", "pdf_page", "engine", "status", "characters", "phien_am_syllables",
    "dich_nghia_words", "flags", "source_image", "output_json"
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_book_root() -> Path:
    cfg = ROOT / "config.yaml"
    if not cfg.is_file():
        return DEFAULT_BOOK
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        return ROOT / str(data.get("output_dir", "output")) / str(data.get("book_name", "Sac_phong_Tap_IV"))
    except (ImportError, OSError, ValueError, TypeError):
        return DEFAULT_BOOK


def valid_crop_path(root: Path, name: str) -> Path:
    # Manifest is local input, but do not allow it to read arbitrary project files.
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe crop path in manifest: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Crop leaves source directory: {name}")
    return path


def find_export(book_root: Path, source_run: str) -> Path:
    if source_run:
        candidate = Path(source_run).expanduser().resolve()
        sources = [candidate]
    else:
        parent = book_root / "01_layout_resolved_draft"
        sources = sorted(parent.glob("run_*"), reverse=True) if parent.exists() else []
    for root in sources:
        manifest = root / "document_manifest.csv"
        if not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row and row.get("han_image") and row.get("viet_image"):
            if valid_crop_path(root, row["han_image"]).is_file() and valid_crop_path(root, row["viet_image"]).is_file():
                return root
    raise FileNotFoundError(
        "No exported Step 1C crops found. First run:\n"
        "  .\\.venv\\Scripts\\python.exe -m steps.step1c_resolve_layout --export-crops"
    )


def load_manifest(source: Path) -> list[dict]:
    with (source / "document_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Step 1C document manifest is empty")
    ids = [row["auto_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate AUTO IDs in manifest")
    for row in rows:
        if not re.fullmatch(r"AUTO\.\d{3}\.\d{2}", row["auto_id"]):
            raise ValueError("Unexpected document ID: " + row["auto_id"])
        for key in ("han_image", "viet_image"):
            if not row.get(key) or not valid_crop_path(source, row[key]).is_file():
                raise FileNotFoundError(f"Missing {key} for {row['auto_id']}")
    return sorted(rows, key=lambda r: (int(r["pdf_page"]), int(r["document_number"])))


def heading_key(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d")
    return re.sub("[^a-z]", "", text)


def split_printed_text(raw: str) -> tuple[str, str, list[str]]:
    """Split printed sections; missing headings are flagged, never fabricated."""
    part = None
    phien, dich, flags = [], [], []
    saw_phien, saw_dich = False, False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = heading_key(stripped[:45])
        if normalized.startswith("phienam"):
            part = "phien"
            saw_phien = True
            # Handle a possible reading on the same heading line.
            match = re.search(r"[:：]", stripped)
            if match and stripped[match.end():].strip():
                phien.append(stripped[match.end():].strip())
            continue
        if normalized.startswith("dichnghia"):
            part = "dich"
            saw_dich = True
            match = re.search(r"[:：]", stripped)
            if match and stripped[match.end():].strip():
                dich.append(stripped[match.end():].strip())
            continue
        if part == "phien":
            phien.append(stripped)
        elif part == "dich":
            dich.append(stripped)
    if not saw_phien:
        flags.append("MISSING_PHIEN_AM_HEADING")
    if not saw_dich:
        flags.append("MISSING_DICH_NGHIA_HEADING")
    if len(" ".join(phien)) < 30:
        flags.append("SHORT_PHIEN_AM")
    if len(" ".join(dich)) < 30:
        flags.append("SHORT_DICH_NGHIA")
    return "\n".join(phien), "\n".join(dich), flags


def word_count(value: str) -> int:
    return len(re.findall(r"[^\W\d_]+", value, flags=re.UNICODE))


def tesseract_exe(override: str) -> str:
    candidates = [override] if override else []
    candidates += [shutil.which("tesseract") or ""]
    if sys.platform == "win32":
        candidates.append(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("Tesseract not found. Use --tesseract PATH, or put it on PATH.")


def tesseract_text(exe: str, image: Path, psm: int) -> str:
    cmd = [exe, str(image), "stdout", "-l", "vie+eng", "--psm", str(psm)]
    proc = subprocess.run(cmd, capture_output=True, check=False, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-1200:])
    return proc.stdout.decode("utf-8-sig", "replace").strip()


def recognize_vietnamese(image: Path, tesseract: str) -> dict:
    alternatives = []
    # Use 6 for normal text; use 11 when sections or text are missing.
    for psm in (6, 11):
        raw = tesseract_text(tesseract, image, psm)
        phien, dich, flags = split_printed_text(raw)
        score = (4 - len(flags)) * 1000 + min(word_count(phien), 220) + min(word_count(dich), 450)
        alternatives.append({
            "psm": psm, "raw_text": raw, "phien_am": phien,
            "dich_nghia": dich, "flags": flags, "score_diagnostic": score
        })
        if psm == 6 and not flags:
            break
    selected = max(alternatives, key=lambda x: x["score_diagnostic"])
    return {
        "status": "OCR_DRAFT_UNVERIFIED" if not selected["flags"] else "OCR_NEEDS_REVIEW",
        "selected_psm": selected["psm"], "raw_text": selected["raw_text"],
        "phien_am": selected["phien_am"], "dich_nghia": selected["dich_nghia"],
        "phien_am_syllables": word_count(selected["phien_am"]),
        "dich_nghia_words": word_count(selected["dich_nghia"]),
        "flags": selected["flags"], "alternatives": alternatives
    }


def han_chars(text: str) -> int:
    return sum(
        0x3400 <= ord(c) <= 0x9FFF
        or 0xF900 <= ord(c) <= 0xFAFF
        or 0x20000 <= ord(c) <= 0x323AF
        for c in text
    )


def han_candidate(text: str) -> str:
    return "".join(
        c for c in text if han_chars(c) or c in "□\n，。、；：！？"
    ).strip()


def repeated(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return "□" * 9 in compact or bool(re.search(r"(.{1,5})\1{7,}", compact))


def encoded_image(image: Image.Image, max_side: int) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=93)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def verify_ollama_model(model: str, timeout: int) -> None:
    with urllib.request.urlopen(OLLAMA_API + "/api/tags", timeout=timeout) as response:
        obj = json.load(response)
    names = [m.get("name", "") for m in obj.get("models", [])]
    if model not in names:
        raise RuntimeError(f"Model {model} not installed. Found: {names}")


def chat_ocr(image: Image.Image, model: str, max_side: int, timeout: int) -> tuple[str, dict]:
    payload = {
        "model": model, "stream": False, "keep_alive": "2m",
        "messages": [{
            "role": "user", "content": "Text Recognition:",
            "images": [encoded_image(image, max_side)]
        }],
        "options": {"temperature": 0, "num_predict": 320, "num_ctx": 4096}
    }
    request = urllib.request.Request(
        OLLAMA_API + "/api/chat", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            obj = json.load(response)
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", "replace")[-1000:]
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
    return str(obj.get("message", {}).get("content", "")), {
        "done_reason": obj.get("done_reason"), "eval_count": obj.get("eval_count")
    }


def recognize_chinese(image_path: Path, model: str, max_side: int, timeout: int,
                       expected_syllables: int) -> dict:
    variants = []
    with Image.open(image_path) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
        options = [("original", img)]
        # The sideways calligraphy exemplified by page 8 often needs rotation.
        # Retry the rotated view only if the original returns empty or wildly wrong length.
        for variant, picture in options:
            start = time.perf_counter()
            try:
                raw, meta = chat_ocr(picture, model, max_side, timeout)
                extracted = han_candidate(raw)
                count = han_chars(extracted)
                flags = []
                if repeated(raw):
                    flags.append("REPETITION_DETECTED")
                if count == 0:
                    flags.append("NO_CHINESE_TEXT")
                if expected_syllables > 10 and (count < .60 * expected_syllables or count > 1.40 * expected_syllables):
                    flags.append("POSSIBLE_COUNT_MISMATCH")
                if "□" in extracted:
                    flags.append("UNREADABLE_MARKERS")
                status = "REPETITION_REJECTED" if "REPETITION_DETECTED" in flags else (
                    "OCR_NEEDS_REVIEW" if flags else "OCR_DRAFT_UNVERIFIED")
                attempt = {
                    "variant": variant, "status": status, "raw_response": raw,
                    "chinese_candidate": extracted, "chinese_characters": count,
                    "flags": flags, "generation_metadata": meta,
                    "elapsed_seconds": round(time.perf_counter() - start, 2)
                }
            except Exception as exc:
                attempt = {
                    "variant": variant, "status": "OCR_ERROR", "raw_response": "",
                    "chinese_candidate": "", "chinese_characters": 0,
                    "flags": ["MODEL_ERROR"], "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - start, 2)
                }
            variants.append(attempt)
            count = attempt["chinese_characters"]
            needs_rotate = (
                count == 0 or attempt["status"] in ("REPETITION_REJECTED", "OCR_ERROR")
                or expected_syllables > 10 and (count < expected_syllables * .6 or count > expected_syllables * 1.4)
            )
            if variant == "original" and needs_rotate:
                options.append(("rotated_clockwise", img.transpose(Image.Transpose.ROTATE_270)))
    # NEVER call this 'correct' or 'verified'. Prefer non-repetitive output near expected length for review only.
    def rank(a: dict) -> tuple[int, float]:
        good = a["status"] not in ("REPETITION_REJECTED", "OCR_ERROR") and a["chinese_characters"] > 0
        distance = abs(a["chinese_characters"] - expected_syllables) if expected_syllables else 0
        return int(good), -distance
    best = max(variants, key=rank)
    chosen = best["status"] not in ("REPETITION_REJECTED", "OCR_ERROR") and bool(best["chinese_characters"])
    return {
        "status": ("OCR_NEEDS_REVIEW" if not chosen or best["flags"] else "OCR_DRAFT_UNVERIFIED"),
        "model": model, "chinese_candidate": best["chinese_candidate"] if chosen else "",
        "chinese_characters": best["chinese_characters"] if chosen else 0,
        "selected_variant": best["variant"] if chosen else None,
        "flags": sorted(set(sum((x["flags"] for x in variants), []))),
        "expected_phien_am_syllables": expected_syllables,
        "attempts": variants
    }


def report_for(row: dict, result: dict, engine: str, path: Path) -> dict:
    return {
        "doc_id": row["auto_id"], "pdf_page": row["pdf_page"], "engine": engine,
        "status": result.get("status", "OCR_ERROR"),
        "characters": result.get("chinese_characters", 0) if engine == HAN else len(result.get("raw_text", "")),
        "phien_am_syllables": result.get("phien_am_syllables", result.get("expected_phien_am_syllables", 0)),
        "dich_nghia_words": result.get("dich_nghia_words", 0),
        "flags": "; ".join(result.get("flags", [])),
        "source_image": row["han_image"] if engine == HAN else row["viet_image"],
        "output_json": str(path)
    }


def update_index(out: Path, rows: list[dict], engine: str) -> None:
    indexed = []
    for row in rows:
        dest = out / engine / f"{row['auto_id']}.json"
        if dest.is_file():
            try:
                result = read_json(dest).get("result", {})
                indexed.append(report_for(row, result, engine, dest.relative_to(out)))
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                indexed.append({
                    "doc_id": row["auto_id"], "pdf_page": row["pdf_page"], "engine": engine,
                    "status": "CORRUPTED_RECORD", "flags": str(exc), "output_json": str(dest.relative_to(out))
                })
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"progress_{engine}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(indexed)
    statuses = Counter(row["status"] for row in indexed)
    atomic_json(out / f"summary_{engine}.json", {
        "stage": "Step 2I", "engine": engine, "generated_at": now_iso(),
        "documents_in_manifest": len(rows), "processed_or_recorded": len(indexed),
        "status_counts": dict(statuses), "verified_transcriptions": 0,
        "warning": "All OCR text is unverified; do not use as verified corpus/training data."
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Step2I resumable batch OCR (unverified drafts)")
    parser.add_argument("--source-run", default="", help="Step1C exported run folder; auto-detected if omitted")
    parser.add_argument("--engine", choices=[VI, HAN, "both"], default=VI)
    parser.add_argument("--limit", type=int, default=5, help="Pilot count (default 5)")
    parser.add_argument("--all", action="store_true", help="Process all 291 provisional documents")
    parser.add_argument("--ids", nargs="+", default=[], help="Specific AUTO.xxx.xx IDs instead")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun existing Step2I checkpoints only")
    parser.add_argument("--tesseract", default="", help="Path to Tesseract executable")
    parser.add_argument("--model", default="glm-ocr:q8_0")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.limit < 1 or not 512 <= args.max_side <= 4096:
        parser.error("--limit must be positive; --max-side must be 512..4096")
    book = load_book_root()
    source = find_export(book, args.source_run)
    rows = load_manifest(source)
    selected = [r for r in rows if r["auto_id"] in set(args.ids)] if args.ids else (rows if args.all else rows[:args.limit])
    if args.ids and len(selected) != len(set(args.ids)):
        parser.error("Some requested IDs are missing from the Step1C manifest")
    if not selected:
        parser.error("No documents selected")
    engines = [VI, HAN] if args.engine == "both" else [args.engine]
    out = book / "02_ocr_tho" / "step2i_batch_draft" / source.name
    print(f"Step 2I: {len(rows)} provisional pairs, {len(selected)} selected")
    print("Source (exported crops):", source)
    print("OCR engines:", ", ".join(engines))
    print("Results folder:", out)
    print("First IDs:", ", ".join(r["auto_id"] for r in selected[:10]))
    if args.dry_run:
        print("DRY RUN: image paths verified; no OCR or files written.")
        return
    exe = tesseract_exe(args.tesseract) if VI in engines else ""
    if HAN in engines:
        try:
            verify_ollama_model(args.model, timeout=15)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit("Cannot connect to local Ollama: " + str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "source_manifest.json"
    if not meta.is_file():
        atomic_json(meta, {
            "source_step1c_run": str(source), "created_at": now_iso(),
            "manifest_count": len(rows), "draft_only": True,
            "model_default": args.model, "tesseract_language": "vie+eng"
        })
    for engine in engines:
        (out / engine).mkdir(parents=True, exist_ok=True)
        errors = 0
        for i, row in enumerate(selected, 1):
            doc_id = row["auto_id"]
            destination = out / engine / f"{doc_id}.json"
            if destination.is_file() and not args.force:
                print(f"[{i}/{len(selected)}] {doc_id} {engine}: cached (skip)")
                continue
            image = valid_crop_path(source, row["viet_image"] if engine == VI else row["han_image"])
            started = time.perf_counter()
            try:
                if engine == VI:
                    result = recognize_vietnamese(image, exe)
                else:
                    ref = out / VI / f"{doc_id}.json"
                    expected = read_json(ref).get("result", {}).get("phien_am_syllables", 0) if ref.is_file() else 0
                    result = recognize_chinese(image, args.model, args.max_side, args.timeout, int(expected))
            except Exception as exc:
                result = {"status": "OCR_ERROR", "flags": ["OCR_EXCEPTION"], "error": str(exc)}
                errors += 1
            data = {
                "doc_id": doc_id, "pdf_page": int(row["pdf_page"]),
                "engine": engine, "source_image": str(image.relative_to(source)),
                "step1c_source_run": source.name, "created_at": now_iso(),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "verified": False, "ready_for_training": False, "result": result
            }
            atomic_json(destination, data)
            print(f"[{i}/{len(selected)}] {doc_id} {engine}: {result['status']} ")
            # Even if interrupted, previous documents are preserved as individual checkpoints.
        update_index(out, rows, engine)
        print(f"{engine}: checkpoint summary written; new errors: {errors}")
    print("Step 2I completed. All text remains UNVERIFIED.")


if __name__ == "__main__":
    main()
