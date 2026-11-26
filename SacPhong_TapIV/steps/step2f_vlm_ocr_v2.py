"""
Step 2F v2 - isolated Chinese OCR diagnostic pilot.

- Runs Ollama locally; no extra Python packages beyond Pillow.
- Saves output in unique, model-specific run directories.
- Preserves original Label.txt, Step 2E reviews and 3B/7B old results.
- Detects obvious repetition and gross length mismatch with Phiên âm.
- Captures HTTP error response body, not just "HTTP 500".
- Never labels generated text as verified.

Test first:
    python -m steps.step2f_vlm_ocr_v2 --dry-run
    python -m steps.step2f_vlm_ocr_v2 --model qwen2.5vl:7b

Optional:
    python -m steps.step2f_vlm_ocr_v2 --model qwen2.5vl:3b --max-side 960
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"
HAN = BOOK / "02_ocr_tho" / "han"
VIET_LABEL = BOOK / "02_ocr_tho" / "viet" / "Label.txt"
OUT = BOOK / "02_ocr_tho" / "han_vlm_v2"
OLLAMA = "http://127.0.0.1:11434"
PILOT = ["PILOT.006.01", "PILOT.008.02"]

PROMPT = """Transcribe ONLY the Chinese characters visibly written on this historical
Vietnamese royal decree. This is transcription, NOT translation or restoration.

Rules:
- Do not fabricate words or reconstruct torn, faded, or missing content.
- Keep traditional and variant characters; never convert to simplified Chinese.
- Preserve reading order: vertical columns right-to-left, top-to-bottom.
- Ignore silk patterns, seals, borders, and blank areas.
- Use □ for a single unreadable character only when a character is visibly present.
- Do not repeat phrases or fill missing passages with plausible historical text.
- Output only the transcribed Chinese characters, one column per line.
- If nothing is legible, output an empty response.
"""

def chinese_only(text: str) -> str:
    def is_han(char: str) -> bool:
        value = ord(char)
        return (0x3400 <= value <= 0x9FFF
                or 0xF900 <= value <= 0xFAFF
                or 0x20000 <= value <= 0x323AF)

    keep = set("□，。、；：！？\n")
    return "".join(c for c in text if is_han(c) or c in keep).strip()


def han_count(text: str) -> int:
    return sum(
        (0x3400 <= ord(c) <= 0x9FFF
         or 0xF900 <= ord(c) <= 0xFAFF
         or 0x20000 <= ord(c) <= 0x323AF)
        for c in text
    )


def has_repetition(text: str) -> tuple[bool, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    common = Counter(lines).most_common(1)
    if common and common[0][1] >= 5:
        return True, f"Identical line repeats {common[0][1]} times"
    compact = re.sub(r"\s+", "", text)
    for group_size in range(2, 9):
        pattern = r"(.{" + str(group_size) + r"})(?:\1){5,}"
        if re.search(pattern, compact):
            return True, f"A {group_size}-character phrase repeats consecutively"
    return False, ""


def heading(text: str) -> str:
    value = unicodedata.normalize("NFD", text.lower())
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", value)


def get_phonetic_count(doc_id: str) -> int | None:
    """Approximate count from printed Phiên âm OCR, not a verified reference."""
    if not VIET_LABEL.exists():
        return None

    lines = []
    for line in VIET_LABEL.read_text(encoding="utf-8-sig").splitlines():
        if "\t" not in line:
            continue
        img, raw_boxes = line.split("\t", 1)
        if Path(img.replace("\\", "/")).name != f"{doc_id}_1.jpg":
            continue
        try:
            boxes = json.loads(raw_boxes)
        except json.JSONDecodeError:
            return None
        for box in boxes:
            pts = box.get("points") or []
            y = min((p[1] for p in pts), default=0)
            x = min((p[0] for p in pts), default=0)
            lines.append((float(y), float(x), str(box.get("transcription", "")).strip()))

    lines.sort()
    section = False
    chunks = []
    for _, _, text in lines:
        normalized = heading(text)
        if normalized.startswith("phienam"):
            section = True
            continue
        if normalized.startswith("dichnghia"):
            section = False
            break
        if section:
            chunks.append(text)

    if not chunks:
        return None

    text = " ".join(chunks)
    text = re.sub(r"\[[^]]*\]|\([^)]*\)", " ", text)
    return len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))


def image_variants(doc_id: str, image: Image.Image, all_variants: bool):
    """Two pilot cases; intentionally not a general full-book layout detector."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size

    if doc_id == "PILOT.006.01":
        first = ("right_text_crop", image.crop((int(w * 0.47), 0, w, h)))
        if all_variants:
            return [first, ("full_original", image)]
        return [first]

    if doc_id == "PILOT.008.02":
        rotated = image.transpose(Image.Transpose.ROTATE_270)
        first = ("rotated_90_clockwise", rotated)
        if all_variants:
            return [first, ("original", image)]
        return [first]

    return [("original", image)]


def encode_image(image: Image.Image, max_side: int) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=91, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def installed_models() -> list[str]:
    request = urllib.request.Request(OLLAMA + "/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return [m.get("name", "") for m in payload.get("models", [])]


def predict(
    image: Image.Image,
    model: str,
    max_side: int,
    num_predict: int,
    timeout: int,
    keep_alive: str,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [encode_image(image, max_side)]
            }
        ],
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": num_predict,
            "repeat_penalty": 1.15,
        },
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)

    message = result.get("message") or {}
    return str(message.get("content", "")), {
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "eval_count": result.get("eval_count"),
        "prompt_eval_count": result.get("prompt_eval_count"),
    }


def error_info(exc: Exception) -> dict:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            message = exc.read(4000).decode("utf-8", errors="replace")
        except Exception:
            message = ""
        return {
            "error_type": "HTTP_ERROR",
            "http_status": exc.code,
            "error": str(exc),
            "server_message": message[:2500],
        }
    return {
        "error_type": type(exc).__name__,
        "error": str(exc)[:1200],
    }


def run_attempt(
    image: Image.Image, doc_id: str, variant: str,
    model: str, expected: int | None, args,
) -> dict:
    record = {
        "doc_id": doc_id,
        "variant": variant,
        "model": model,
        "expected_phien_am_syllables": expected,
        "status": "OCR_ERROR",
        "verified": False,
        "raw_response": "",
        "chinese_candidate": "",
        "chinese_characters": 0,
        "elapsed_seconds": 0.0,
        "flags": [],
    }
    start = time.perf_counter()

    for attempt_no in range(1, args.retries + 2):
        try:
            raw, metadata = predict(
                image=image,
                model=model,
                max_side=args.max_side,
                num_predict=args.num_predict,
                timeout=args.timeout,
                keep_alive=args.keep_alive,
            )

            candidate = chinese_only(raw)
            count = han_count(candidate)
            repeat, repetition_reason = has_repetition(candidate)

            flags = []
            if repeat:
                flags.append(repetition_reason)
            if count == 0:
                flags.append("No Chinese characters returned")
            if expected:
                ratio = count / expected
                if ratio < 0.80:
                    flags.append("Likely incomplete: fewer characters than Phiên âm")
                if ratio > 1.50:
                    flags.append("Suspiciously long: more characters than Phiên âm")
            if "□" in candidate:
                flags.append("Uncertain characters: □ used")

            if repeat:
                status = "REPETITION_REJECTED"
            elif expected and count > expected * 1.50:
                status = "LENGTH_REJECTED"
            elif count == 0:
                status = "EMPTY_OCR"
            else:
                status = "OCR_DRAFT_UNVERIFIED"

            record.update({
                "status": status,
                "raw_response": raw,
                "chinese_candidate": candidate,
                "chinese_characters": count,
                "flags": flags,
                "generation_metadata": metadata,
                "attempt_number": attempt_no,
            })
            break

        except Exception as exc:
            record.update(error_info(exc))
            record["attempt_number"] = attempt_no
            record["status"] = "OCR_ERROR"
            if attempt_no < args.retries + 1:
                time.sleep(2)
            else:
                break

    record["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    return record


def make_run_directory(model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = OUT / safe / ("run_" + stamp)
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2F v2: safe Ollama Chinese OCR comparison pilot"
    )
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--docs", nargs="+", default=PILOT)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--num-predict", type=int, default=220)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-alive", default="0s")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--all-variants", action="store_true",
        help="Include full image variants (more requests, more memory)."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_side < 256 or args.max_side > 2048:
        parser.error("--max-side must be between 256 and 2048")
    if args.num_predict < 32 or args.num_predict > 2048:
        parser.error("--num-predict must be between 32 and 2048")
    if args.retries < 0 or args.retries > 2:
        parser.error("--retries must be 0, 1 or 2")

    if not HAN.is_dir():
        raise SystemExit(f"Missing Chinese image directory: {HAN}")

    jobs = []
    for doc_id in args.docs:
        images = sorted(HAN.glob(f"{doc_id}_*.jpg"))
        if not images:
            print(f"WARNING: No images found for {doc_id}")
            continue
        expected = get_phonetic_count(doc_id)
        for image_path in images:
            with Image.open(image_path) as source:
                for name, variant in image_variants(
                    doc_id, source, args.all_variants
                ):
                    jobs.append((doc_id, image_path.name, name, variant, expected))

    print(f"Model: {args.model}")
    print(f"Pilot images / attempts: {len(jobs)}")
    print(f"Maximum image side: {args.max_side}px")
    for doc_id, image_name, name, variant, expected in jobs:
        print(
            f"  {doc_id}: {name} ({variant.width}x{variant.height}), "
            f"approximate Phiên âm syllables: {expected}"
        )

    if not jobs:
        raise SystemExit("No pilot images found")

    if args.dry_run:
        print("DRY RUN ONLY: No Ollama requests or output files generated.")
        return

    try:
        available = installed_models()
    except Exception as exc:
        raise SystemExit(
            "Cannot contact Ollama at http://127.0.0.1:11434. "
            f"Is Ollama running? Details: {exc}"
        )

    if args.model not in available:
        raise SystemExit(
            f"Model {args.model!r} not installed. Run: ollama pull {args.model}"
        )

    dest = make_run_directory(args.model)
    report = {
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verified": False,
        "settings": vars(args),
        "attempts": [],
    }

    for doc_id, image_name, variant_name, image, expected in jobs:
        print(f"Running {doc_id} / {variant_name} ...", flush=True)
        attempt = run_attempt(
            image, doc_id, variant_name, args.model, expected, args
        )
        attempt["source_image"] = image_name
        report["attempts"].append(attempt)
        print(
            f"  {attempt['status']}: {attempt['chinese_characters']} chars"
            f" ({attempt['elapsed_seconds']}s)"
        )
        if "server_message" in attempt:
            print(f"  Server detail: {attempt['server_message'][:300]}")

        # Write partial work after every attempt so crashes do not lose results.
        (dest / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    fields = [
        "doc_id", "source_image", "variant", "model", "status",
        "expected_phien_am_syllables", "chinese_characters",
        "elapsed_seconds", "flags", "error", "server_message",
    ]
    with (dest / "comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for attempt in report["attempts"]:
            row = {
                field: attempt.get(field, "")
                for field in fields
            }
            row["flags"] = "; ".join(attempt.get("flags", []))
            writer.writerow(row)

    print(f"\nSaved to: {dest}")
    print("All transcriptions are UNVERIFIED and must not be used as training data.")


if __name__ == "__main__":
    main()
