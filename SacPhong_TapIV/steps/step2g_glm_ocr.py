
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "output" / "Sac_phong_Tap_IV"

IMAGE_DIR = BOOK / "02_ocr_tho" / "han"
RESULT_DIR = BOOK / "02_ocr_tho" / "han_glm_candidates"

API = "http://127.0.0.1:11434"
MODEL = "glm-ocr:q8_0"

DEFAULT_DOCS = [
    "PILOT.006.01",
    "PILOT.008.02",
]

# Approximate syllable counts from the current pilot.
# These are diagnostic references, not ground truth.
EXPECTED_COUNTS = {
    "PILOT.006.01": 76,
    "PILOT.008.02": 87,
}


def is_han(char: str) -> bool:
    n = ord(char)
    return (
        0x3400 <= n <= 0x9FFF
        or 0xF900 <= n <= 0xFAFF
        or 0x20000 <= n <= 0x323AF
    )


def extract_candidate(text: str) -> str:
    keep = {"□", "\n", "。", "，", "、", "；", "：", "！", "？"}
    return "".join(
        c for c in text if is_han(c) or c in keep
    ).strip()


def character_count(text: str) -> int:
    return sum(is_han(c) for c in text)


def detect_repetition(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)

    if "□" * 10 in compact:
        return True

    return bool(re.search(r"(.{1,5})\1{7,}", compact))


def prepare_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=94)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def variants_for_document(doc_id: str, image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    if doc_id == "PILOT.006.01":
        width, height = image.size

        right_crop = image.crop((
            int(width * 0.52),
            int(height * 0.02),
            width,
            int(height * 0.98)
        ))

        return [
            ("original", image),
            ("right_text_crop", right_crop),
        ]

    if doc_id == "PILOT.008.02":
        rotated = image.transpose(
            Image.Transpose.ROTATE_270
        )

        return [
            ("rotated_clockwise", rotated),
        ]

    return [("original", image)]


def verify_model(model: str):
    request = urllib.request.Request(API + "/api/tags")

    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)

    names = [
        m.get("name", "")
        for m in data.get("models", [])
    ]

    if model not in names:
        raise RuntimeError(
            f"Model not installed: {model}\n"
            f"Run: ollama pull {model}\n"
            f"Available: {names}"
        )


def ask_model(model: str, image: Image.Image, timeout: int):
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "0s",
        "messages": [{
            "role": "user",
            "content": "Text Recognition:",
            "images": [prepare_image(image)]
        }],
        "options": {
            "temperature": 0,
            "num_predict": 320,
            "num_ctx": 4096,
        }
    }

    request = urllib.request.Request(
        API + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout
    ) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2G: Dedicated Chinese OCR pilot"
    )

    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--docs", nargs="+", default=DEFAULT_DOCS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not IMAGE_DIR.exists():
        raise SystemExit(f"Missing image folder: {IMAGE_DIR}")

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    model_folder = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        args.model
    )

    run_dir = RESULT_DIR / model_folder / f"run_{stamp}"

    jobs = []

    for doc_id in args.docs:
        image_files = sorted(
            IMAGE_DIR.glob(f"{doc_id}_*.jpg")
        )

        for path in image_files:
            with Image.open(path) as source:
                for name, image in variants_for_document(
                    doc_id, source
                ):
                    jobs.append((
                        doc_id,
                        path.name,
                        name,
                        image.copy()
                    ))

    print("STEP 2G - GLM-OCR")
    print("Model:", args.model)
    print("Image variants:", len(jobs))

    for doc_id, filename, variant, image in jobs:
        print(
            f"  {doc_id}: {filename} / {variant}, "
            f"dimensions={image.size}"
        )

    if args.dry_run:
        print("DRY RUN COMPLETE - no model requests.")
        return

    verify_model(args.model)

    run_dir.mkdir(parents=True, exist_ok=False)

    attempts = []

    for doc_id, filename, variant, image in jobs:
        print(f"\nRunning {doc_id} / {variant}")

        safe_doc = doc_id.replace(".", "_")
        image_name = f"{safe_doc}_{variant}_input.jpg"

        image.save(
            run_dir / image_name,
            format="JPEG",
            quality=94
        )

        start = time.perf_counter()

        record = {
            "doc_id": doc_id,
            "source_image": filename,
            "variant": variant,
            "model": args.model,
            "input_image": image_name,
            "expected_syllables": EXPECTED_COUNTS.get(doc_id),
            "verified": False,
            "raw_response": "",
            "chinese_candidate": "",
            "chinese_characters": 0,
            "flags": [],
            "status": "OCR_ERROR",
        }

        try:
            response = ask_model(
                args.model,
                image,
                args.timeout
            )

            raw = str(
                response.get("message", {}).get("content", "")
            )

            candidate = extract_candidate(raw)
            count = character_count(candidate)

            flags = []

            if detect_repetition(raw):
                flags.append("REPETITION_DETECTED")

            if count == 0:
                flags.append("NO_CHINESE_TEXT")

            expected = EXPECTED_COUNTS.get(doc_id)

            if expected and (
                count < expected * 0.65
                or count > expected * 1.35
            ):
                flags.append("COUNT_MISMATCH")

            if "□" in candidate:
                flags.append("UNREADABLE_MARKERS")

            status = "OCR_DRAFT_UNVERIFIED"

            if "REPETITION_DETECTED" in flags:
                status = "REPETITION_REJECTED"

            elif count == 0:
                status = "EMPTY_OCR"

            record.update({
                "raw_response": raw,
                "chinese_candidate": candidate,
                "chinese_characters": count,
                "flags": flags,
                "status": status,
                "done_reason": response.get("done_reason"),
            })

        except urllib.error.HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace"
            )

            record.update({
                "status": "OCR_ERROR",
                "http_status": error.code,
                "server_message": detail,
            })

        except Exception as error:
            record.update({
                "status": "OCR_ERROR",
                "error": str(error),
            })

        record["elapsed_seconds"] = round(
            time.perf_counter() - start, 2
        )

        attempts.append(record)

        print(
            f"  Status: {record['status']}, "
            f"characters: {record['chinese_characters']}"
        )

        report = {
            "model": args.model,
            "created_at": stamp,
            "verified": False,
            "attempts": attempts,
        }

        (run_dir / "results.json").write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    columns = [
        "doc_id",
        "variant",
        "model",
        "status",
        "chinese_characters",
        "expected_syllables",
        "flags",
        "elapsed_seconds",
    ]

    with (run_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for record in attempts:
            writer.writerow({
                field: (
                    "; ".join(record[field])
                    if field == "flags"
                    else record.get(field, "")
                )
                for field in columns
            })

    print("\nSTEP 2G COMPLETED")
    print("Results:", run_dir)
    print("All OCR results remain UNVERIFIED.")


if __name__ == "__main__":
    main()
