
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]

BOOK_OUTPUT = ROOT / "output" / "Sac_phong_Tap_IV"

HAN_IMAGES = BOOK_OUTPUT / "02_ocr_tho" / "han"

VLM_OUTPUT = BOOK_OUTPUT / "02_ocr_tho" / "han_vlm_candidates"

API_URL = "http://localhost:11434"

DEFAULT_MODEL = "qwen2.5vl:3b"

DEFAULT_DOCUMENTS = [
    "PILOT.006.01",
    "PILOT.008.02",
]

PROMPT = """
You are an OCR transcription engine for historical Vietnamese
royal decrees written in Classical Chinese.

Your task is transcription, NOT translation.

Rules:
1. Read only Chinese characters visible in the image.
2. Preserve original traditional and variant characters.
3. Do not convert traditional characters to simplified Chinese.
4. For vertical text, read columns from right to left,
   and each column from top to bottom.
5. Preserve the original reading order.
6. Ignore decorative patterns, seals without readable text,
   backgrounds, borders, and empty regions.
7. Do not invent characters based on historical context.
8. If a character is unreadable, use □.
9. Do not reconstruct missing portions of damaged documents.
10. Return only the Chinese transcription.
11. Do not include Vietnamese, romanization, explanations,
    Markdown, or commentary.
12. If no Chinese writing is legible, return an empty string.

Transcribe the visible Chinese text now.
""".strip()


def is_chinese_character(char: str) -> bool:
    value = ord(char)

    return (
        0x3400 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x20000 <= value <= 0x323AF
    )


def count_chinese_characters(text: str) -> int:
    return sum(
        1 for char in text
        if is_chinese_character(char)
    )


def extract_chinese_candidate(text: str) -> str:
    allowed_punctuation = set("□。、，；：！？")

    return "".join(
        char for char in text
        if is_chinese_character(char)
        or char in allowed_punctuation
        or char == "\n"
    ).strip()


def encode_image(image: Image.Image) -> str:
    image = image.convert("RGB")

    image.thumbnail(
        (2048, 2048),
        Image.Resampling.LANCZOS
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=93
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def check_ollama(model: str) -> None:
    request = urllib.request.Request(
        API_URL + "/api/tags",
        method="GET"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=15
        ) as response:
            data = json.load(response)

    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            "Cannot connect to Ollama.\n"
            "Open the Ollama application and try again.\n"
            f"Details: {error}"
        ) from error

    available = [
        item.get("name", "")
        for item in data.get("models", [])
    ]

    if model not in available:
        raise RuntimeError(
            f"Model '{model}' is not installed.\n"
            f"Run: ollama pull {model}\n"
            f"Installed models: {available}"
        )


def call_ollama(
    image: Image.Image,
    model: str,
    timeout: int
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "5m",
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [
                    encode_image(image)
                ]
            }
        ],
        "options": {
            "temperature": 0,
            "num_predict": 2048
        }
    }

    request = urllib.request.Request(
        API_URL + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout
    ) as response:
        result = json.load(response)

    raw_text = str(
        result.get("message", {}).get("content", "")
    )

    candidate = extract_chinese_candidate(raw_text)

    return {
        "raw_response": raw_text,
        "chinese_candidate": candidate,
        "chinese_characters": count_chinese_characters(
            candidate
        ),
        "unreadable_markers": candidate.count("□"),
        "model_reported_done": result.get("done", False)
    }


def create_variants(
    image: Image.Image,
    doc_id: str
) -> list:
    image = ImageOps.exif_transpose(image).convert("RGB")

    variants = [
        ("original", image.copy())
    ]

    if doc_id == "PILOT.006.01":
        width, height = image.size

        text_region = image.crop(
            (
                int(width * 0.54),
                int(height * 0.03),
                width,
                int(height * 0.97)
            )
        )

        variants.append(
            ("right_text_region", text_region)
        )

    if doc_id == "PILOT.008.02":
        rotated = image.transpose(
            Image.Transpose.ROTATE_270
        )

        variants.append(
            ("rotated_90_clockwise", rotated)
        )

    return variants


def get_document_images(doc_id: str) -> list:
    return sorted(
        HAN_IMAGES.glob(f"{doc_id}_*.jpg")
    )


def get_all_document_ids() -> list:
    ids = set()

    for path in HAN_IMAGES.glob("*.jpg"):
        stem = path.stem

        if "_" not in stem:
            continue

        doc_id, number = stem.rsplit("_", 1)

        if number.isdigit():
            ids.add(doc_id)

    return sorted(ids)


def run_document(
    doc_id: str,
    model: str,
    timeout: int,
    force: bool
) -> list:
    images = get_document_images(doc_id)

    destination = VLM_OUTPUT / f"{doc_id}.json"

    if destination.exists() and not force:
        print(
            f"{doc_id}: existing result skipped "
            "(use --force to rerun)"
        )
        return []

    if not images:
        print(f"{doc_id}: no source images found")
        return []

    attempts = []
    summary_rows = []

    for path in images:
        with Image.open(path) as source:
            variants = create_variants(source, doc_id)

        for variant_name, image in variants:
            print(
                f"Processing {doc_id} / "
                f"{path.name} / {variant_name}"
            )

            start = time.perf_counter()

            try:
                result = call_ollama(
                    image=image,
                    model=model,
                    timeout=timeout
                )

                status = "OCR_DRAFT_UNVERIFIED"

            except Exception as error:
                result = {
                    "raw_response": "",
                    "chinese_candidate": "",
                    "chinese_characters": 0,
                    "unreadable_markers": 0,
                    "error": str(error)
                }

                status = "OCR_ERROR"

            elapsed = round(
                time.perf_counter() - start,
                2
            )

            record = {
                "doc_id": doc_id,
                "source_image": path.name,
                "variant": variant_name,
                "model": model,
                "status": status,
                "elapsed_seconds": elapsed,
                **result
            }

            attempts.append(record)

            summary_rows.append({
                "doc_id": doc_id,
                "source_image": path.name,
                "variant": variant_name,
                "model": model,
                "status": status,
                "chinese_characters": result[
                    "chinese_characters"
                ],
                "unreadable_markers": result[
                    "unreadable_markers"
                ],
                "elapsed_seconds": elapsed,
                "candidate_preview": result[
                    "chinese_candidate"
                ].replace("\n", " ")[:120]
            })

            print(
                f"  Result: {status}, "
                f"{result['chinese_characters']} "
                "Chinese characters"
            )

    output = {
        "doc_id": doc_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": model,
        "verified": False,
        "note": (
            "These are OCR candidates only. "
            "Character counts and model output "
            "do not establish transcription accuracy."
        ),
        "attempts": attempts
    }

    destination.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"Saved: {destination}")

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2F: Vision-language Chinese OCR pilot"
        )
    )

    parser.add_argument(
        "--docs",
        nargs="+",
        default=DEFAULT_DOCUMENTS
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all existing Chinese crop images"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900
    )

    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()

    if not HAN_IMAGES.exists():
        raise SystemExit(
            f"Chinese OCR image folder is missing:\n"
            f"{HAN_IMAGES}"
        )

    VLM_OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        check_ollama(args.model)
    except RuntimeError as error:
        raise SystemExit(str(error))

    if args.all:
        document_ids = get_all_document_ids()
    else:
        document_ids = args.docs

    if not document_ids:
        raise SystemExit(
            "No Chinese document crops were found."
        )

    print("=" * 55)
    print("STEP 2F - VISION-LANGUAGE OCR")
    print(f"Model: {args.model}")
    print(f"Documents: {len(document_ids)}")
    print("Existing OCR labels will not be changed.")
    print("=" * 55)

    summary = []

    for doc_id in document_ids:
        rows = run_document(
            doc_id=doc_id,
            model=args.model,
            timeout=args.timeout,
            force=args.force
        )

        summary.extend(rows)

    if summary:
        summary_path = VLM_OUTPUT / "comparison.csv"

        with summary_path.open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "doc_id",
                    "source_image",
                    "variant",
                    "model",
                    "status",
                    "chinese_characters",
                    "unreadable_markers",
                    "elapsed_seconds",
                    "candidate_preview"
                ]
            )

            writer.writeheader()
            writer.writerows(summary)

        print(f"\nComparison saved: {summary_path}")

    print("\nSTEP 2F COMPLETED")
    print("All output is unverified.")
    print("Do not use it as final training data.")


if __name__ == "__main__":
    main()
