"""
Step 2: Run OCR / Text Extraction on classified images.

- Hán Nôm (ch/):
    Runs RapidOCR (PP-OCRv4 ONNX model optimized for Chinese / Sino-Vietnamese)
    Columns sorted right-to-left, top-to-bottom.
- Quốc Ngữ (vi/, intro/, ref/, toc/):
    Configurable via config.yaml:
      1. 'vietocr' (Default): Hybrid pipeline using RapidOCR for accurate line detection
         and VietOCR (VGG-Transformer) for state-of-the-art Vietnamese text recognition.
      2. 'pdf_text': Extracts directly from PDF embedded text layer.
    Lines sorted left-to-right, top-to-bottom.

Outputs Label.txt and fileState.txt in strict PPOCRLabel format for each folder.

Usage:
    python -m steps.step2_ocr [--folder ch|vi|all] [--config config.yaml] [--force]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pymupdf
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ppocrlabel import (
    format_label_line, format_filestate_line,
    write_label_file, write_filestate
)
from utils.text_cleaning import (
    clean_nom_text, clean_vn_text,
    remove_page_numbers, remove_headers, is_noise_box
)


def load_config(config_path="config.yaml"):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_page_num(filename: str) -> int:
    import re
    match = re.search(r'\.(\d{3,})\.jpg$', filename)
    return int(match.group(1)) if match else 0


def sort_boxes_nom(boxes):
    """
    Sort bounding boxes for Nôm text: right-to-left columns, top-to-bottom.
    Each box is [points, (text, confidence)]
    """
    if not boxes:
        return boxes

    def get_center(item):
        pts = item[0]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        return cx, cy

    def get_box_dim(item):
        pts = item[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(xs) - min(xs), max(ys) - min(ys)

    vertical_count = sum(1 for b in boxes if get_box_dim(b)[1] > get_box_dim(b)[0] * 1.2)
    is_vertical = vertical_count >= len(boxes) * 0.4

    if is_vertical:
        return sorted(boxes, key=lambda b: (-get_center(b)[0], get_center(b)[1]))
    else:
        return sorted(boxes, key=lambda b: (get_center(b)[1], get_center(b)[0]))


def sort_boxes_vn(boxes):
    """
    Sort bounding boxes for Vietnamese: top-to-bottom, left-to-right.
    """
    if not boxes:
        return boxes

    def get_center(item):
        pts = item[0]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        return cx, cy

    return sorted(boxes, key=lambda b: (get_center(b)[1], get_center(b)[0]))


def extract_vn_from_pdf(pdf_doc: pymupdf.Document, page_num: int, img_width: int, img_height: int):
    """
    Extract line bounding boxes and text from PDF text layer for a Vietnamese page.
    Scales coordinates to match the image dimensions (300 DPI).
    """
    page_idx = page_num - 1
    if page_idx < 0 or page_idx >= pdf_doc.page_count:
        return []

    page = pdf_doc[page_idx]
    rect = page.rect
    scale_x = img_width / rect.width if rect.width > 0 else 1.0
    scale_y = img_height / rect.height if rect.height > 0 else 1.0

    d = page.get_text("dict")
    boxes = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:  # text only
            continue
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not line_text:
                continue

            bbox = line["bbox"]  # (x0, y0, x1, y1)
            x0 = int(round(bbox[0] * scale_x))
            y0 = int(round(bbox[1] * scale_y))
            x1 = int(round(bbox[2] * scale_x))
            y1 = int(round(bbox[3] * scale_y))

            # Filter running headers (y1 <= 260) and scan noise
            if y1 <= 260 or is_noise_box(line_text):
                continue

            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            boxes.append([points, (line_text, 1.0)])

    return boxes


def get_vietocr_predictor():
    """Initialize VietOCR predictor with cached weights and optimized settings."""
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    config = Cfg.load_config_from_name("vgg_transformer")
    weights_path = os.path.expanduser("~/.vietocr/vgg_transformer.pth")
    if os.path.exists(weights_path):
        config["weights"] = weights_path
    config["cnn"]["pretrained"] = False
    config["device"] = "cpu"
    return Predictor(config)


def ocr_vn_with_vietocr(img_path: str, rapid_engine, vietocr_predictor):
    """
    Recognize Vietnamese text using RapidOCR for line detection + VietOCR for recognition.
    """
    result, _ = rapid_engine(img_path)
    if not result:
        return []

    im = Image.open(img_path)
    img_w, img_h = im.size

    raw_boxes = [[item[0], (item[1], float(item[2]))] for item in result]
    sorted_boxes = sort_boxes_vn(raw_boxes)

    valid_boxes = []
    crops = []

    for box_entry in sorted_boxes:
        pts = box_entry[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, int(round(min(xs))))
        y0 = max(0, int(round(min(ys))))
        x1 = min(img_w, int(round(max(xs))))
        y1 = min(img_h, int(round(max(ys))))

        # Filter running headers (y1 <= 260)
        if y1 <= 260 or (y1 - y0) < 8 or (x1 - x0) < 8:
            continue

        int_points = [[int(round(p[0])), int(round(p[1]))] for p in pts]
        crop = im.crop((x0, y0, x1, y1))
        crops.append(crop)
        valid_boxes.append((int_points, box_entry[1][1], y1))

    if not crops:
        im.close()
        return []

    # Batch prediction with VietOCR
    try:
        texts = vietocr_predictor.predict_batch(crops)
    except Exception:
        # Fallback to single-line prediction if batch fails
        texts = [vietocr_predictor.predict(c) for c in crops]

    im.close()

    extracted = []
    for (int_points, conf, y1), text in zip(valid_boxes, texts):
        cleaned = clean_vn_text(text)
        if is_noise_box(cleaned, min_chars=1, y_max=y1):
            continue
        extracted.append([int_points, (cleaned, round(conf, 4))])

    return extracted


def process_folder(folder_name: str, cfg: dict, force: bool = False):
    book_name = cfg["book_name"]
    output_dir = Path(cfg["output_dir"]) / book_name / folder_name
    pdf_path = cfg["pdf_path"]
    ocr_cfg = cfg.get("ocr", {})
    vi_engine_type = ocr_cfg.get("vi_engine", "vietocr")

    if not output_dir.exists():
        print(f"Folder not found: {output_dir}")
        return

    images = sorted(
        [f for f in output_dir.iterdir() if f.suffix.lower() == ".jpg"],
        key=lambda x: extract_page_num(x.name)
    )
    if not images:
        print(f"No images found in {output_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Text Extraction / OCR: {folder_name}/ ({len(images)} images)")
    if folder_name != "ch":
        print(f"  Vietnamese Engine: {vi_engine_type.upper()}")
    else:
        print(f"  Nôm Engine: RapidOCR (PP-OCRv4)")
    print(f"{'='*60}")

    label_path = output_dir / "Label.txt"
    filestate_path = output_dir / "fileState.txt"
    existing_labels = {}

    is_vn_folder = (folder_name != "ch")
    if force:
        if label_path.exists():
            label_path.unlink()
        if filestate_path.exists():
            filestate_path.unlink()
    elif label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    img_name = os.path.basename(parts[0])
                    existing_labels[img_name] = parts[1]

    if not label_path.exists():
        label_path.touch()
    if not filestate_path.exists():
        filestate_path.touch()

    rapid_engine = None
    vietocr_predictor = None
    pdf_doc = None

    from rapidocr_onnxruntime import RapidOCR
    rapid_engine = RapidOCR()

    if is_vn_folder:
        if vi_engine_type == "vietocr":
            vietocr_predictor = get_vietocr_predictor()
        else:
            pdf_doc = pymupdf.open(pdf_path)

    label_file = open(str(label_path), "a", encoding="utf-8")
    filestate_file = open(str(filestate_path), "a", encoding="utf-8")

    processed_count = 0
    try:
        for img_path in tqdm(images, desc=f"Processing {folder_name}"):
            img_name = img_path.name
            page_num = extract_page_num(img_name)

            if img_name in existing_labels:
                continue

            boxes = []
            try:
                if folder_name == "ch":
                    result, _ = rapid_engine(str(img_path))
                    if result:
                        raw_boxes = [[item[0], (item[1], float(item[2]))] for item in result]
                        sorted_boxes = sort_boxes_nom(raw_boxes)

                        for box_entry in sorted_boxes:
                            pts = box_entry[0]
                            text = box_entry[1][0]
                            conf = box_entry[1][1]

                            cleaned = clean_nom_text(text)
                            cleaned = remove_page_numbers(cleaned)

                            if is_noise_box(cleaned):
                                continue

                            int_points = [[int(round(p[0])), int(round(p[1]))] for p in pts]
                            if len(int_points) == 4:
                                boxes.append({
                                    "transcription": cleaned,
                                    "points": int_points,
                                    "confident": round(conf, 4)
                                })
                else:
                    if vi_engine_type == "vietocr":
                        extracted = ocr_vn_with_vietocr(str(img_path), rapid_engine, vietocr_predictor)
                        for pts, (text, conf) in extracted:
                            cleaned = clean_vn_text(text)
                            cleaned = remove_headers(cleaned, book_name)
                            cleaned = remove_page_numbers(cleaned)
                            if is_noise_box(cleaned):
                                continue
                            boxes.append({
                                "transcription": cleaned,
                                "points": pts,
                                "confident": conf
                            })
                    else:
                        with Image.open(str(img_path)) as im:
                            img_w, img_h = im.size
                        extracted = extract_vn_from_pdf(pdf_doc, page_num, img_w, img_h)
                        sorted_boxes = sort_boxes_vn(extracted)

                        for box_entry in sorted_boxes:
                            pts = box_entry[0]
                            text = box_entry[1][0]
                            conf = box_entry[1][1]

                            cleaned = clean_vn_text(text)
                            cleaned = remove_headers(cleaned, book_name)
                            cleaned = remove_page_numbers(cleaned)

                            if is_noise_box(cleaned):
                                continue

                            boxes.append({
                                "transcription": cleaned,
                                "points": pts,
                                "confident": round(conf, 4)
                            })

            except Exception as e:
                print(f"\n  Error processing {img_name}: {e}")

            boxes_json = json.dumps(boxes, ensure_ascii=False)
            label_file.write(f"{folder_name}/{img_name}\t{boxes_json}\n")
            label_file.flush()
            filestate_file.write(f"{folder_name}/{img_name}\t1\n")
            filestate_file.flush()
            existing_labels[img_name] = boxes_json
            processed_count += 1
    finally:
        label_file.close()
        filestate_file.close()
        if pdf_doc:
            pdf_doc.close()

    print(f"\n✓ Completed {folder_name}/: {len(existing_labels)} total entries in Label.txt ({processed_count} newly processed).")
    print(f"  Label file: {label_path}")
    print(f"  FileState:  {filestate_path}")


def main():
    parser = argparse.ArgumentParser(description="Run OCR on classified images")
    parser.add_argument("--folder", default="all",
                        choices=["ch", "vi", "intro", "toc", "ref", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-run and overwrite existing Label.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.folder == "all":
        folders = ["vi", "ch", "intro", "toc", "ref"]
    else:
        folders = [args.folder]

    for f in folders:
        process_folder(f, cfg, force=args.force)


if __name__ == "__main__":
    main()
