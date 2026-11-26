"""
PPOCRLabel format helpers.

Label.txt format (one line per image):
    <folder>/<image>\t[{"transcription":"text","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]

fileState.txt format (one line per image):
    <folder>/<image>\t1
"""

import json
from typing import List, Dict, Tuple


def format_label_line(folder: str, image_name: str,
                      boxes: List[Dict]) -> str:
    """
    Format one line of Label.txt.

    Args:
        folder: folder name (e.g. "ch", "vi")
        image_name: image filename
        boxes: list of dicts with 'transcription' and 'points' keys
            points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    Returns:
        formatted line (without trailing newline)
    """
    img_path = f"{folder}/{image_name}"
    # Ensure each box has exactly the required fields
    cleaned_boxes = []
    for box in boxes:
        entry = {
            "transcription": box["transcription"],
            "points": box["points"]
        }
        # Include optional fields like 'confident', 'score' if present
        for key in box:
            if key not in ("transcription", "points"):
                entry[key] = box[key]
        cleaned_boxes.append(entry)

    boxes_json = json.dumps(cleaned_boxes, ensure_ascii=False)
    return f"{img_path}\t{boxes_json}"


def format_filestate_line(folder: str, image_name: str,
                          completed: bool = True) -> str:
    """Format one line of fileState.txt."""
    img_path = f"{folder}/{image_name}"
    state = 1 if completed else 0
    return f"{img_path}\t{state}"


def write_label_file(output_path: str, lines: List[str]):
    """Write Label.txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_filestate(output_path: str, lines: List[str]):
    """Write fileState.txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def paddleocr_to_ppocrlabel(ocr_result, folder: str,
                             image_name: str) -> Tuple[str, str]:
    """
    Convert PaddleOCR result to PPOCRLabel format lines.

    Args:
        ocr_result: PaddleOCR result for one image
            Format: list of [box, (text, confidence)]
            where box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        folder: folder name
        image_name: image filename

    Returns:
        (label_line, filestate_line)
    """
    boxes = []
    if ocr_result:
        for item in ocr_result:
            if len(item) >= 2:
                points = item[0]
                text = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                confidence = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0.0

                # Ensure points are integers and exactly 4 points
                int_points = [[int(round(p[0])), int(round(p[1]))] for p in points]
                if len(int_points) == 4:
                    boxes.append({
                        "transcription": text,
                        "points": int_points,
                        "confident": round(float(confidence), 4)
                    })

    label_line = format_label_line(folder, image_name, boxes)
    filestate_line = format_filestate_line(folder, image_name, completed=True)
    return label_line, filestate_line

