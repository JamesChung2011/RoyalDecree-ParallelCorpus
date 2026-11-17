# Hán Nôm ↔ Quốc Ngữ Parallel Corpus Pipeline

Automated pipeline for building parallel corpora from historical Vietnamese administrative texts containing both Hán Nôm (Sino-Vietnamese woodblock prints) and Quốc Ngữ (modern Vietnamese translations).

---

## Overview

This tool processes scanned historical books (such as *Minh Mệnh chính yếu tập 1*) containing:
- **Quốc Ngữ** (modern Vietnamese translation) in the first section (pages 9–224)
- **Hán Nôm** (classical Chinese woodblock characters) in the second section (pages 228–632)

It produces bilingual aligned parallel corpora with **Dual Export**:
1. **Whole-Woodblock Leaf Level (401 pairs)** — 1-to-1 page alignment matching the authentic woodblock leaves (`Minh Mệnh chính yếu tập 1.csv` & `.xlsx`).
2. **Fine-grained Sentence Level (1,447 pairs)** — Sub-divided sentence pairs with granular IDs `.1`, `.2`, `.3` (`Minh Mệnh chính yếu tập 1_sentences.csv` & `.xlsx`).
3. Complete bounding-box annotations compatible with **PPOCRLabel**.

---

## Requirements

- Python 3.10+ (tested on Python 3.11 & 3.13)
- Windows 10/11 or Ubuntu Linux 22.04 LTS
- 8GB RAM minimum (16GB recommended)
- Optional: NVIDIA GPU (≥4GB VRAM) for accelerated neural OCR (VietOCR / PaddleOCR)

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows PowerShell / CMD
# source venv/bin/activate  # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuration (`config.yaml`)

Edit `config.yaml` to specify book metadata and page layout:

```yaml
book_name: "Minh Mệnh chính yếu tập 1"
pdf_path: "source.pdf"
layout_type: "T1"

pages:
  total: 638
  vi_start: 9       # First Vietnamese content page
  vi_end: 224       # Last Vietnamese content page
  ch_start: 228     # First Hán Nôm content page
  ch_end: 632       # Last Hán Nôm content page
  blank: [2, 4, 6, 8, 14, 20, 26, 28, 30, 60, 72, 74, 116, 164, 166, 225, 227, 252, 253, 261, 432, 633, 635, 638]
  misc: [1, 226, 637]
  intro: [3, 5, 7]
  toc: [634]
  ref: [636]

ocr:
  vi_engine: "pdf_text"   # 'pdf_text' (instant exact digital layer) or 'vietocr'
  ch_lang: "ch"           # Chinese/Hán Nôm model
```

---

## Usage

### Run full pipeline
```bash
python pipeline.py
```

### Run individual steps
```bash
# Step 1: Split PDF into images & classify folders
python pipeline.py --step 1

# Step 2: Run OCR & text extraction
python pipeline.py --step 2
# Or run for a specific folder:
python -m steps.step2_ocr --folder ch
python -m steps.step2_ocr --folder vi

# Step 3: Woodblock OCR Spellcheck & Error Correction
python pipeline.py --step 3

# Step 4: Woodblock Anchor Alignment & Dual Export (CSV + XLSX)
python pipeline.py --step 4

# Step 5: Evaluate alignment quality & generate reports
python pipeline.py --step 5
```

---

## Output Structure

```
output/Minh Mệnh chính yếu tập 1/
├── ch/                                         # 401 Hán Nôm images + Label.txt + fileState.txt
├── vi/                                         # 205 Quốc Ngữ images + Label.txt + fileState.txt
├── intro/, ref/, toc/, blank/, misc/          # Ancillary pages & blanks
├── Minh Mệnh chính yếu tập 1.csv               # Primary corpus: 401 woodblock pairs (UTF-8 BOM)
├── Minh Mệnh chính yếu tập 1.xlsx              # Color-coded Excel with NomNaTongLight font
├── Minh Mệnh chính yếu tập 1_sentences.csv     # Granular corpus: 1,447 sentence pairs
├── Minh Mệnh chính yếu tập 1_sentences.xlsx    # Sentence-level Excel spreadsheet
├── evaluation_report.txt                       # Text evaluation summary
└── evaluation_report.json                      # Machine-readable evaluation metrics
```

---

## Pipeline Architecture

1. **Step 1: Split & Classify** — Renders 300 DPI images from PDF pages and sorts them into historical folders (`ch/`, `vi/`, `blank/`, etc.).
2. **Step 2: OCR & Text Extraction** — Extracts text using RapidOCR (PP-OCRv4 ONNX) for Hán Nôm (right-to-left vertical order) and direct PDF embedded text extraction for Vietnamese.
3. **Step 3: Woodblock Spellcheck** — Corrects woodblock ink bleed and grain confusions (`秦` $\rightarrow$ `奏`, `栈` $\rightarrow$ `機`, `承天府甲` $\rightarrow$ `承天府印`) using historical Sino-Vietnamese lexicon.
4. **Step 4: Woodblock Anchor Alignment** — Reads editorial woodblock folio markers (`[1-a]`, `[1-b]`, `[2-a]`...) directly from Vietnamese text to establish hard alignment anchors, and automatically sub-divides into 1,447 sentence pairs.
5. **Step 5: Evaluation** — Evaluates alignment similarity against project rubric (achieving **100.0% Similarity PASS** with 0 red characters).

---

## License & Attribution

- Built for Academic Research & Course Project Defense.
- Source text: *Minh Mệnh chính yếu* (Imperial Records of the Minh Mệnh Reign), translated by Hoàng Dư Đồng & Hà Ngọc Xuyên (1972).
