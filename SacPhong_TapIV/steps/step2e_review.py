
from __future__ import annotations

import json
import re
import shutil
import tkinter as tk
import unicodedata
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from PIL import Image, ImageTk


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "output" / "Sac_phong_Tap_IV"
OCR = OUTPUT / "02_ocr_tho"
HAN = OCR / "han"
VIET = OCR / "viet"
REVIEW = OCR / "manual_review"
HISTORY = REVIEW / "history"
DICTIONARIES = PROJECT / "dictionaries"

TONES = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def is_han(char):
    value = ord(char)
    return (
        0x3400 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x20000 <= value <= 0x323AF
    )


def han_characters(text):
    return [char for char in text if is_han(char)]


def syllables(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)


def normalize_reading(value):
    value = unicodedata.normalize("NFD", str(value).casefold().strip())
    tone = "".join(char for char in value if char in TONES)
    base = "".join(char for char in value if char not in TONES)
    return base + tone


def normalize_heading(value):
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z]", "", value)


def document_id(filename):
    match = re.fullmatch(r"(.+)_([0-9]+)\.jpg", filename, re.I)
    return match.group(1) if match else None


def load_labels(folder):
    path = folder / "Label.txt"
    results = {}

    if not path.is_file():
        return results

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "\t" not in line:
            continue

        image_path, raw_json = line.split("\t", 1)
        doc_id = document_id(Path(image_path).name)

        if not doc_id:
            continue

        try:
            boxes = json.loads(raw_json)
        except json.JSONDecodeError:
            boxes = []

        if not isinstance(boxes, list):
            boxes = []

        results.setdefault(doc_id, []).extend(
            box for box in boxes
            if isinstance(box, dict)
        )

    return results


def box_position(box):
    points = box.get("points") or []
    if not points:
        return 0.0, 0.0

    x = sum(point[0] for point in points) / len(points)
    y = min(point[1] for point in points)
    return float(x), float(y)


def chinese_ocr_text(boxes):
    ordered = sorted(
        boxes,
        key=lambda box: (
            -box_position(box)[0],
            box_position(box)[1]
        )
    )

    return "\n".join(
        str(box.get("transcription", "")).strip()
        for box in ordered
        if str(box.get("transcription", "")).strip()
    )


def vietnamese_ocr_text(boxes):
    ordered = sorted(
        boxes,
        key=lambda box: (
            box_position(box)[1],
            box_position(box)[0]
        )
    )

    reading = []
    translation = []
    section = None

    for box in ordered:
        value = str(box.get("transcription", "")).strip()

        if not value:
            continue

        heading = normalize_heading(value)

        if heading.startswith("phienam"):
            section = "reading"
            continue

        if heading.startswith("dichnghia"):
            section = "translation"
            continue

        if section == "reading":
            reading.append(value)

        elif section == "translation":
            translation.append(value)

    return "\n".join(reading), "\n".join(translation)


def load_dictionary():
    files = sorted(
        DICTIONARIES.glob("*QuocNgu_SinoNom_TongHop3.xlsx")
    )

    if not files:
        raise FileNotFoundError(
            "QuocNgu_SinoNom_TongHop3.xlsx was not found "
            "inside the dictionaries folder."
        )

    from openpyxl import load_workbook

    workbook = load_workbook(
        files[0],
        read_only=True,
        data_only=True
    )

    char_to_readings = {}
    reading_to_chars = {}

    for sheet in workbook.worksheets:
        header_row = None
        han_index = None
        viet_index = None

        for row_number, row in enumerate(
            sheet.iter_rows(max_row=10, values_only=True),
            start=1
        ):
            headings = [normalize_heading(str(v or "")) for v in row]

            if "sinonom" in headings and "quocngu" in headings:
                header_row = row_number
                han_index = headings.index("sinonom")
                viet_index = headings.index("quocngu")
                break

        if header_row is None:
            continue

        for row in sheet.iter_rows(
            min_row=header_row + 1,
            values_only=True
        ):
            if len(row) <= max(han_index, viet_index):
                continue

            han_value = row[han_index]
            viet_value = row[viet_index]

            if han_value is None or viet_value is None:
                continue

            chars = han_characters(str(han_value))

            readings = re.split(
                r"[,;/|、，]+",
                str(viet_value)
            )

            for reading in readings:
                reading = normalize_reading(reading)

                if not reading:
                    continue

                for char in chars:
                    char_to_readings.setdefault(
                        char, set()
                    ).add(reading)

                    reading_to_chars.setdefault(
                        reading, set()
                    ).add(char)

    workbook.close()
    return char_to_readings, reading_to_chars


class ReviewApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sắc phong Tập IV - Step 2E Transcription Review")
        self.geometry("1450x900")
        self.minsize(1050, 650)

        self.han_labels = load_labels(HAN)
        self.viet_labels = load_labels(VIET)

        self.images = {}

        for path in sorted(HAN.glob("*.jpg")):
            doc_id = document_id(path.name)
            if doc_id:
                self.images.setdefault(doc_id, []).append(path)

        self.document_ids = sorted(self.images)

        if not self.document_ids:
            raise RuntimeError(
                "No Chinese decree images found in:\n" + str(HAN)
            )

        REVIEW.mkdir(parents=True, exist_ok=True)
        HISTORY.mkdir(parents=True, exist_ok=True)

        self.current = None
        self.image_index = 0
        self.rotation = 0
        self.image_original = None
        self.image_tk = None

        self.dictionary = None
        self.reverse_dictionary = None

        self.doc_var = tk.StringVar()
        self.reviewed_var = tk.BooleanVar(value=False)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_interface()
        self._open_document(self.document_ids[0])

        self.bind("<Control-s>", self._save_shortcut)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_interface(self):
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Document:"
        ).pack(side="left", padx=(0, 5))

        self.doc_picker = ttk.Combobox(
            header,
            textvariable=self.doc_var,
            values=self.document_ids,
            state="readonly",
            width=22
        )
        self.doc_picker.pack(side="left", padx=5)
        self.doc_picker.bind(
            "<<ComboboxSelected>>",
            self._document_changed
        )

        ttk.Button(
            header,
            text="Previous",
            command=lambda: self._navigate(-1)
        ).pack(side="left", padx=5)

        ttk.Button(
            header,
            text="Next",
            command=lambda: self._navigate(1)
        ).pack(side="left", padx=5)

        ttk.Button(
            header,
            text="Check readings",
            command=self._check_readings
        ).pack(side="left", padx=15)

        ttk.Button(
            header,
            text="Save review",
            command=self.save
        ).pack(side="right", padx=5)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=5)

        left = ttk.Frame(main)
        right = ttk.Frame(main)

        main.add(left, weight=3)
        main.add(right, weight=4)

        image_toolbar = ttk.Frame(left)
        image_toolbar.pack(fill="x", pady=5)

        ttk.Button(
            image_toolbar,
            text="Rotate 90°",
            command=self._rotate
        ).pack(side="left", padx=3)

        ttk.Button(
            image_toolbar,
            text="Next image",
            command=self._next_image
        ).pack(side="left", padx=3)

        ttk.Label(
            image_toolbar,
            text="Zoom"
        ).pack(side="left", padx=(15, 5))

        ttk.Scale(
            image_toolbar,
            from_=0.5,
            to=4.0,
            variable=self.zoom_var,
            command=lambda value: self._draw_image()
        ).pack(side="left", fill="x", expand=True)

        image_frame = ttk.Frame(left)
        image_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            image_frame,
            bg="#363636",
            highlightthickness=0
        )

        vertical = ttk.Scrollbar(
            image_frame,
            orient="vertical",
            command=self.canvas.yview
        )

        horizontal = ttk.Scrollbar(
            image_frame,
            orient="horizontal",
            command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set
        )

        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(
                -int(event.delta / 120),
                "units"
            )
        )

        ttk.Label(
            left,
            textvariable=self.info_var,
            wraplength=600
        ).pack(fill="x", pady=5)

        self.han_text = self._text_section(
            right,
            "Chinese transcription — editable OCR draft",
            10,
            "Microsoft YaHei"
        )

        self.reading_text = self._text_section(
            right,
            "Phiên âm — from the printed book",
            6,
            "Segoe UI"
        )

        self.translation_text = self._text_section(
            right,
            "Dịch nghĩa — preserve the book's translation",
            7,
            "Segoe UI"
        )

        self.notes_text = self._text_section(
            right,
            "Review notes",
            3,
            "Segoe UI"
        )

        ttk.Checkbutton(
            right,
            text="I manually checked this transcription against the original image",
            variable=self.reviewed_var
        ).pack(anchor="w", pady=8)

        self.check_output = scrolledtext.ScrolledText(
            right,
            height=5,
            wrap="word",
            font=("Consolas", 10)
        )
        self.check_output.pack(fill="both", expand=True, pady=5)
        self.check_output.configure(state="disabled")

        ttk.Label(
            self,
            textvariable=self.status_var,
            padding=8
        ).pack(fill="x")

    def _text_section(self, parent, title, height, font):
        ttk.Label(
            parent,
            text=title,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=(5, 2))

        widget = scrolledtext.ScrolledText(
            parent,
            height=height,
            wrap="word",
            font=(font, 12)
        )

        widget.pack(fill="both", expand=True, pady=(0, 5))
        return widget

    def _set_text(self, widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def _get_text(self, widget):
        return unicodedata.normalize(
            "NFC",
            widget.get("1.0", "end-1c").strip()
        )

    def _state(self):
        return {
            "han": self._get_text(self.han_text),
            "phien_am": self._get_text(self.reading_text),
            "dich_nghia": self._get_text(self.translation_text),
            "notes": self._get_text(self.notes_text),
            "reviewed": bool(self.reviewed_var.get())
        }

    def _open_document(self, doc_id):
        self.current = doc_id
        self.doc_var.set(doc_id)

        raw_han = chinese_ocr_text(
            self.han_labels.get(doc_id, [])
        )

        raw_reading, raw_translation = vietnamese_ocr_text(
            self.viet_labels.get(doc_id, [])
        )

        path = REVIEW / f"{doc_id}.json"

        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))

            han = saved.get("han", raw_han)
            reading = saved.get("phien_am", raw_reading)
            translation = saved.get("dich_nghia", raw_translation)
            notes = saved.get("notes", "")
            reviewed = saved.get("reviewed", False)
        else:
            han = raw_han
            reading = raw_reading
            translation = raw_translation
            notes = ""
            reviewed = False

        self._set_text(self.han_text, han)
        self._set_text(self.reading_text, reading)
        self._set_text(self.translation_text, translation)
        self._set_text(self.notes_text, notes)

        self.reviewed_var.set(reviewed)
        self.baseline = self._state()

        self.image_index = 0
        self.rotation = 0
        self.zoom_var.set(1.0)

        self._load_image()
        self._show_check("No dictionary check performed yet.")

        self.status_var.set(
            f"Loaded {doc_id}. Changes are not saved until Save review."
        )

    def _load_image(self):
        paths = self.images.get(self.current, [])

        if not paths:
            return

        path = paths[self.image_index]

        with Image.open(path) as image:
            self.image_original = image.convert("RGB")

        self.rotation = 0
        self._draw_image()

        self.info_var.set(
            f"{path.name} | image {self.image_index + 1}/{len(paths)} "
            "| Blue/yellow decorations and seals may obscure characters."
        )

    def _draw_image(self):
        if self.image_original is None:
            return

        image = self.image_original.rotate(
            -self.rotation,
            expand=True
        )

        fit = min(
            700 / image.width,
            700 / image.height,
            1.0
        )

        scale = max(0.05, fit * self.zoom_var.get())

        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        self.image_tk = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0,
            anchor="nw",
            image=self.image_tk
        )
        self.canvas.configure(
            scrollregion=(0, 0, width, height)
        )

    def _rotate(self):
        self.rotation = (self.rotation + 90) % 360
        self._draw_image()

    def _next_image(self):
        paths = self.images.get(self.current, [])
        if not paths:
            return

        self.image_index = (self.image_index + 1) % len(paths)
        self._load_image()

    def _show_check(self, value):
        self.check_output.configure(state="normal")
        self.check_output.delete("1.0", "end")
        self.check_output.insert("1.0", value)
        self.check_output.configure(state="disabled")

    def _check_readings(self):
        chars = han_characters(self._get_text(self.han_text))
        words = syllables(self._get_text(self.reading_text))

        lines = [
            f"Chinese characters: {len(chars)}",
            f"Phiên âm syllables: {len(words)}"
        ]

        if len(chars) != len(words):
            lines.append(
                "WARNING: Counts differ. Positional matching "
                "would be unreliable."
            )
            lines.append(
                "Check for missing OCR characters or reading syllables."
            )
            self._show_check("\n".join(lines))
            return

        if not chars:
            self._show_check(
                "No Chinese characters available for comparison."
            )
            return

        if self.dictionary is None:
            try:
                self.status_var.set(
                    "Loading the Hán-Việt dictionary..."
                )
                self.update_idletasks()

                (
                    self.dictionary,
                    self.reverse_dictionary
                ) = load_dictionary()

            except Exception as error:
                self._show_check(
                    "Could not load dictionary:\n" + str(error)
                )
                return

        matches = 0
        mismatches = []

        for position, (char, word) in enumerate(
            zip(chars, words),
            start=1
        ):
            reading = normalize_reading(word)
            accepted = self.dictionary.get(char, set())

            if reading in accepted:
                matches += 1
                continue

            alternatives = sorted(
                self.reverse_dictionary.get(
                    reading, set()
                )
            )

            mismatches.append(
                f"{position}: {char} → {word} "
                f"| Possible characters: "
                f"{''.join(alternatives[:12]) or '(none)'}"
            )

        percentage = matches / len(chars) * 100

        lines.append(
            f"Dictionary-compatible positions: "
            f"{matches}/{len(chars)} ({percentage:.1f}%)"
        )

        lines.append(
            "This is a diagnostic, NOT verified accuracy."
        )

        lines.append(
            "Unmatched characters:"
        )

        lines.extend(
            mismatches[:30] if mismatches
            else ["None found by the dictionary check."]
        )

        self._show_check("\n".join(lines))
        self.status_var.set(
            "Dictionary check complete. Verify ambiguous characters "
            "against the source photograph."
        )

    def _confirm_leave(self):
        if self._state() == self.baseline:
            return True

        result = messagebox.askyesnocancel(
            "Unsaved changes",
            "Save changes to the current document?"
        )

        if result is None:
            return False

        if result:
            return self.save()

        return True

    def _document_changed(self, event=None):
        new_doc = self.doc_var.get()

        if new_doc == self.current:
            return

        if self._confirm_leave():
            self._open_document(new_doc)
        else:
            self.doc_var.set(self.current)

    def _navigate(self, direction):
        index = self.document_ids.index(self.current)
        new_index = index + direction

        if not 0 <= new_index < len(self.document_ids):
            return

        new_doc = self.document_ids[new_index]

        if self._confirm_leave():
            self._open_document(new_doc)

    def save(self):
        data = self._state()

        chars = han_characters(data["han"])
        words = syllables(data["phien_am"])

        if data["reviewed"] and len(chars) != len(words):
            messagebox.showwarning(
                "Incomplete phonetic alignment",
                "The Chinese-character and Phiên âm counts differ.\n\n"
                "You can save as an unreviewed draft, but cannot mark "
                "this document as reviewed yet."
            )
            return False

        raw_han = chinese_ocr_text(
            self.han_labels.get(self.current, [])
        )

        raw_reading, raw_translation = vietnamese_ocr_text(
            self.viet_labels.get(self.current, [])
        )

        record = {
            "doc_id": self.current,
            "status": (
                "manually_reviewed"
                if data["reviewed"]
                else "draft_unverified"
            ),
            "saved_at": datetime.now().astimezone().isoformat(),
            "han": data["han"],
            "phien_am": data["phien_am"],
            "dich_nghia": data["dich_nghia"],
            "notes": data["notes"],
            "reviewed": data["reviewed"],
            "character_count": len(chars),
            "phonetic_syllable_count": len(words),
            "count_match": len(chars) == len(words),
            "original_ocr": {
                "han": raw_han,
                "phien_am": raw_reading,
                "dich_nghia": raw_translation
            },
            "source_images": [
                path.name
                for path in self.images.get(self.current, [])
            ]
        }

        destination = REVIEW / f"{self.current}.json"

        if destination.exists():
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            shutil.copy2(
                destination,
                HISTORY / f"{self.current}_{timestamp}.json"
            )

        temporary = REVIEW / f"{self.current}.json.tmp"

        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        temporary.replace(destination)

        self.baseline = self._state()

        self.status_var.set(
            f"Saved {destination.name} | "
            f"status: {record['status']}"
        )

        return True

    def _save_shortcut(self, event=None):
        self.save()
        return "break"

    def _close(self):
        if self._confirm_leave():
            self.destroy()


if __name__ == "__main__":
    app = ReviewApp()
    app.mainloop()
