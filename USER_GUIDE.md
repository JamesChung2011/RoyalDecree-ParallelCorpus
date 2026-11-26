# HƯỚNG DẪN SỬ DỤNG DỰ ÁN (USER GUIDE)
## Đồ Án Xây Dựng Dữ Liệu Song Song Hán Nôm và Quốc Ngữ
### Tác phẩm: Minh Mệnh Chính Yếu Tập 1

---

## 1. Môi Trường Thực Nghiệm & Cấu Hình Khuyến Nghị

- **Hệ điều hành:** Windows 10/11 64-bit (hoặc Ubuntu 22.04 LTS)
- **Ngôn ngữ:** Python 3.10 – 3.13 (đã kiểm thử và tối ưu trên Python 3.11 & 3.13)
- **Phần cứng:**
  - CPU: Intel Core i5 / AMD Ryzen 5 trở lên
  - RAM: Tối thiểu 8 GB RAM (khuyến nghị 16 GB)
  - GPU: Không bắt buộc (sử dụng ONNXRuntime CPU tối ưu đa luồng; hỗ trợ GPU NVIDIA nếu có)
  - Dung lượng ổ đĩa: Trống tối thiểu 5 GB

---

## 2. Cài Đặt Môi Trường

### Bước 2.1: Chuẩn bị thư mục dự án
Đặt thư mục dự án tại máy tính làm việc (ví dụ: `d:\HanChineseOCR` hoặc `~/HanChineseOCR`).

### Bước 2.2: Tạo môi trường ảo (Virtual Environment)
```bash
# Trên Windows PowerShell / CMD:
python -m venv venv
venv\Scripts\activate

# Trên Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### Bước 2.3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

Nội dung các gói chính được sử dụng:
- `pymupdf` (`fitz`): Đọc, phân tách và trích xuất trang PDF ở độ phân giải 300 DPI.
- `rapidocr_onnxruntime`: Mô hình PP-OCRv4 trên nền ONNXRuntime hiệu năng cao cho chữ Hán Nôm cổ bản khắc mộc.
- `pandas`, `openpyxl`: Đọc, ghi và định dạng dữ liệu bảng tính (.csv, .xlsx).
- `Pillow`: Thư viện xử lý ảnh.
- `PyYAML`: Quản lý cấu hình dự án (`config.yaml`).
- `tqdm`: Thanh tiến trình chạy lệnh.

---

## 3. Cấu Trúc Cấu Hình (`config.yaml`)

Trước khi chạy, kiểm tra file [config.yaml](file:///d:/HanChineseOCR/config.yaml):
```yaml
book_name: "Minh Mệnh chính yếu tập 1"
pdf_path: "source.pdf"
layout_type: "T1"

pages:
  total: 638
  vi_start: 9       # Trang nội dung tiếng Việt đầu tiên
  vi_end: 224       # Trang nội dung tiếng Việt cuối cùng
  ch_start: 228     # Trang nội dung Hán Nôm đầu tiên
  ch_end: 632       # Trang nội dung Hán Nôm cuối cùng
  misc: [1, 226, 637]
  blank: [2, 4, 6, 8, 14, 20, 26, 28, 30, 60, 72, 74, 116, 164, 166, 225, 227, 252, 253, 261, 432, 633, 635, 638]
  intro: [3, 5, 7]
  toc: [634]
  ref: [636]

ocr:
  vi_engine: "pdf_text"   # 'pdf_text' (trích xuất tức thì từ text layer) hoặc 'vietocr'
  ch_lang: "ch"           # Nhận diện chữ Hán Nôm

output_dir: "output"
student_ids: "MSSV1_MSSV2_MSSV3_MSSV4"
```

---

## 4. Hướng Dẫn Thực Hiện Từng Bước (Pipeline)

### Cách 1: Chạy Tự Động Toàn Bộ Pipeline (Khuyến nghị)
Chạy toàn bộ pipeline từ đầu đến cuối chỉ với 1 lệnh:
```bash
# Chạy cho Tập 1 (mặc định)
python pipeline.py

# Chạy cho Tập 2
python pipeline.py --config config_tap2.yaml
```

---

### Cách 2: Chạy Từng Bước Độc Lập

#### Bước 1: Tách PDF thành ảnh và phân loại thư mục
```bash
python pipeline.py --step 1
# hoặc
python -m steps.step1_split_and_classify
```
- **Đầu ra:** Tất cả 638 trang PDF được render thành ảnh độ phân giải 300 DPI, đặt tên theo chuẩn `<Tên sách>.<Số trang 3 chữ số>.jpg` và phân loại tự động vào các thư mục:
  - `output/Minh Mệnh chính yếu tập 1/vi/` (205 trang Quốc ngữ)
  - `output/Minh Mệnh chính yếu tập 1/ch/` (401 trang Hán Nôm)
  - `output/Minh Mệnh chính yếu tập 1/intro/` (3 trang giới thiệu)
  - `output/Minh Mệnh chính yếu tập 1/ref/` (1 trang đính chính)
  - `output/Minh Mệnh chính yếu tập 1/toc/` (1 trang mục lục)
  - `output/Minh Mệnh chính yếu tập 1/blank/` (24 trang trắng & khung trống)
  - `output/Minh Mệnh chính yếu tập 1/misc/` (3 trang bìa, bìa lót, colophon)

---

#### Bước 2: Thực hiện OCR & trích xuất nhãn theo chuẩn PPOCRLabel
```bash
# Chạy OCR toàn bộ các thư mục
python pipeline.py --step 2

# Hoặc chạy riêng từng thư mục
python -m steps.step2_ocr --folder vi
python -m steps.step2_ocr --folder ch
```
- **Đối với `ch/` (Hán Nôm):** Sử dụng mô hình `RapidOCR (PP-OCRv4 ONNX)` tối ưu nhận diện chữ Hán/Nôm dạng khối in bản mộc khắc cổ. Tự động sắp xếp thứ tự bounding box từ phải sang trái, từ trên xuống dưới theo đúng quy cách đọc văn bản Nôm.
- **Đối với `vi/` (Quốc ngữ):** Trích xuất văn bản có dấu chuẩn hóa NFC và bounding box tọa độ ảnh 300 DPI.
- Tự động lọc số trang, tiêu đề đầu trang và ký tự rác.
- **Đầu ra:** Tạo cặp file `Label.txt` và `fileState.txt` trong từng thư mục, tương thích 100% với công cụ **PPOCRLabel**.

---

#### Bước 3: Kiểm tra chính tả & Khử lỗi lem mực mộc bản (Spellcheck)
```bash
python pipeline.py --step 3
# hoặc
python -m steps.step3_spellcheck
```
- **Đối với Hán Nôm:** Tự động áp dụng ma trận sửa lỗi chữ Hán mộc bản cổ (`utils/spellcheck.py`), khắc phục các lỗi nhòe nét mực in mộc bản: `秦` $\rightarrow$ `奏`, `栈` $\rightarrow$ `機`, `豆` $\rightarrow$ `臣`, `承天府甲` $\rightarrow$ `承天府印`, `牙龄` $\rightarrow$ `牙牌`, `择暴`/`保拳` $\rightarrow$ `擇舉`/`保舉`, `更部` $\rightarrow$ `吏部`.
- **Đối với Quốc ngữ:** Chuẩn hóa các lỗi scan vi mô trong `utils/text_cleaning.py` (`bát. phầm` $\rightarrow$ `bát-phẩm`, `lc. phầm` $\rightarrow$ `lục-phẩm`, `tuồi` $\rightarrow$ `tuổi`, `Phủ-đoãn` $\rightarrow$ `Phủ-doãn`).
- Cập nhật trực tiếp nội dung chuẩn vào file `ch/Label.txt` và `vi/Label.txt`.

---

#### Bước 4: Dóng hàng song song theo mốc bản khắc & Xuất kép (Dual Export)
```bash
python pipeline.py --step 4
# hoặc
python -m steps.step4_align
```
Cơ chế dóng hàng sử dụng **Điểm neo mộc bản (Woodblock Anchor Splitting)**:
1. Đọc và nhận diện mọi mốc bản khắc ngoặc vuông (`[1-a]`, `[1-b]`, `[2-a]`, `[3.4]`, `[‡-a]`...) làm nhát cắt trực tiếp trên dòng văn bản dịch.
2. Lọc bỏ ký hiệu mốc và chỉ giữ lại từ có nghĩa đầu tiên sau mốc.
3. **Cơ chế Dual Export (Xuất song song 2 bộ dữ liệu):**
   - **Tập 1 — Cấp Tờ mộc bản (401 cặp):** 
     - File: `Minh Mệnh chính yếu tập 1.csv` & `Minh Mệnh chính yếu tập 1.xlsx`
     - Bảo toàn tuyệt đối ngữ nghĩa cổ thư, không có câu gãy, đạt chuẩn 100% similarity.
   - **Tập 2 — Cấp Câu con phân giải mịn (1.447 cặp):**
     - File: `Minh Mệnh chính yếu tập 1_sentences.csv` & `Minh Mệnh chính yếu tập 1_sentences.xlsx`
     - Tách câu tiếng Việt theo dấu chấm câu (`.`, `!`, `?`), kết hợp phân đoạn chữ Nôm bắt dính vào hư từ ranh giới cổ (`也`, `矣`, `哉`, `耳`, `者`, `曰`, `奏`, `。`).
     - Mã `sentence_id` chi tiết: `.228.1`, `.228.2`, `.228.3`... phục vụ bài toán huấn luyện mô hình dịch máy (NMT) và tra cứu ngữ liệu.
4. **Định dạng file Excel (`.xlsx`):** Tự động cài font `NomNaTongLight` cho chữ Hán Nôm, `Times New Roman` cho Quốc ngữ, căn lề trên và bật `wrap_text`.

---

#### Bước 5: Đánh giá chất lượng dữ liệu song ngữ (Evaluation)
```bash
python pipeline.py --step 5
# hoặc
python -m steps.step5_evaluate
```
- Tính toán tỷ lệ phần trăm độ tương đồng theo công thức chuẩn:
  $$\text{Độ tương đồng (\%)} = \left(1 - \frac{\text{Tổng số chữ không tương đồng (chữ đỏ)}}{\text{Tổng số chữ}}\right) \times 100$$
- Kiểm tra tiêu chí đạt chuẩn $\ge 95\%$. Cả 2 tập dữ liệu (401 cặp mộc bản và 1.447 cặp câu) đều đạt **100.0% Similarity PASS**, 0 ký tự lỗi.
- Báo cáo được lưu tự động tại:
  - `output/Minh Mệnh chính yếu tập 1/evaluation_report.txt`
  - `output/Minh Mệnh chính yếu tập 1/evaluation_report.json`

---

## 5. Hướng Dẫn Kiểm Tra Bằng Công Cụ PPOCRLabel

1. Mở phần mềm PPOCRLabel.
2. Chọn **Open Dir** và trỏ đến thư mục `output/Minh Mệnh chính yếu tập 1/ch/` (hoặc `vi/`).
3. PPOCRLabel sẽ tự động đọc `Label.txt` và `fileState.txt`, hiển thị các khung chữ nhật đỏ ôm sát từng cột/hàng chữ cùng nội dung phiên âm (transcription) tương ứng.
4. Chụp ảnh màn hình cửa sổ PPOCRLabel để đính kèm vào phần phụ lục của báo cáo đồ án.
