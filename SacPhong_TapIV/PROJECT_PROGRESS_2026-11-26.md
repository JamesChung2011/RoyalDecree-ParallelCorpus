# Sắc phong – Tập IV: Project Progress Report
**Date:** 26 November 2026  
**Status:** Research/development prototype; final evaluation and packaging are pending.  
**Prepared for:** Team code review and GitHub handoff.

## Objective
Produce a reproducible pipeline that extracts royal-decree photographs and their corresponding Vietnamese **Phiên âm** and **Dịch nghĩa** from the 175-page *Sắc phong – Tập IV* PDF, recognizes text, proposes Hán–Phiên âm–Dịch nghĩa alignments, audits quality, and exports the assignment's seven-field corpus (`sent_id`, `doc_id`, `doc_type`, `han`, `phien_am`, `dich_nghia`, `ghi_chu`). The final one-command workflow is **not yet implemented**.

## Latest verified execution milestones
| Stage | Implementation and result | Limitation |
|---|---|---|
| 1B/1C: layout and crops | 175 PDF pages scanned; 10 non-content pages excluded; 165 body pages; **291 provisional image/text crop pairs** exported; four special layout pages handled | All pairs are provisional; not every crop has been manually verified |
| 2I: OCR | **291/291 Vietnamese** and **291/291 Chinese OCR draft records** generated; Vietnamese uses Tesseract and Chinese uses GLM-OCR | Successful execution is not proof of correct recognition |
| 2J/2K: quality and recovery | Initially 22 *Phiên âm* readings missing; **20 recovered as unverified candidates** using crop expansion/heading parsing; two unresolved (`AUTO.082.01`, `AUTO.090.01`) | Recovered readings can contain OCR errors and neighboring text |
| 2L: source-pair exceptions | Four records on PDF pages 82 and 90 blocked from automatic source pairing | Two decree photos appear with one printed Vietnamese block per page; relationship still requires verification |
| 3A: conservative merge | **291** merged document records: **276 complete draft**, **11 incomplete**, **4 blocked** | No draft is verified |
| 3B: quality screen | **288/291** placed on some level of review queue; median Hán–Việt dictionary compatibility previously reported **44.2%** | Dictionary compatibility is not a transcription-accuracy metric; some character alignments may be shifted |
| 4A: document-level alignment | **276 provisional triplets**, **15 excluded**, **27,177 character-level diagnostics** | No verified alignment |
| 4B: segmentation proposals | **5,745 fragments**: 2,191 Hán lines, 1,706 Phiên âm segments, 1,848 Dịch nghĩa segments | Fragments are not sentence triplets |
| **4C: latest** | **276 documents** examined; **506 unverified order-based sentence candidates** from **96 documents**; **180 documents withheld**; **3,968 unpaired fragment rows** | **Zero verified sentence pairs; no semantic translation alignment performed** |

### Step 4C detail (from the uploaded `alignment_report.json`)
- 56 documents: `ORDER_ONLY_SENTENCE_PROPOSALS_UNVERIFIED`.
- 40 documents: `PARTIAL_ORDER_PROPOSALS_UNVERIFIED`.
- 180 documents: `WITHHELD_FROM_SENTENCE_PROPOSALS`.
- Candidate file: `sentence_candidates_UNVERIFIED.csv`, containing the seven required field names, but **not suitable as gold-standard training data**.
- Additional files: `candidate_diagnostics.csv`, `document_pairing_status.csv`, `unpaired_segments.csv`, `alignment_report.json`.

## Main technical risks
1. **Historical Hán OCR is the dominant bottleneck.** Some outputs are unreadable, missing characters, or semantically wrong even when character/syllable totals match.
2. **Document-pairing exceptions:** Some pages do not contain a one-to-one image/Vietnamese-block layout; do not duplicate a translation across decree images without evidence.
3. **Sentence segmentation and pairing remain provisional:** Step 4C aligns Vietnamese segments mainly by order, not independently verified semantics.
4. **Teacher's ≥95% matching/accuracy requirement has not been measured or achieved.** A verified reference set and appropriate metrics are still required.
5. **Large local assets:** The PDF, dictionaries, hundreds of crop images, Ollama weights, and OCR outputs should be stored/shared separately from source-code review unless permitted by the teacher/team.

## Next planned tasks
1. Create a small, manually checked reference set covering both intact and damaged decrees.
2. Measure Chinese character error rate, Phiên âm transcription quality, source-pair correctness, and semantic three-way alignment; improve weakest cases based on evidence.
3. Improve uncertain/missing Chinese OCR and document-pairing exceptions; preserve abstentions where characters are unreadable.
4. Refine sentence alignment beyond equal-count/order proposals; keep provenance and uncertainty fields.
5. Implement final evaluation, CSV/XLSX export and a **single reproducible `pipeline.py` entrypoint**.
6. Produce a clean submission package and safely exclude obsolete experiments, `.venv`, caches, images, and local model files.

## Recommended GitHub review scope
Commit Python source files (`pipeline.py`, `steps/`, `utils/`), `config.yaml` after removing any private local settings, `requirements.txt`, project documentation, `step1c_page_overrides.json` if present, this report, and a `.gitignore`.

**Do not commit:** `.venv/`, `output/` (including the 291 crop images), local Ollama model weights, private datasets/dictionaries, temporary ZIP archives, credentials, or `.env` files. Share input data separately through an authorized channel if team members need to rerun the pipeline.

## Team handoff note
The Step 1–4C prototype is executable, **but it is not yet a validated corpus**. Its current most tangible artifact is the 506-row **unverified** sentence-candidate file. The next engineering milestone is quality improvement and evaluation, followed by final consolidation—not adding more models merely to increase character counts.
