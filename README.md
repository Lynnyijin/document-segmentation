# Data segmentation — PDF pages and document-start classification

This repository supports a small pipeline for **dossier documents**: turn PDFs into page images, align them with Excel annotations, measure **inter-annotator agreement** on “document start” labels, and train **baseline classifiers** that predict whether a page is the start of a new document.

---

## Typical workflow

1. **Export PDFs to PNGs** with `pdf_to_png.py` (one folder per PDF stem under an output root).
2. **Annotate** pages in Excel workbooks (`image path`, `page number`, `Start page` = yes/no). Multiple annotators can use separate `.xlsx` files.
3. **Check agreement** with `compare_annotation_labels.py` (disagreements + Fleiss’ κ where every rater has the row).
4. **Train baselines** with `train_page_classifier.py` (merges labels across files, splits by dossier, trains KNN/XGBoost on VGG features plus optional CNN/LSTM models).
5. **Score a single page** with `predict_page_classifier.py` using the saved KNN or XGBoost models.

---

## Setup

Python 3.10+ recommended. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** The first PyTorch/torchvision run may download ImageNet weights (EfficientNet-B0, VGG-16).

---

## Expected data layout

### Page images

Training expects PNGs under:

`pdf_pages_png/<dossier_stem>/<dossier_stem>_page_0001.png`

That matches the default naming from `pdf_to_png.py` (`<stem>_page_####.png` inside `<stem>/`).

### Annotation Excel files

Sheets must include columns (names case-insensitive except where noted):

- **image path** — path or filename whose **stem** identifies the dossier (same as the folder name under `pdf_pages_png`).
- **page number** — integer page index (must match the `####` in `..._page_####.png`).
- **Start page** — treat as binary: **`yes`** → start of document (label 1); anything else or empty → not start (label 0).

The reader prefers a sheet named `Annotation`; otherwise it picks the first sheet whose header row contains those three concepts (`annotation_io.read_annotation_sheet`).

---

## Files in this repo

| File | Role |
|------|------|
| `requirements.txt` | Dependencies: PyMuPDF, pandas/openpyxl, scikit-learn, XGBoost, PyTorch/torchvision, statsmodels, etc. |
| `annotation_io.py` | Shared Excel I/O: find the right sheet/column, normalize **Start page** to 0/1, dossier stem order, sanity check that the first *N* dossiers match across all annotation files (used before training). |
| `pdf_to_png.py` | Walks an input folder for `.pdf` files, renders each page to PNG via PyMuPDF (`fitz`), optional `--recursive` and `--dpi`. |
| `page_classifier_features.py` | **VGG16FeatureExtractor**: loads VGG-16 with ImageNet weights, drops the final classifier layer so each image becomes a **4096-D** vector (penultimate FC). Used for KNN/XGBoost and as input to the LSTM. |
| `compare_annotation_labels.py` | For the first *K* dossiers (order from the first workbook), lists **binary** disagreements on Start page across files. Computes **Fleiss’ κ** only on `(stem, page)` rows present in **every** listed workbook. |
| `train_page_classifier.py` | End-to-end training: merge multi-file annotations by **majority vote** (ties broken by file order), split **dossiers** into train/val/test, extract VGG features, fit **KNN** (scaled) and **XGBoost**, fine-tune **EfficientNet-B0** and **VGG16** image classifiers, train **LSTM** on per-dossier VGG feature sequences. Saves `knn.joblib`, `xgboost.json`, `baseline_config.json`, and `.pt` checkpoints under `outputs/page-classifier-baselines/`. |
| `predict_page_classifier.py` | Loads `baseline_config.json` and either **KNN** or **XGBoost**; encodes one image with the same VGG pipeline (or legacy resized pixels if an old config had `image_size`). Prints class probabilities and the predicted label. |

---

## How the code fits together

### `annotation_io.py`

- **`read_annotation_sheet`** — opens `.xlsx`, selects the annotation sheet by name or by required columns.
- **`find_start_column`** — resolves the Start column even if spacing/casing differs slightly.
- **`binary_start_label`** — maps cell values to 0/1 consistently with training (NaN → 0).
- **`stem_order_first_appearance`** — stable dossier order from the first occurrence of each `image path` stem.
- **`assert_first_n_dossiers_match_across_files`** — ensures all workbooks share the same leading dossier list so merges and comparison scripts stay aligned.

### `pdf_to_png.py`

- **`pdf_to_png_pages`** — opens one PDF, applies a DPI-based zoom matrix, writes `stem_page_0001.png`, … under `output_root/stem/`.
- **`iter_pdfs`** — lists PDFs in a folder, optionally recursively.

### `page_classifier_features.py`

- **`VGG16FeatureExtractor.encode_paths`** — batches images through ImageNet preprocessing and the truncated VGG; returns a float32 matrix of shape `(n_images, 4096)`.

### `compare_annotation_labels.py`

- Builds a map `(stem, page) → { workbook_name: 0/1 }` restricted to the first `--first-dossiers` stems.
- **Conflicts** — any key where annotators disagree on the binary label (among files that contain that row).
- **Fleiss’ κ** — built from a count table `[n_zeros, n_ones]` per subject, only for subjects rated by **all** files; uses `statsmodels.stats.inter_rater.fleiss_kappa`.

### `train_page_classifier.py`

- **`load_merged_annotation_rows`** — for each `(stem, page)`, collects votes from every workbook; **majority** wins; on a tie, the **first file in `tie_break_file_order`** that voted for one of the tied labels wins.
- **Splits** — unique dossier stems are shuffled with a fixed seed; default 39 train / 13 val / 13 test dossiers (65 total); remaining dossiers are unused if any.
- **Sklearn baselines** — `StandardScaler` + **KNN** (distance-weighted); **XGBoost** with `scale_pos_weight` for imbalance.
- **CNNs** — short Adam fine-tuning with class-weighted cross-entropy on full images.
- **LSTM** — groups pages by parent folder (`dossier_stem`), sorts by page index parsed from the filename, packs variable-length sequences, cross-entropy with `ignore_index=-100` for padding.
- **`save_outputs`** — persists sklearn/XGB models, JSON config (feature type, label names, metrics), and Torch state dicts.

### `predict_page_classifier.py`

- Reads `feature_extractor` from JSON; if `vgg16`, runs **`VGG16FeatureExtractor`** on the single path; loads **KNN** (`joblib`) or **XGBoost** (`load_model`), applies `--threshold` on **P(class 1)**.

---

## Commands (quick reference)

```bash
# PDF → PNG (edit defaults or pass flags)
python pdf_to_png.py --input /path/to/pdfs --output /path/to/pdf_pages_png

# Inter-annotator comparison
python compare_annotation_labels.py --first-dossiers 5 --annotations "annotation 1.xlsx" "annotation 2.xlsx"

# Train (expects default paths in TrainConfig: annotation 1–4.xlsx, pdf_pages_png/)
python train_page_classifier.py

# Predict one page (VGG + saved KNN/XGB)
python predict_page_classifier.py \
  --model-dir outputs/page-classifier-baselines \
  --backend knn \
  --image pdf_pages_png/<stem>/<stem>_page_0001.png
```

---

## Outputs

After training, `outputs/page-classifier-baselines/` typically contains:

- `knn.joblib`, `xgboost.json` — sklearn/XGB baselines on VGG features.
- `baseline_config.json` — metadata, metrics, label map.
- `efficientnet_b0.pt`, `vgg16_classifier.pt`, `vgg_feature_lstm.pt` — PyTorch checkpoints (use your own loading code if you deploy those models; `predict_page_classifier.py` only exposes KNN/XGB today).

---

## License / data

This repository is a local tooling project. Add your own license and keep sensitive PDFs or annotations out of version control if applicable.
