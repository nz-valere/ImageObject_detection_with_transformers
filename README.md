# Table Detector — Document Table Extraction

A Python class that detects tables in invoice and bank document images using the
pre-trained [`TahaDouaji/detr-doc-table-detection`](https://huggingface.co/TahaDouaji/detr-doc-table-detection)
model (DETR — Detection Transformer).

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+

Install all dependencies:

---

## Installation

```bash
# 1 — Clone the repository
git clone <https://github.com/nz-valere/ImageObject_detection_with_transformers.git>
cd <ImageObject_detection_with_transformers>

# 2 — Create a virtual environment if not
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 — Install dependencies
pip install -r requirements.txt
```

> **Note:** The HuggingFace model weights (~160 MB) are downloaded automatically
> on the first call to `TableDetector()` and cached locally.

---

---

## Project Structure

```
pythonProject/
├── table_detector.py       # TableDetector class
├── test_table_detector.py  # pytest test suite
├── conftest.py             # shared fixtures
├── requirements.txt
└── README.md
```

---

## Usage

### Detect tables in a local image file

```python
from TableDetector import TableDetector

detector = TableDetector()

# Works with invoice images
results = detector.predict_from_path("invoice.png", threshold=0.9)

for detection in results:
    print(
        f"Detected {detection['label']} with confidence "
        f"{detection['score']} at location {detection['box']}"
    )
```

### Detect tables from a URL

```python
from TableDetector import TableDetector

detector = TableDetector()
results = detector.predict_from_url("https://example.com/bank_statement.png", threshold=0.9)
```

### Sample output

```
Detected table with confidence 0.997 at location [73.57, 177.5, 761.43, 611.0]
Detected table with confidence 0.943 at location [73.21, 623.14, 760.89, 892.45]
```

Each detection dict contains:

| Key     | Type    | Description                          |
|---------|---------|--------------------------------------|
| `label` | `str`   | `"table"` or `"table rotated"`       |
| `score` | `float` | Confidence score (0 – 1)             |
| `box`   | `list`  | `[x_min, y_min, x_max, y_max]` in px |

---

## API Reference

### `TableDetector()`

Loads the model and processor from HuggingFace on construction. Requires an
internet connection on first use (weights are cached locally afterwards).

### `predict_from_path(image_path, threshold=0.9) → list[dict]`

| Parameter    | Type    | Default | Description                          |
|--------------|---------|---------|--------------------------------------|
| `image_path` | `str`   | —       | Absolute or relative path to image   |
| `threshold`  | `float` | `0.9`   | Minimum confidence score to include  |

**Raises:**
- `FileNotFoundError` — path does not exist or is a directory
- `ValueError` — threshold outside `[0, 1]`
- `OSError` / `PIL.UnidentifiedImageError` — file is not a valid image

### `predict_from_url(url, threshold=0.9) → list[dict]`

| Parameter   | Type    | Default | Description                         |
|-------------|---------|---------|-------------------------------------|
| `url`       | `str`   | —       | Public URL of the image             |
| `threshold` | `float` | `0.9`   | Minimum confidence score to include |

**Raises:**
- `requests.HTTPError` — non-200 HTTP response
- `ValueError` — threshold outside `[0, 1]`

---

## Running the Tests

```bash
pytest test_table_detector.py -v
```

All tests use mocked model outputs — **no GPU or internet connection required**
to run the test suite.

### Test Coverage

| Scenario | Class |
|---|---|
| Single table detected in invoice | `TestInvoiceSuccessfulExtraction` |
| Multiple tables detected in invoice | `TestInvoiceSuccessfulExtraction` |
| Result dict has required keys | `TestInvoiceSuccessfulExtraction` |
| Box values rounded to 2 dp | `TestInvoiceSuccessfulExtraction` |
| Lower threshold yields more results | `TestInvoiceSuccessfulExtraction` |
| Single table detected in bank document | `TestBankDocumentSuccessfulExtraction` |
| Rotated table detected in bank document | `TestBankDocumentSuccessfulExtraction` |
| Multiple tables (mixed labels) in bank doc | `TestBankDocumentSuccessfulExtraction` |
| No tables detected (empty result) | `TestBankDocumentSuccessfulExtraction` |
| File not found raises `FileNotFoundError` | `TestFileErrorHandling` |
| Directory path raises `FileNotFoundError` | `TestFileErrorHandling` |
| Corrupted image raises exception | `TestFileErrorHandling` |
| Threshold > 1 raises `ValueError` | `TestFileErrorHandling` |
| Threshold < 0 raises `ValueError` | `TestFileErrorHandling` |
| Boundary thresholds 0 and 1 accepted | `TestFileErrorHandling` |
| HTTP error raises `requests.HTTPError` | `TestUrlErrorHandling` |
| URL success returns detections | `TestUrlErrorHandling` |
| Correct HuggingFace model loaded | `TestModelInitialisation` |
| Model set to eval mode on init | `TestModelInitialisation` |

---