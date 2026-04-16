# Table Detector — Document Table Extraction

A Python back-end module that detects tables in invoice and bank document images
using the pre-trained DETR model
[`TahaDouaji/detr-doc-table-detection`](https://huggingface.co/TahaDouaji/detr-doc-table-detection).

---

## Requirements

- Python 3.10+
- pip

---

## Installation

```bash
# 1 — Clone the repository
git clone <https://github.com/nz-valere/ImageObject_detection_with_transformers.git>
cd <ImageObject_detection_with_transformers>

# 2 — Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 — Install dependencies
pip install -r requirements.txt
```

> **Note:** The HuggingFace model weights (~160 MB) are downloaded automatically
> on the first call to `TableDetector()` and cached locally.

---
## Usage

```python
from table_detector import TableDetector

# Initialise the detector (downloads model on first run)
detector = TableDetector(threshold=0.5)