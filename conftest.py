"""
Shared pytest fixtures for table detector tests.
"""
import io
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
import torch


# ---------------------------------------------------------------------------
# Helpers to build fake model outputs
# ---------------------------------------------------------------------------

def _make_detection_output(scores, labels, boxes):
    """Build a mock post_process result dict."""
    return {
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Shared image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def invoice_image(tmp_path):
    """Create a minimal valid invoice-like image on disk."""
    img = Image.new("RGB", (800, 1100), color=(255, 255, 255))
    path = tmp_path / "invoice.png"
    img.save(path)
    return str(path)


@pytest.fixture
def bank_document_image(tmp_path):
    """Create a minimal valid bank-document-like image on disk."""
    img = Image.new("RGB", (800, 600), color=(240, 240, 240))
    path = tmp_path / "bank_document.png"
    img.save(path)
    return str(path)


@pytest.fixture
def corrupted_image(tmp_path):
    """Create a file that is NOT a valid image."""
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"this is not image data \x00\x01\x02")
    return str(path)


# ---------------------------------------------------------------------------
# Mock model fixture — patches both from_pretrained calls at once
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_detector():
    """
    Return a TableDetector whose HuggingFace model and processor are fully
    mocked so no network calls or GPU are required.
    """
    from TableDetector import TableDetector

    id2label = {0: "table", 1: "table rotated"}

    mock_model_cfg = MagicMock()
    mock_model_cfg.id2label = id2label

    mock_model = MagicMock()
    mock_model.config = mock_model_cfg

    mock_processor = MagicMock()

    with (
        patch.object(TableDetector, "__init__", lambda self: None),
    ):
        detector = TableDetector()
        detector.processor = mock_processor
        detector.model = mock_model
        yield detector, mock_model, mock_processor
