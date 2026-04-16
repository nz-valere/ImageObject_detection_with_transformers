"""
pytest test suite for TableDetector.

Covers:
  - Successful table extraction from invoice images
  - Successful table extraction from bank document images
  - Error handling: file not found, corrupted image, invalid threshold
  - Edge cases: no detections, multiple detections, rotated-table label
  - URL-based prediction (success and HTTP error)
"""

import io
import pytest
import torch
import requests
from PIL import Image
from unittest.mock import MagicMock, patch, PropertyMock

from TableDetector import TableDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_post_process_result(scores, labels, boxes):
    """Return a list matching the shape of post_process_object_detection output."""
    return [
        {
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
    ]


def _wire_processor(mock_processor, scores, labels, boxes):
    """Configure mock_processor to return the given detections."""
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    mock_processor.post_process_object_detection.return_value = (
        _make_post_process_result(scores, labels, boxes)
    )


# ---------------------------------------------------------------------------
# Successful extraction — Invoice
# ---------------------------------------------------------------------------

class TestInvoiceSuccessfulExtraction:
    """Table detection succeeds on invoice document images."""

    def test_single_table_detected_invoice(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.95], labels=[0], boxes=[[10.0, 20.0, 300.0, 400.0]])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(invoice_image, threshold=0.9)

        assert len(results) == 1
        assert results[0]["label"] == "table"
        assert results[0]["score"] == pytest.approx(0.95, abs=1e-3)
        assert len(results[0]["box"]) == 4

    def test_multiple_tables_detected_invoice(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(
            mock_processor,
            scores=[0.97, 0.93],
            labels=[0, 0],
            boxes=[[10.0, 20.0, 200.0, 300.0], [210.0, 20.0, 400.0, 300.0]],
        )
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(invoice_image, threshold=0.9)

        assert len(results) == 2
        assert all(r["label"] == "table" for r in results)
        assert results[0]["score"] >= results[1]["score"]  # order preserved

    def test_detection_result_has_required_keys_invoice(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.91], labels=[0], boxes=[[5.0, 5.0, 100.0, 100.0]])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(invoice_image)

        assert "label" in results[0]
        assert "score" in results[0]
        assert "box" in results[0]

    def test_box_values_are_rounded_invoice(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(
            mock_processor,
            scores=[0.92],
            labels=[0],
            boxes=[[10.123456, 20.987654, 300.111111, 400.999999]],
        )
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(invoice_image)

        for val in results[0]["box"]:
            # at most 2 decimal places
            assert val == round(val, 2)

    def test_lower_threshold_yields_more_results_invoice(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector

        # Simulate processor returning different sets per threshold call
        def side_effect(outputs, target_sizes, threshold):
            if threshold <= 0.5:
                return _make_post_process_result(
                    [0.91, 0.75, 0.60], [0, 0, 0],
                    [[0]*4, [0]*4, [0]*4]
                )
            return _make_post_process_result([0.91], [0], [[0]*4])

        mock_processor.post_process_object_detection.side_effect = side_effect
        mock_model.return_value = MagicMock()

        high = detector.predict_from_path(invoice_image, threshold=0.9)
        low = detector.predict_from_path(invoice_image, threshold=0.5)

        assert len(low) >= len(high)


# ---------------------------------------------------------------------------
# Successful extraction — Bank Document
# ---------------------------------------------------------------------------

class TestBankDocumentSuccessfulExtraction:
    """Table detection succeeds on bank document images."""

    def test_single_table_detected_bank(self, mock_detector, bank_document_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.98], labels=[0], boxes=[[50.0, 60.0, 500.0, 350.0]])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(bank_document_image, threshold=0.9)

        assert len(results) == 1
        assert results[0]["label"] == "table"
        assert results[0]["score"] == pytest.approx(0.98, abs=1e-3)

    def test_rotated_table_detected_bank(self, mock_detector, bank_document_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.94], labels=[1], boxes=[[10.0, 10.0, 200.0, 200.0]])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(bank_document_image, threshold=0.9)

        assert results[0]["label"] == "table rotated"

    def test_multiple_tables_bank(self, mock_detector, bank_document_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(
            mock_processor,
            scores=[0.99, 0.96, 0.91],
            labels=[0, 0, 1],
            boxes=[[0]*4, [0]*4, [0]*4],
        )
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(bank_document_image, threshold=0.9)

        assert len(results) == 3
        labels = {r["label"] for r in results}
        assert "table" in labels
        assert "table rotated" in labels

    def test_no_tables_detected_bank(self, mock_detector, bank_document_image):
        """Empty results list when no table passes the threshold."""
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[], labels=[], boxes=[])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(bank_document_image, threshold=0.9)

        assert results == []

    def test_score_is_float_bank(self, mock_detector, bank_document_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.955], labels=[0], boxes=[[0]*4])
        mock_model.return_value = MagicMock()

        results = detector.predict_from_path(bank_document_image)

        assert isinstance(results[0]["score"], float)


# ---------------------------------------------------------------------------
# Error handling — file/path errors
# ---------------------------------------------------------------------------

class TestFileErrorHandling:
    """predict_from_path raises proper errors for bad inputs."""

    def test_file_not_found_raises(self, mock_detector):
        detector, _, _ = mock_detector
        with pytest.raises(FileNotFoundError, match="not found"):
            detector.predict_from_path("/nonexistent/path/image.png")

    def test_directory_path_raises(self, mock_detector, tmp_path):
        detector, _, _ = mock_detector
        with pytest.raises(FileNotFoundError, match="not a file"):
            detector.predict_from_path(str(tmp_path))

    def test_corrupted_image_raises(self, mock_detector, corrupted_image):
        """OSError (or PIL variant) is raised for unreadable image data."""
        detector, _, _ = mock_detector
        with pytest.raises(Exception):
            # PIL will raise when trying to open the corrupt bytes
            detector.predict_from_path(corrupted_image)

    def test_invalid_threshold_too_high(self, mock_detector, invoice_image):
        detector, _, _ = mock_detector
        with pytest.raises(ValueError, match="Threshold"):
            detector.predict_from_path(invoice_image, threshold=1.5)

    def test_invalid_threshold_negative(self, mock_detector, invoice_image):
        detector, _, _ = mock_detector
        with pytest.raises(ValueError, match="Threshold"):
            detector.predict_from_path(invoice_image, threshold=-0.1)

    def test_threshold_boundary_zero_accepted(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[], labels=[], boxes=[])
        mock_model.return_value = MagicMock()
        # Should not raise
        results = detector.predict_from_path(invoice_image, threshold=0.0)
        assert isinstance(results, list)

    def test_threshold_boundary_one_accepted(self, mock_detector, invoice_image):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[], labels=[], boxes=[])
        mock_model.return_value = MagicMock()
        results = detector.predict_from_path(invoice_image, threshold=1.0)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Error handling — URL-based prediction
# ---------------------------------------------------------------------------

class TestUrlErrorHandling:
    """predict_from_url raises proper errors for bad URLs/HTTP responses."""

    def test_url_http_error_raises(self, mock_detector):
        detector, _, _ = mock_detector
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("TableDetector.requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                detector.predict_from_url("http://example.com/missing.png")

    def test_url_success_returns_detections(self, mock_detector):
        detector, mock_model, mock_processor = mock_detector
        _wire_processor(mock_processor, scores=[0.95], labels=[0], boxes=[[0]*4])
        mock_model.return_value = MagicMock()

        fake_image = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        fake_image.save(buf, format="PNG")
        buf.seek(0)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.raw = buf

        with patch("TableDetector.requests.get", return_value=mock_response):
            with patch("TableDetector.Image.open", return_value=fake_image):
                results = detector.predict_from_url("http://example.com/doc.png")

        assert len(results) == 1
        assert results[0]["label"] == "table"

    def test_url_invalid_threshold_raises(self, mock_detector):
        detector, _, _ = mock_detector
        with pytest.raises(ValueError, match="Threshold"):
            detector.predict_from_url("http://example.com/doc.png", threshold=2.0)


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

class TestModelInitialisation:
    """TableDetector loads the correct HuggingFace model on construction."""

    def test_loads_correct_model(self):
        with (
            patch("TableDetector.DetrImageProcessor.from_pretrained") as mock_proc,
            patch("TableDetector.DetrForObjectDetection.from_pretrained") as mock_mdl,
        ):
            mock_mdl.return_value = MagicMock(eval=MagicMock())
            TableDetector()
            mock_proc.assert_called_once_with("TahaDouaji/detr-doc-table-detection")
            mock_mdl.assert_called_once_with("TahaDouaji/detr-doc-table-detection")

    def test_model_set_to_eval_mode(self):
        with (
            patch("TableDetector.DetrImageProcessor.from_pretrained"),
            patch("TableDetector.DetrForObjectDetection.from_pretrained") as mock_mdl,
        ):
            mock_instance = MagicMock()
            mock_mdl.return_value = mock_instance
            TableDetector()
            mock_instance.eval.assert_called_once()
