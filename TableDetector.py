from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from pathlib import Path


class TableDetector:
    MODEL_NAME = "TahaDouaji/detr-doc-table-detection"

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = DetrForObjectDetection.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def predict_from_path(self, image_path: str, threshold: float = 0.9) -> list[dict]:
        """
        Run table detection on an image file.

        Args:
            image_path: Path to the image file.
            threshold: Confidence score threshold (0.0 - 1.0).

        Returns:
            List of detections, each with keys: label, score, box.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If threshold is not between 0 and 1.
            OSError: If the file cannot be opened as an image.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {image_path}")

        image = Image.open(image_path)
        return self._run_detection(image, threshold)

    def predict_from_url(self, url: str, threshold: float = 0.9) -> list[dict]:
        """
        Run table detection on an image fetched from a URL.

        Args:
            url: URL of the image.
            threshold: Confidence score threshold (0.0 - 1.0).

        Returns:
            List of detections, each with keys: label, score, box.

        Raises:
            ValueError: If threshold is not between 0 and 1.
            requests.HTTPError: If the URL returns a non-200 status.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        image = Image.open(response.raw)
        return self._run_detection(image, threshold)

    def _run_detection(self, image: Image.Image, threshold: float) -> list[dict]:
        """
        Core detection logic matching the reference snippet exactly.

        Args:
            image: PIL Image object.
            threshold: Confidence score threshold.

        Returns:
            List of detection dicts with label, score, and box.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > threshold
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            detections.append(
                {
                    "label": self.model.config.id2label[label.item()],
                    "score": round(score.item(), 3),
                    "box": box,
                }
            )

        return detections