"""BeamFinder Drone Detector Module.

Wraps YOLO26n model for drone detection in stationary camera images.
"""

from pathlib import Path

from ultralytics import YOLO


class DroneDetector:
    """Detects objects (drones) in images using YOLO26n.

    Attributes:
        model: Loaded YOLO26n model instance.
        confidence: Minimum confidence threshold for detections.
        image_size: Input image size for inference (pixels).
    """

    def __init__(self, model_name: str, confidence: float, image_size: int) -> None:
        """Initialize the detector with model and inference settings.

        Args:
            model_name: YOLO model file name (e.g., 'yolo26n.pt').
            confidence: Minimum confidence to keep a detection (0.0–1.0).
            image_size: Inference image size in pixels.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, image_path: Path) -> list[dict]:
        """Run detection on a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            List of detection dictionaries, each containing:
                - x (float): Top-left X coordinate of bounding box.
                - y (float): Top-left Y coordinate of bounding box.
                - width (float): Width of bounding box.
                - height (float): Height of bounding box.
                - confidence (float): Detection confidence score.
                - class (str): Detected object class name.
        """
        results = self.model(
            str(image_path),
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            # xywh returns [center_x, center_y, width, height]
            cx, cy, w, h = boxes.xywh[i].tolist()

            # Convert center-format to top-left format
            x = cx - w / 2
            y = cy - h / 2

            conf = boxes.conf[i].item()
            class_id = int(boxes.cls[i].item())
            class_name = result.names[class_id]

            detections.append({
                "x": round(x, 2),
                "y": round(y, 2),
                "width": round(w, 2),
                "height": round(h, 2),
                "confidence": round(conf, 4),
                "class": class_name,
            })

        return detections
