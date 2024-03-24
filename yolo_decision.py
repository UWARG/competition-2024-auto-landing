"""
Landing Pad Detection Class using 2023 YOLO model
"""

import ultralytics
import numpy as np

class Detection:
    """
    A detected object in image space.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, bounds: np.ndarray, label: int, confidence: float
    ) -> "tuple[bool, Detection | None]":
        """
        bounds are of form x_1, y_1, x_2, y_2 .
        """
        # Check every element in bounds is >= 0.0
        if bounds.shape != (4,) or not np.greater_equal(bounds, 0.0).all():
            return False, None

        # n_1 <= n_2
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            return False, None

        if label < 0:
            return False, None

        if confidence < 0.0 or confidence > 1.0:
            return False, None

        return True, Detection(cls.__create_key, bounds, label, confidence)

    def __init__(
        self, class_private_create_key: object, bounds: np.ndarray, label: int, confidence: float
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Detection.__create_key, "Use create() method"

        self.x_1 = bounds[0]
        self.y_1 = bounds[1]
        self.x_2 = bounds[2]
        self.y_2 = bounds[3]

        self.label = label
        self.confidence = confidence

    def __str__(self) -> str:
        """
        To string.
        """
        return f"cls: {self.label}, conf: {self.confidence}, bounds: {self.x_1} {self.y_1} {self.x_2} {self.y_2}"

class DetectLandingPad:
    """
    Contains 2023 YOLOv8 model for prediction. 
    """

    def __init__(self, device: "str | int", conf: int, model_path: str) -> None:
        """
        device: name of target device to run inference on (i.e. "cpu" or cuda device 0, 1, 2, 3).
        conf: confidence threshold for detection
        model_path: path to 2023 YOLO model
        """
        self.__device = device
        self.__conf = conf
        self.__model = ultralytics.YOLO(model_path)

    def get_landing_pads(self, image: np.ndarray) -> "tuple[bool,  list[Detection] | None]":
        """
        Runs object detection on image and returns detections for blue landing pads.

        Return: success and detections
        """
        predictions = self.__model.predict(
            source=image,
            conf=self.__conf,
            device=self.__device,
            verbose=False,
        )

        if len(predictions) == 0:
            return False, None

        boxes = predictions[0].boxes
        if boxes.shape[0] == 0:
            return False, None

        object_bounds = boxes.xyxy.detach().cpu().numpy()

        # Loop over individual boxes and create list of valid bounding boxes.
        detections = []
        for i in range(0, boxes.shape[0]):
            bounds = object_bounds[i]
            label = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            result, detection = Detection.create(bounds, label, confidence)
            if result and detection:
                detections.append(detection)

        return True, detections

    def find_best_pad(self, detections: list[Detection]) -> "Detection | None":
        """
        Determine best landing pad to land on based on confidence.
        """
        if len(detections) == 0:
            return None

        best_landing_pad = min(detections, key=lambda pad: pad.confidence)
        return best_landing_pad
    