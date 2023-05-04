from cvutils.detection import AbstractDetector, Detection
import json


class DetectorJSON(AbstractDetector):
    def __init__(self, model_path=None, weights_path=None, confidence_thresh=0.05, device="cuda"):
        pass

    def detect(self, image, color_order='BGR', path=None):
        detections = []
        if path is not None:
            with path.open('r') as f:
                detections = json.load(f)
                detections = [Detection(d['tlwh'], d['label'], d['confidence'], d.get("feature"), d.get("mask")) for d in detections]
        return detections

    def close(self):
        return
