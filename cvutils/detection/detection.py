import numpy as np
import json
from pathlib import Path


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    label : int | str
        Index or label of detection class
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    mask : array_like
        A mask matrix corresponding to detection bounding box.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    label : int | str
        Index or label of detection class
    confidence : float
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    mask : ndarray | NoneType
        A mask matrix corresponding to detection bounding box.

    """

    def __init__(self, tlwh, label, confidence, feature=None, mask=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.label = label
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None
        self.mask = np.asarray(mask, dtype=np.float32) if mask is not None else None

    def __repr__(self):
        return "Detection: label={}, conf={}, tlwh=[{:.3f}, {:.3f}, {:.3f}, {:.3f}], feature={}, mask={}".format(
            self.label,
            self.confidence,
            self.tlwh[0], self.tlwh[1], self.tlwh[2], self.tlwh[3],
            self.feature,
            self.mask)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_dict(self):
        d = {
            'tlwh': self.tlwh.tolist(),
            'label': self.label,
            'confidence': self.confidence
        }
        if self.feature is not None:
            d['feature'] = self.feature.tolist()
        if self.mask is not None:
            d['mask'] = self.mask.tolist()
        return d

    def get_tlbr(self):
        return self.to_tlbr()


class AbstractDetector:
    def __init__(self, model_path, weights_path=None, labels_file=None):
        if labels_file is not None:
            self.labels = self.load_label_map(labels_file)

    def detect(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        raise NotImplementedError("Please Implement this method")

    def predict(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        raise NotImplementedError("Please Implement this method")

    def close(self):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def decode_predictions(predictions, label_map):
        for p in predictions:
            p.label = label_map[p.label]
        return predictions

    # load label map from a file
    @staticmethod
    def load_label_map(labels_file):
        labels = []
        labels_path = Path(labels_file)
        with open(labels_path, 'r') as f:
            if labels_path.suffix == '.json':
                data = json.load(f)
                labels = {int(k): v[1] for k, v in data.items()}
            elif labels_path.suffix == '.txt':
                labels = {i: line.strip() for i, line in enumerate(f.readlines())}
            elif labels_path.suffix == '.pbtxt':
                items = []
                line = f.readline()
                while line:
                    data = line.strip()
                    if data == "item {":
                        item = {}
                        line = f.readline()
                        while line:
                            data = line.strip()
                            if data == "}":
                                items.append(item)
                                break
                            data = data.split(":")
                            item[data[0].strip()] = data[1].strip().replace('"', '').replace("'", "")
                            line = f.readline()
                    line = f.readline()
                category_index = {}
                for item in items:
                    cat_id = int(item['id']) if 'id' in item else int(item['label'])
                    category_index[cat_id] = {}
                    for key, val in item.items():
                        category_index[cat_id][key] = val
                labels = {int(k): category_index[k]['display_name'] for k in category_index.keys()}
        return labels

class Detector(AbstractDetector):
    def __init__(self, model_path, weights_path=None, labels_file=None, detector_type='detectron2', *args, **kwargs):
        if detector_type == "detectron2":
            from .detectron2 import DetectorDetectron2
            self.detector = DetectorDetectron2(model_path, weights_path=weights_path, labels_file=labels_file, *args, **kwargs)
        elif detector_type == "tensorflow":
            from .tensorflow import DetectorTF
            self.detector = DetectorTF(model_path, weights_path=weights_path, labels_file=labels_file, *args, **kwargs)
        elif detector_type == "tensorflow2":
            from .tensorflow import DetectorTF2
            self.detector = DetectorTF2(model_path, weights_path=weights_path, labels_file=labels_file, *args, **kwargs)
        else:
            raise ValueError("'detector_type' can only be one of {'detectron2', 'tensorflow', 'tensorflow2'}")

    def detect(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        return self.detector.detect(image, color_order=color_order, confidence_thresh=confidence_thresh, mask_thresh=mask_thresh, desired_classes=desired_classes, path=path)

    def predict(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        return self.detector.predict(image, color_order=color_order, confidence_thresh=confidence_thresh, mask_thresh=mask_thresh, desired_classes=desired_classes, path=path)

    def close(self):
        return self.detector.close()
class Trainer:
    def __init__(self, config):
        raise NotImplementedError("Please Implement this method")

    def prepare_datasets(self):
        raise NotImplementedError("Please Implement this method")

    def train(self):
        raise NotImplementedError("Please Implement this method")

    def export(self):
        raise NotImplementedError("Please Implement this method")
