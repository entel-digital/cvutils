# Detection
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.structures import ImageList
from detectron2.config import get_cfg
from ..detection import AbstractDetector, Detection


class DetectorDetectron2(AbstractDetector):
    def __init__(self, model_path, weights_path, labels_file=None, confidence_thresh=0.05, device="cuda"):
        super().__init__(model_path, weights_path=weights_path, labels_file=labels_file)
        # Carga de modelo y pesos
        self.cfg = get_cfg()
        self.cfg.set_new_allowed(True)
        self.cfg.merge_from_file(model_path)
        # self.cfg.merge_from_list(args.opts)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = device  # cuda or cpu
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_thresh
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_thresh
        # Make this CfgNode and all of its children immutable
        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        image_BGR = image[:, :, ::-1] if color_order == 'RGB' else image  # RGB2BGR

        # Run inference
        outputs = self.predictor(image_BGR)

        # Get all output details
        boxes = outputs['instances'].pred_boxes.tensor.tolist()
        classes = outputs['instances'].pred_classes.tolist()
        scores = outputs['instances'].scores.tolist()
        has_masks = outputs['instances'].has('pred_masks')
        if has_masks:
            masks = outputs['instances'].pred_masks.tolist()
        num_detections = len(boxes)
        (H, W) = image_BGR.shape[:2]

        # create Detections array
        detections = []
        for i in range(num_detections):
            # filter out objects not in desired classes
            if desired_classes is not None and int(classes[i]) not in desired_classes:
                continue
            # filter out weak detections by requiring a minimum confidence
            if scores[i] < confidence_thresh:
                continue

            (xmin, ymin, xmax, ymax) = boxes[i] / np.array([W, H, W, H])
            d = Detection(
                (xmin, ymin, xmax-xmin, ymax-ymin),
                int(classes[i]),
                scores[i])
            if has_masks:
                d.mask = np.array(masks[i])[ymin:ymax, xmin:xmax]
            detections.append(d)
        return detections

    def close(self):
        return


class DetectorDetectron2Features(AbstractDetector):
    def __init__(self, model_path, weights_path, labels_file=None, confidence_thresh=0.05, device="cuda"):
        super().__init__(model_path, weights_path=weights_path, labels_file=labels_file)
        # Carga de modelo y pesos
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_path)
        # self.cfg.merge_from_list(args.opts)
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = device  # cuda or cpu
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_thresh
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_thresh
        # Make this CfgNode and all of its children immutable
        self.cfg.freeze()
        self.predictor = DetectorDetectron2Predictor(self.cfg)

    def detect(self, image, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        image_BGR = image[:, :, ::-1] if color_order == 'RGB' else image  # RGB2BGR

        # Run inference
        predictions = self.predictor([image_BGR])
        detections = self._translate_predictions(predictions, desired_classes, confidence_thresh)
        return detections[0]

    def detect_on_batch(self, images, color_order='BGR', confidence_thresh=0, mask_thresh=None, desired_classes=None, path=None):
        images_BGR = [image[:, :, ::-1] if color_order == 'RGB' else image for image in images]  # RGB2BGR
        predictions = self.predictor(images_BGR)
        detections = self._translate_predictions(predictions, desired_classes, confidence_thresh)
        return detections

    def _translate_predictions(self, predictions, desired_classes, confidence_thresh):
        # Get all output details
        images_detections = []
        for prediction in predictions:
            num_detections = len(prediction['instances'])
            (H, W) = prediction['instances'].image_size
            boxes = prediction['instances'].pred_boxes.tensor.tolist()
            classes = prediction['instances'].pred_classes.tolist()
            scores = prediction['instances'].scores.tolist()
            has_masks = prediction['instances'].has('pred_masks')
            if has_masks:
                masks = prediction['instances'].pred_masks.tolist()
            features = prediction.get('features')
            if features is None:
                features = [None for _ in range(num_detections)]
            else:
                features = features.cpu()

            # create Detections array
            detections = []
            for i in range(num_detections):
                feature = features[i]
                # filter out objects not in desired classes
                if desired_classes is not None and int(classes[i]) not in desired_classes:
                    continue
                # filter out weak detections by requiring a minimum confidence
                if scores[i] < confidence_thresh:
                    continue

                (xmin, ymin, xmax, ymax) = boxes[i] / np.array([W, H, W, H])
                d = Detection(
                    (xmin, ymin, xmax-xmin, ymax-ymin),
                    int(classes[i]),
                    scores[i], feature=feature)
                if has_masks:
                    d.mask = np.array(masks[i])[ymin:ymax, xmin:xmax]
                detections.append(d)
            images_detections.append(detections)
        return images_detections

    def close(self):
        return


class DetectorDetectron2Predictor(DefaultPredictor):

    def __call__(self, images):
        """
        Args:
            images (list): List of images of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for the images with its features
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            features, outputs = self.predict(images)
            predictions = self.get_mask_features(features, outputs)
            return predictions

    def _preprocess_images(self, batched_imgs):
        """
        List of numpy array imgs
        Normalize, pad and batch the input images.
        """
        images = []
        batched_input = []
        for im in batched_imgs:
            height, width = im.shape[:2]
            image = self.aug.get_transform(im).apply_image(im)
            image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1)).cuda()
            processed_img = {'image': image, "height": height, "width": width}
            image = (processed_img['image'] - self.model.pixel_mean) / self.model.pixel_std
            images.append(image)
            batched_input.append(processed_img)
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return batched_input, images

    def predict(self, im_list):
        """
        Args:
            im_list (list): List of images of shape (H, W, C) (in BGR order).
        """
        batched_input, images = self._preprocess_images(im_list)
        with torch.no_grad():
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features)
            instances, _ = self.model.roi_heads(images, features, proposals)

        outputs = self.model._postprocess(instances, batched_input, images.image_sizes)
        return features, outputs

    def get_mask_features(self, features, outputs):
        """
        Appends 'features' to instance dict.
        """
        for i, instance in enumerate(outputs):
            mask_features = [features[f][i].unsqueeze(0) for f in self.model.roi_heads.in_features]
            pred_boxes = instance['instances'].pred_boxes
            mask_instance_features = self.model.roi_heads.box_pooler(mask_features, [pred_boxes])
            instance['features'] = mask_instance_features
        return outputs
