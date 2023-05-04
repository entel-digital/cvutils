import numpy as np
import cv2
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def threshold_scale_detections(detections, width, height, confidence_thresh=0.5, mask_thresh=0.5, scale=1, translate=(0, 0), desired_classes=None):
    """
    Returns only detections that are above the confidence threshold and
    belong to the desired classes but detections are scaled by the width
    and height of the image. Detections can also be scaled by a scale factor
    and translated by a translate factor.
    """
    dets = []
    # loop over the detections
    for d in detections:
        # filter out objects not in desired classes
        if desired_classes is not None and d.label not in desired_classes:
            continue
        # filter out weak detections by requiring a minimum confidence
        if d.confidence <= confidence_thresh:
            continue
        # scale and translate detection boundingbox
        d.tlwh = (d.tlwh * np.array([width, height, width, height]) + np.array([translate[0], translate[1], 0, 0])) * scale
        # handle detections with masks:
        if d.mask is not None:
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            d.mask = cv2.resize(d.mask, tuple(d.tlwh[2:].astype(int)), interpolation=cv2.INTER_NEAREST)
            d.mask = (d.mask > mask_thresh)
        dets.append(d)
    return dets


def threshold_detections(detections, confidence_thresh=0.5, mask_thresh=0.5, desired_classes=None):
    """
    Returns only detections that are above the confidence threshold and
    belong to the desired classes.
    """
    dets = []
    for d in detections:
        # filter out objects not in desired classes
        if desired_classes is not None and d.label not in desired_classes:
            continue
        # filter out weak detections by requiring a minimum confidence
        if d.confidence > confidence_thresh:
            if d.mask is not None:
                # threshold to create a *binary* mask
                d.mask = (d.mask > mask_thresh)
            dets.append(d)
    return dets


def scale_detections(detections, width, height, scale=1, translate=(0, 0)):
    """
    Detections are scaled by the width and height of the image.
    Detections can also be scaled by a scale factor
    and translated by a translate factor.
    """
    dets = copy.deepcopy(detections)
    for d in dets:
        # scale and translate detection boundingbox
        d.tlwh = (d.tlwh * np.array([width, height, width, height]) + np.array([translate[0], translate[1], 0, 0])) * scale
        if d.mask is not None:
            # extract the pixel-wise segmentation for the object and resize
            # the mask such that it's the same dimensions of the bounding box
            d.mask = cv2.resize(d.mask, tuple(d.tlwh[2:].astype(int)), interpolation=cv2.INTER_NEAREST)
    return dets


def class_filter_detections(detections, desired_classes):
    """
    Returns only detections that belongs to the desired classes.
    """
    return [d for d in detections if d.label in desired_classes]


def non_max_suppression(detections, max_bbox_overlap):
    """Suppress overlapping detections.
    Original code from [1]_ has been adapted to include confidence score.
    .. [1] http://www.pyimagesearch.com/2015/02/16/
        faster-non-maximum-suppression-python/
    Examples
    --------
        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]
    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.
    Returns
    -------
    List[int]
        Returns detections that have survived non-maxima suppression.
    """
    if len(detections) == 0:
        return []
    boxes = (np.array([d.tlwh for d in detections])).astype(np.float)
    scores = np.array([d.confidence for d in detections])
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # All scores are not None
    idxs = np.argsort(scores) if all(scores) else np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))
    return [detections[i] for i in pick]


def non_max_suppression_soft(detections, thresh):
    # https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/py_cpu_nms.py
    """Pure Python NMS baseline."""
    if len(detections) == 0:
        return []
    dets = np.array([d.to_tlbr() for d in detections]).astype(np.float)
    scores = np.array([d.confidence for d in detections])

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    # return keep
    return [detections[k] for k in keep]


def inside_roi_detections(detections, roi, point_of_interest='bottom-center', epsilon=0.0001):
    """
    Returns all the detections which point of interest is inside the roi.
    """
    x_translation = {'left': 0, 'center': 1/2, 'right': 1}
    y_translation = {'top': 0, 'middle': 1/2, 'bottom': 1}
    y, x = point_of_interest.split('-')
    y, x = y_translation[y], x_translation[x]
    roi = np.array(roi)
    roi = np.where(roi < epsilon, -epsilon, roi)
    roi = np.where(roi > 1 - epsilon, 1 + epsilon, roi)
    return [d for d in detections if Polygon(roi).intersects(Point(d.tlwh[:2] + [x*d.tlwh[2], y*d.tlwh[3]]))]


def intersection_roi_detections(detections, roi, intersection_threshold, relative_to="detection"):
    """
    Returns all the detections that have a ROI intersection area
    above the intersection threshold. The intersection can be considered relative to
    the detection are or the ROI area.
    """
    filtered_detections = []
    roi_polygon = Polygon(roi)
    for d in detections:
        x, y, w, h = d.tlwh
        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        detection_polygon = Polygon(points)
        intersection_area = roi_polygon.intersection(detection_polygon).area
        intersection_area = intersection_area/detection_polygon.area if relative_to == "detection" else intersection_area/roi_polygon.area
        if intersection_area > intersection_threshold:
            filtered_detections.append(d)
    return filtered_detections


def smaller_than_size_detections(detections, w, h):
    """
    Return detections if either width or height are smaller than a
    respective size.
    """
    return [d for d in detections if (d.tlwh[2] <= w and d.tlwh[3] <= h)]


def bigger_than_size_detections(detections, w, h):
    """
    Return detections if either width or height are bigger than a
    respective size.
    """
    return [d for d in detections if (d.tlwh[2] >= w and d.tlwh[3] >= h)]
