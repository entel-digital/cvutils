from cvutils.pipeline_task.pipeline_task import PipelineTask
from shapely import box, centroid, contains


class ObjectClassInROI(PipelineTask):
    def __init__(self, detector_classes, rois, overlapping='centroid'):
        self.detector_classes = detector_classes
        self.rois = rois
        self.overlapping = overlapping

        super().__init__()

    def map(self, data):
        for detection in data['detection_result'].detections:
            xmin, ymin = detection.bounding_box.bbox.origin_x, detection.bounding_box.bbox.origin_y
            xmax = xmin + detection.bounding_box.bbox.width
            ymax = ymin + detection.bounding_box.bbox.height
            for roi in self.rois:
                box = box(xmin, ymin, xmax, ymax)
                if contains(roi, centroid(box)):
                    print('PERSON IN ROI!!!!!!')

        return data
