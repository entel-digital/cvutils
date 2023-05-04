import numpy as np
from ..helpers import box_iou_matrix

import cv2
from tqdm.auto import tqdm
from .filters import scale_detections
from ..database import dataframe_to_coco_dict
from ..video import VideoStream
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# from pycocotools.cocoevalLRP import COCOevalLRP


class ConfusionMatrix():
    def __init__(self, num_classes, confidence_thresh=0.5, iou_thresh=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.data = []
        self.ious_tp = {i: [] for i in range(num_classes)}

    def process_batch(self, gt_boxes, det_boxes):
        '''
        Arguments:
            gt_boxes (Array[M, 5]), class, xmin, ymin, xmax, ymax
            det_boxes (Array[N, 6]), class, xmin, ymin, xmax, ymax, conf
        Returns:
            None, updates confusion matrix accordingly
        '''

        det_boxes = det_boxes[det_boxes[:, 5] > self.confidence_thresh] if len(det_boxes) > 0 else np.array([])
        gt_classes = gt_boxes[:, 0].astype(np.int16)
        det_classes = det_boxes[:, 0].astype(np.int16) if len(det_boxes) > 0 else np.array([])

        all_ious = box_iou_matrix(gt_boxes[:, 1:], det_boxes[:, 1:5]) if len(det_boxes) > 0 else np.array([])
        want_idx = np.where(all_ious > self.iou_thresh)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            # Sort list of matches by descending IOU so we can remove duplicate detections while keeping the highest IOU entry.
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            # Remove duplicate detections from the list.
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            # Sort the list again by descending IOU. Removing duplicates doesn't preserve our previous sort.
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            # Remove duplicate ground truths from the list.
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        data = {"TP": [], "FP": [], "FN": [], "wrong_class": []}
        for i, label in enumerate(gt_boxes):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                det_class = det_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), det_class] += 1
                if gt_class != det_class:
                    data['wrong_class'].append({'gt': gt_boxes[i], 'det': det_boxes[int(all_matches[all_matches[:, 0] == i, 1][0])]})
                else:
                    iou_ex = self.iou(gt_boxes[i], det_boxes[int(all_matches[all_matches[:, 0] == i, 1][0])])
                    self.ious_tp[gt_class].append(iou_ex)
                    data['TP'].append({'gt': gt_boxes[i], 'det': det_boxes[int(all_matches[all_matches[:, 0] == i, 1][0])], 'iou': iou_ex})
            else:
                gt_class = gt_classes[i]
                self.matrix[(gt_class), self.num_classes] += 1
                data['FN'].append({'gt': gt_boxes[i], 'det': None})

        for i, detection in enumerate(det_boxes):
            if all_matches.shape[0] == 0 or all_matches[all_matches[:, 1] == i].shape[0] == 0:
                det_class = det_classes[i]
                self.matrix[self.num_classes, det_class] += 1
                data['FP'].append({'gt': None, 'det': det_boxes[i]})

        self.data.append(data)

    def return_matrix(self):
        return np.transpose(self.matrix)

    def return_data(self):
        return self.data

    def metrics_report(self):
        cm = self.return_matrix()
        nc = self.num_classes
        class_metrics = []
        general_metrics = {}
        for i in range(nc):
            TP = int(cm[i, i])
            FP = int(np.sum(cm[i, :])-TP)
            FN = int(np.sum(cm[:, i])-TP)
            class_metrics.append({'TP': TP, 'FP': FP, 'FN': FN,
                                  'Precision': TP/(TP+FP) if TP+FP > 0 else 0,
                                  'Recall': TP/(TP+FN) if TP+FN > 0 else 0,
                                  'F1': 2*TP/(2*TP+FP+FN) if TP+FP+FN > 0 else 0,
                                  'Average IOU (TP)': np.mean(self.ious_tp[i]),
                                  'Support': int(np.sum(cm[:, i]))
                                  })

        total_samples = int(np.sum([m['Support'] for m in class_metrics]))
        general_metrics['Accuracy'] = np.sum([m['TP'] for m in class_metrics])/total_samples
        general_metrics['Average Precision'] = np.mean([m['Precision'] for m in class_metrics])
        general_metrics['Average Recall'] = np.mean([m['Recall'] for m in class_metrics])
        general_metrics['Average IOU (TP)'] = np.sum([np.sum(itp) for itp in self.ious_tp.values()])/np.sum([len(itp) for itp in self.ious_tp.values()])
        general_metrics['Average F1'] = np.mean([m['F1'] for m in class_metrics])
        general_metrics['Weighted Precision'] = np.sum([m['Precision']*m['Support'] for m in class_metrics])/total_samples
        general_metrics['Weighted Recall'] = np.sum([m['Recall']*m['Support'] for m in class_metrics])/total_samples
        general_metrics['Weighted F1'] = np.sum([m['F1']*m['Support'] for m in class_metrics])/total_samples
        general_metrics['Total samples'] = total_samples
        return class_metrics, general_metrics

    def print_matrix(self):
        cm = self.return_matrix()
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, cm[i])))

    def iou(self, tlbr_left, tlbr_right):

        l_xmin, l_ymin, l_xmax, l_ymax = tlbr_left[:4]
        r_xmin, r_ymin, r_xmax, r_ymax = tlbr_right[:4]

        xa = max(l_xmin, r_xmin)
        ya = max(l_ymin, r_ymin)
        xb = min(l_xmax, r_xmax)
        yb = min(l_ymax, r_ymax)

        intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        boxAArea = (l_xmax - l_xmin + 1) * (l_ymax - l_ymin + 1)
        boxBArea = (r_xmax - r_xmin + 1) * (r_ymax - r_ymin + 1)

        return intersection / float(boxAArea + boxBArea - intersection)


class Evaluator():

    def __init__(self):
        pass

    @staticmethod
    def confusion_matrix_on_df(groundtruth_df, detections_df, confidence_thresh, iou_thresh, *args, labels=None, **kwargs):

        if labels is None:
            labels = sorted(groundtruth_df.label.unique())

        cm = ConfusionMatrix(len(labels), confidence_thresh=confidence_thresh, iou_thresh=iou_thresh)
        for filename in tqdm(groundtruth_df.filename.unique()):
            img_gt = groundtruth_df[groundtruth_df.filename == filename]
            img_dets = detections_df[detections_df.filename == filename]
            gts = np.array([[labels.index(r.label), r.xmin, r.ymin, r.xmax, r.ymax] for idx, r in img_gt.iterrows() if r.label in labels])
            dets = np.array([[labels.index(r.label), r.xmin, r.ymin, r.xmax, r.ymax, r.confidence] for idx, r in img_dets.iterrows() if r.label in labels])
            cm.process_batch(gts, dets)
        cm.data = {f: d for f, d in zip(groundtruth_df.filename.unique(), cm.data)}
        return cm, labels

    @staticmethod
    def pr_curves_on_df(groundtruth_df, detections_df, iou_thresh, *args, thresh_ini=0.3, thresh_end=1.0, thresh_samples=15, use_thresh_end=False, **kwargs):
        class_pr_curves = []
        general_pr_curves = {'Average Precision': [], 'Average Recall': [], 'Average F1': [],
                             'Weighted Precision': [], 'Weighted Recall': [], 'Weighted F1': [],
                             'Threshold': []}
        conf_threshs = np.linspace(thresh_ini, thresh_end, thresh_samples, endpoint=use_thresh_end)
        for conf_thresh in tqdm(conf_threshs):
            cm, labels = Evaluator.confusion_matrix_on_df(groundtruth_df, detections_df, conf_thresh, iou_thresh)
            class_metrics, general_metrics = cm.metrics_report()
            if len(class_pr_curves) == 0:
                class_pr_curves = [{'Precision': [], 'Recall': [], 'F1': [], 'Threshold': []} for c in class_metrics]
            for i, c in enumerate(class_metrics):
                class_pr_curves[i]['Precision'].append(c['Precision'])
                class_pr_curves[i]['Recall'].append(c['Recall'])
                class_pr_curves[i]['F1'].append(c['F1'])
                class_pr_curves[i]['Threshold'].append(conf_thresh)
            general_pr_curves['Average Precision'].append(general_metrics['Average Precision'])
            general_pr_curves['Average Recall'].append(general_metrics['Average Recall'])
            general_pr_curves['Average F1'].append(general_metrics['Average F1'])
            general_pr_curves['Weighted Precision'].append(general_metrics['Weighted Precision'])
            general_pr_curves['Weighted Recall'].append(general_metrics['Weighted Recall'])
            general_pr_curves['Weighted F1'].append(general_metrics['Weighted F1'])
            general_pr_curves['Threshold'].append(conf_thresh)
        return class_pr_curves, general_pr_curves, labels

    @staticmethod
    def inference_on_df(df, images_dir_path, detector, label_map, *args, confidence_thresh=0.3, **kwargs):
        # detector = DetectorXXX(model_path)
        # label_map = detector.load_label_map(label_map_path)
        filenames = df.filename.unique()
        detections = {
            'filename': [],
            'label': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
            'xtotal': [],
            'ytotal': [],
            'confidence': []
        }

        print("Processing images...")
        for filename in tqdm(filenames):
            image = cv2.imread(images_dir_path + '/' + filename)
            H, W = image.shape[:2]
            dets = detector.detect(image, confidence_thresh=confidence_thresh)
            dets = scale_detections(dets, W, H)
            for d in dets:
                (xmin, ymin, xmax, ymax) = d.to_tlbr()
                detections['filename'].append(filename)
                detections['label'].append(label_map[d.label])
                detections['xmin'].append(xmin)
                detections['ymin'].append(ymin)
                detections['xmax'].append(xmax)
                detections['ymax'].append(ymax)
                detections['xtotal'].append(W)
                detections['ytotal'].append(H)
                detections['confidence'].append(d.confidence)

        return detections

    @staticmethod
    def inference_on_video(video_path, detector, label_map, confidence_thresh):
        # detector = DetectorXXX(model_path)
        # label_map = detector.load_label_map(label_map_path)
        detections = {
            'filename': [],
            'label': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': [],
            'xtotal': [],
            'ytotal': [],
            'confidence': []
        }
        vs = VideoStream(video_path).start()
        W, H = (int(vs.width), int(vs.height))
        total_frames = int(vs.length)
        for nf in tqdm(range(total_frames)):
            frame = vs.read()
            if frame is None:
                break
            dets = detector.detect(frame, confidence_thresh=confidence_thresh)
            dets = scale_detections(dets, W, H)
            for d in dets:
                (xmin, ymin, xmax, ymax) = d.to_tlbr()
                detections['filename'].append(str(nf))
                detections['label'].append(label_map[d.label])
                detections['xmin'].append(xmin)
                detections['ymin'].append(ymin)
                detections['xmax'].append(xmax)
                detections['ymax'].append(ymax)
                detections['xtotal'].append(W)
                detections['ytotal'].append(H)
                detections['confidence'].append(d.confidence)

        detector.close()
        vs.stop()
        return detections

    @staticmethod
    def coco_metrics_on_df(groundtruth_df, detections_df, annotation_type='bbox'):
        # annotation_type: 'segm', 'bbox' or 'keypoints'
        # Corremos COCO API
        cocoGt = COCO()
        cocoGt.dataset = dataframe_to_coco_dict(groundtruth_df)
        cocoGt.createIndex()

        cocoDt = COCO()
        cocoDt.dataset = dataframe_to_coco_dict(detections_df, gt_dict=cocoGt.dataset)
        cocoDt.createIndex()

        cocoEval = COCOeval(cocoGt, cocoDt, annotation_type)
        # cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        stats_arr = [
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
        ]
        return {k: v for k, v in zip(stats_arr, cocoEval.stats)}

    '''
    @staticmethod
    def lrp_metrics_on_df(groundtruth_df, detections_df):
        # annotation_type: 'segm', 'bbox' or 'keypoints'
        # Corremos COCO API

        cocoGt = COCO()
        cocoGt.dataset = dataframe_to_coco_dict(groundtruth_df)
        cocoGt.createIndex()

        cocoDt = COCO()
        cocoDt.dataset = dataframe_to_coco_dict(detections_df, gt_dict=cocoGt.dataset)
        cocoDt.createIndex()

        cocoEvalLRP = COCOevalLRP(cocoGt, cocoDt)
        # cocoEval.params.imgIds  = imgIds
        cocoEvalLRP.evaluate()
        cocoEvalLRP.accumulate()
        cocoEvalLRP.summarize(detailed=1)
        return cocoEvalLRP.eval
    '''
