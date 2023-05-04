<p align="left">
<img src="https://zenodo.org/badge/DOI/10.5281/zenodo.2554189.svg">
</p>

# Metrics for object detection

### Dependencies

  - pandas
  - numpy
  - pyarrow
  - pathlib (python 3.5 or superior)
  - matplotlib
  - opencv-contrib (cv2)

## Table of contents

- [Different competitions, different metrics](#dependencies)
- [Different competitions, different metrics](#different-competitions-different-metrics)
- [Important definitions](#important-definitions)
- [Metrics](#metrics)
  - [Precision x Recall curve](#precision-x-recall-curve)
  - [Average Precision](#average-precision)
    - [11-point interpolation](#11-point-interpolation)
    - [Interpolating all  points](#interpolating-all-points)
- [**How to use this project**](#how-to-use-this-project)
- [References](#references)

<a name="different-competitions-different-metrics"></a>
## Different competitions, different metrics

* **[PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)** offers a Matlab script in order to evaluate the quality of the detected objects. Participants of the competition can use the provided Matlab script to measure the accuracy of their detections before submitting their results. The official documentation explaining their criteria for object detection metrics can be accessed [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000). The current metrics used by the current PASCAL VOC object detection challenge are the **Precision x Recall curve** and **Average Precision**.
The PASCAL VOC Matlab evaluation code reads the ground truth bounding boxes from XML files, requiring changes in the code if you want to apply it to other datasets or to your speficic cases. Even though projects such as [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) implement PASCAL VOC evaluation metrics, it is also necessary to convert the detected bounding boxes into their specific format. [Tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) framework also has their PASCAL VOC metrics implementation.

* **[COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)** uses different metrics to evaluate the accuracy of object detection of different algorithms. [Here](http://cocodataset.org/#detection-eval) you can find a documentation explaining the 12 metrics used for characterizing the performance of an object detector on COCO. This competition offers Python and Matlab codes so users can verify their scores before submitting the results. It is also necessary to convert the results to a [format](http://cocodataset.org/#format-results) required by the competition.

* **[Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)** also uses mean Average Precision (mAP) over the 500 classes to evaluate the object detection task.

* **[ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)** defines an error for each image considering the class and the overlapping region between ground truth and detected boxes. The total error is computed as the average of all min errors among all test dataset images. [Here](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation) are more details about their evaluation method.

## Important definitions

### Intersection Over Union (IOU)

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) and a predicted bounding box ![](http://latex.codecogs.com/gif.latex?B_p). By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Ctext%7BIOU%7D%20%3D%20%5Cfrac%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccap%20B_%7Bgt%7D%5Cright%29%7D%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccup%20B_%7Bgt%7D%5Cright%29%7D">
</p>

The image below illustrates the IOU between a ground truth bounding box (in green) and a detected bounding box (in red).

<!--- IOU --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/iou.png" align="center"/></p>

### True Positive, False Positive, False Negative and True Negative

Some basic concepts used by the metrics:

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_
* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_
* **False Negative (FN)**: A ground truth not detected
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

_threshold_: depending on the metric, it is usually set to 50%, 75% or 95%.

### Precision

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

### Recall

Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

## Metrics

In the topics below there are some comments on the most popular metrics used for object detection.

### Precision x Recall curve

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).

A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.

### Average Precision

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1.

From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. Currently, **the interpolation performed by PASCAL VOC challenge uses all data points, rather than interpolating only 11 equally spaced points as stated in their [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)**. As we want to reproduce their default implementation, our default code (as seen further) follows their most recent application (interpolating all data points). However, we also offer the 11-point interpolation approach.

#### 11-point interpolation

The 11-point interpolation tries to summarize the shape of the Precision x Recall curve by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%5Csum_%7Br%5Cin%5C%7B0%2C0.1%2C...%2C1%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D">
</p>

with

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D%3D%5Cmax_%7B%5Cwidetilde%7Br%7D%3A%5Cwidetilde%7Br%7D%5Cgeqslant%7Br%7D%7D%20%5Crho%20%5Cleft%20%28%5Cwidetilde%7Br%7D%20%5Cright%29">
</p>

where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision only at the 11 levels ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r)**.

#### Interpolating all points

Instead of interpolating only in the 11 equally spaced points, you could interpolate through all points in such way that:

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Csum_%7Br%3D0%7D%5E%7B1%7D%20%5Cleft%20%28%20r_%7Bn&plus;1%7D%20-%20r_n%5Cright%20%29%20%5Crho_%7Binterp%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29">
</p>


with

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Crho_%7Binterp%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%5Cgeq%20r_%7Bn&plus;1%7D%7D%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>


where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

In this case, instead of using the precision observed at only few points, the AP is now obtained by interpolating the precision at **each level**, ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater or equal than ![](http://latex.codecogs.com/gif.latex?r&plus;1)**. This way we calculate the estimated area under the curve.


## How to use this project

### Ground truth files

- The ground truth files are loaded directly from the Pascal VOC `Annotations` folder in xml format.
- E.g. The ground truth bounding boxes of the image "2008_000034.jpg" are represented in the file "2008_000034.xml". An example of this xml is shown below:

  ```
    -<annotation>
    <filename>0b2c989d-b970-4ae1-9cc8-ad3be652cf31.jpg</filename>
    -<size>
        <width>1300</width>
        <height>867</height>
        <depth>3</depth>
    </size>
    -<object>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <name>sin casco</name>
        -<bndbox>
            <xmin>188</xmin>
            <ymin>114</ymin>
            <xmax>328</xmax>
            <ymax>214</ymax>
        </bndbox>
    </object>
    </annotation>
  ```

### Load your detection files

- Create a separate detection json file for each image.
- The names of the detection files must match their correspond ground truth (e.g. "detections/2008_000182.json" represents the detections of the ground truth: "groundtruths/2008_000182.xml").
- In these files each dictionary should be in the following format: `{"tlwh": [top, left, width, height], "label": int | str, "confidence": float}
- E.g. "2008_000034.json":
    ```
    [{"tlwh": [838.4173135757446, 654.5107040405273, 172.59111213684082, 263.5826110839844],
      "label": 3,
      "confidence": 0.7578125},
     {"tlwh": [207.66520071029663, 722.5497894287109, 292.9911060333252, 264.42333984375],
      "label": 0,
      "confidence": 0.7265625},
     {"tlwh": [392.91198348999023, 657.409309387207, 125.44750785827637, 101.21553039550781],
      "label": 0,
      "confidence": 0.69140625}]
    ```

### Do the detections on the fly

- You need to pass a tflite model which is compatible with Coral TPU (only kind supported rigth now).
- The model will save the detections in the same way as the detections are loaded.
- Next model to be supported: tensorflowlite and tensorflow model.


### Arguments

Arguments:

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `--output_folder` | Folder where the plots, detections and results are saved | `python metrics.py --output_folder /home/whatever/` | |
| `--dataset_folder` | VOC folder. Contains `Annotations` with ground_truth files, `JPEGImages` with the jpg images and `ImageSets` with the images splits. | `python metrics.py -dataset_folder /home/whatever/EPP/` | |
| `--labelmap_path` | Labelmap dictionary. Necesary to transform the labels. | `python metrics.py --labelmap_path /home/whatever/detections/config/epp_label_map.pbtxt` | |
| `--iou_threshold` | IOU thershold that tells if a detection is TP or FP | `python metrics.py -iou_threshold 0.75` | `0.50` |
| `--model` | Can be a detection model or a directory with stored detections. | `python metrics.py --model /home/whatever/epp_model_edgetpu.tflite` or `python metrics.py --model /home/whatever/stored_detections/` | None |
| `--save_det` | Save detections. Only needed if detections are done on the fly. | `python metrics.py --save_det True` | False |
| `--metric` | Metric that you want. Either `pascal` or `coco`. |  `python metrics.py --metric coco` | pascal |


## References

* The Relationship Between Precision-Recall and ROC Curves (Jesse Davis and Mark Goadrich)
Department of Computer Sciences and Department of Biostatistics and Medical Informatics, University of
Wisconsin
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

* The PASCAL Visual Object Classes (VOC) Challenge
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf

* Evaluation of ranked retrieval results (Salton and Mcgill 1986)
https://www.amazon.com/Introduction-Information-Retrieval-COMPUTER-SCIENCE/dp/0070544840
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
