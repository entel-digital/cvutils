import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pathlib import Path

demos_folder = Path(__file__).parent
# Prueba básica detectron
im = cv2.imread(f"{demos_folder}/resources/images/detectron2.jpg")
cv2.imshow("Image", im)
cv2.waitKey(0)

# Carga de configuración y modelo
cfg = get_cfg()
cfg.merge_from_file(f"{demos_folder}/resources/models/detectron2/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = f"{demos_folder}/resources/models/detectron2/mask_rcnn_R_50_FPN_3x.pkl"

cfg.merge_from_file(f"{demos_folder}/resources/models/detectron2/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = f"{demos_folder}/resources/models/detectron2/mask_rcnn_R_50_FPN_3x.pkl"

# Descargar la configuración desde la web, quedan en la carpeta del usuario, por ejemplo:
# C:\Users\Pablo/.torch/fvcore_cache\detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431\model_final_a54504.pkl
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda"  # we use a GPU Detectron copy
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
predictor = DefaultPredictor(cfg)

# Detección de objetos
outputs = predictor(im)

# Visualización de detecciones en imagen
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Detections", v.get_image()[:, :, ::-1])
cv2.waitKey(0)
