# Train
import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorboard import program
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.samplers import RepeatFactorTrainingSampler
from ..detection import Trainer

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()


def range_is_correct(min_value, max_value, minimum_allowed_value=None, maximum_allowed_value=None):
    """
    Checks that the max_value is bigger than the min_value
    """
    if min_value is None or max_value is None:
        return False
    if max_value <= min_value:
        return False
    if minimum_allowed_value is not None:
        if min_value < minimum_allowed_value:
            return False
    if maximum_allowed_value is not None:
        if max_value > maximum_allowed_value:
            return False
    return True


def check_random_flip_direction(direction):
    """
    Check the direction and return
    """
    if direction is None:
         return None
    if direction.lower() == "horizontal":
        return {"horizontal": True, "vertical": False}
    if direction.lower() == "vertical":
        return {"horizontal": False, "vertical": True}
    return None


def build_train_aug(cfg):
    """
    Add augmentations methods to the dataloader
    """
    # Default model_final.yaml augmentation methods are ResizeShortestEdge
    # and RandomCrop
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    # Crop
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    # Horizontal Random flip
    random_flip_dict = check_random_flip_direction(cfg.INPUT.RANDOM_FLIP)
    if random_flip_dict:
        augs.append(T.RandomFlip(horizontal=random_flip_dict['horizontal'], vertical=random_flip_dict['vertical']))
    vertical_random_flip_dict = check_random_flip_direction(cfg.INPUT.VERTICAL_RANDOM_FLIP)
    if vertical_random_flip_dict:
        augs.append(T.RandomFlip(horizontal=vertical_random_flip_dict['horizontal'], vertical=vertical_random_flip_dict['vertical']))
    # Random rotation
    if range_is_correct(cfg.INPUT.RANDOM_ROTATION_MIN_ANGLE, cfg.INPUT.RANDOM_ROTATION_MAX_ANGLE, -360, 360):
        augs.append(T.RandomRotation([cfg.INPUT.RANDOM_ROTATION_MIN_ANGLE, cfg.INPUT.RANDOM_ROTATION_MAX_ANGLE]))
    # Random contrast
    if range_is_correct(cfg.INPUT.RANDOM_CONTRAST_MIN_INTENSITY, cfg.INPUT.RANDOM_CONTRAST_MAX_INTENSITY, 0, 2):
        augs.append(T.RandomContrast(cfg.INPUT.RANDOM_CONTRAST_MIN_INTENSITY, cfg.INPUT.RANDOM_CONTRAST_MAX_INTENSITY))
    # Random brightness
    if range_is_correct(cfg.INPUT.RANDOM_BRIGHTNESS_MIN_INTENSITY, cfg.INPUT.RANDOM_BRIGHTNESS_MAX_INTENSITY, 0, 2):
        augs.append(T.RandomBrightness(cfg.INPUT.RANDOM_BRIGHTNESS_MIN_INTENSITY, cfg.INPUT.RANDOM_BRIGHTNESS_MAX_INTENSITY))
    # Random saturation
    if range_is_correct(cfg.INPUT.RANDOM_SATURATION_MIN_INTENSITY, cfg.INPUT.RANDOM_SATURATION_MAX_INTENSITY, 0, 2):
        augs.append(T.RandomSaturation(cfg.INPUT.RANDOM_SATURATION_MIN_INTENSITY, cfg.INPUT.RANDOM_SATURATION_MAX_INTENSITY))
    # Random ligthting
    if cfg.INPUT.RANDOM_LIGHTING_INTENSITY is not None:
        augs.append(T.RandomLighting(cfg.INPUT.RANDOM_LIGHTING_INTENSITY))
    # Random extent
    if (range_is_correct(cfg.INPUT.RANDOM_EXTENT_MIN_SCALE, cfg.INPUT.RANDOM_EXTENT_MAX_SCALE, 0, 2) and
        range_is_correct(cfg.INPUT.RANDOM_EXTENT_MIN_SHIFT, cfg.INPUT.RANDOM_EXTENT_MAX_SHIFT, 0, 1)):
        augs.append(T.RandomExtent([cfg.INPUT.RANDOM_EXTENT_MIN_SCALE, cfg.INPUT.RANDOM_EXTENT_MAX_SCALE],
                                   [cfg.INPUT.RANDOM_EXTENT_MIN_SHIFT, cfg.INPUT.RANDOM_EXTENT_MAX_SHIFT]))
    return augs


def add_default_data_augmentation(cfg):
    """
    Adds default augmentation params to the CfgNode that
    can be overwriten.
    """
    # Random flip
    cfg.INPUT.RANDOM_FLIP = None
    cfg.INPUT.VERTICAL_RANDOM_FLIP = None
    # Random rotation
    cfg.INPUT.RANDOM_ROTATION_MIN_ANGLE = None
    cfg.INPUT.RANDOM_ROTATION_MAX_ANGLE = None
    # Random contrast
    cfg.INPUT.RANDOM_CONTRAST_MIN_INTENSITY = None
    cfg.INPUT.RANDOM_CONTRAST_MAX_INTENSITY = None
    # Random brightness
    cfg.INPUT.RANDOM_BRIGHTNESS_MIN_INTENSITY = None
    cfg.INPUT.RANDOM_BRIGHTNESS_MAX_INTENSITY = None
    # Random saturation
    cfg.INPUT.RANDOM_SATURATION_MIN_INTENSITY = None
    cfg.INPUT.RANDOM_SATURATION_MAX_INTENSITY = None
    # Random lighting
    cfg.INPUT.RANDOM_LIGHTING_INTENSITY = None
    # Random extent
    cfg.INPUT.RANDOM_EXTENT_MIN_SCALE = None
    cfg.INPUT.RANDOM_EXTENT_MAX_SCALE = None
    cfg.INPUT.RANDOM_EXTENT_MIN_SHIFT = None
    cfg.INPUT.RANDOM_EXTENT_MAX_SHIFT = None
    return cfg


def add_balance_sampler_to_cfg(cfg, classes):
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 1/len(classes)


class TrainerDetectron2(Trainer):


    def __init__(self, config, dataset_prefix='odd_db_', balance_sampler=False):
        self.config = config
        self.dataset_prefix = dataset_prefix
        self.balance_sampler = balance_sampler
        # Lectura de base de datos
        self.df = pd.read_feather(self.config['annotations_path'], columns=None, use_threads=True)
        self.classes = sorted(self.df.label.unique().tolist())
        self.config_detectron = self.load_configuration()


    def prepare_datasets(self):
        self.imgs_dir_path = self.config['images_dir_path']
        dataset_json_format = os.path.join(self.config['intermediate_files_dir_path'], "annotation_{}.json")
        dataset_name_format = self.dataset_prefix + "{}"

        # Creamos el directorio de salida si no existe
        Path(self.config['intermediate_files_dir_path']).mkdir(parents=True, exist_ok=True)

        # Registramos los conjuntos
        print("Registering Datasets...")
        for split in ["train", "val", "test"]:
            print(f"{split} Dataset")
            split_df = self.df[self.df.split == split]
            self.create_dataset_json(split_df, self.classes, dataset_json_format.format(split))
            register_coco_instances(dataset_name_format.format(split), {}, dataset_json_format.format(split), self.imgs_dir_path)
        # DatasetCatalog.register(DATASET_PREFIX + "train", lambda: create_dataset_dicts(train_df, classes, self.imgs_dir_path))
        # MetadataCatalog.get(DATASET_PREFIX + "train").set(thing_classes=classes)


    def train(self):
        sampler = None
        cfg = self.config_detectron
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config['train_dir_path'] + '/eval', exist_ok=True)

        # Previous Checkpoint check
        if 'previous_checkpoint_dir_path' in self.config and self.config['previous_checkpoint_dir_path'] is not None and os.path.isdir(self.config['previous_checkpoint_dir_path']):
            print('Previous checkpoint found')
            src_files = os.listdir(self.config['previous_checkpoint_dir_path'])
            for file_name in src_files:
                full_file_name = os.path.join(self.config['previous_checkpoint_dir_path'], file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, cfg.OUTPUT_DIR)

        # Tensorboard
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--host', '0.0.0.0', '--logdir', cfg.OUTPUT_DIR])
        url = tb.launch()
        print(f"Launched tensorboard at {url}")

        # Entrenamiento
        print("Training...")
        if self.balance_sampler:
            add_balance_sampler_to_cfg(cfg, self.classes)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()


    def export(self):
        export_dir = self.config['train_dir_path'] + '/export'
        os.makedirs(export_dir, exist_ok=True)
        shutil.copy2(self.config_detectron.OUTPUT_DIR + '/model_final.pth', export_dir)
        with open(export_dir + '/model_final.yaml', 'w') as f:
            self.config_detectron.dump(stream=f)
        with open(export_dir + '/label_map.pbtxt', 'w') as f:
            for i in range(len(self.classes)):
                f.write(f'item {{\n  id: {i}\n  name: "{self.classes[i]}"\n  display_name: "{self.classes[i]}"\n}}\n')


    def load_configuration(self):
        print("Loading Configuration...")
        model_path = model_zoo.get_config_file(self.config['model']) if self.is_remote(self.config['model']) else self.config['model']
        weights_path = model_zoo.get_checkpoint_url(self.config['weights']) if self.is_remote(self.config['weights']) else self.config['weights']

        cfg = get_cfg()
        cfg = add_default_data_augmentation(cfg)
        cfg.merge_from_file(model_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.OUTPUT_DIR = self.config['train_dir_path'] + '/train'
        cfg.DATASETS.TRAIN = (self.dataset_prefix + "train",)
        cfg.DATASETS.TEST = (self.dataset_prefix + "val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.classes)
        cfg = self.merge_params(self.config, cfg)
        return cfg


    def merge_params(self, params, cfg):
        cfg_list = []
        for k, v in params['cfg_setup'].items():
            cfg_list.append(k)
            cfg_list.append(v)
        cfg.merge_from_list(cfg_list)
        return cfg


    def is_remote(self, path_or_url: str):
        if path_or_url.startswith('https://') or \
           path_or_url.startswith('http://') or \
           path_or_url.startswith('COCO-Detection/') or \
           path_or_url.startswith('COCO-InstanceSegmentation/') or \
           path_or_url.startswith('COCO-Keypoints/') or \
           path_or_url.startswith('COCO-PanopticSegmentation/') or \
           path_or_url.startswith('LVISv0.5-InstanceSegmentation/') or \
           path_or_url.startswith('Cityscapes/') or \
           path_or_url.startswith('PascalVOC-Detection/') or \
           path_or_url.startswith('Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/') or \
           path_or_url.startswith('Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv/'):
            return True
        return False


    @staticmethod
    def create_dataset_dicts(df, classes, imgs_dir_path):
        dataset_dicts = []
        for img_id, img_name in enumerate(df.filename.unique()):
            record = {}
            img_df = df[df.filename == img_name]
            file_path = f'{imgs_dir_path}/{img_name}'
            record["file_name"] = file_path
            record["image_id"] = img_id
            record["height"] = int(img_df.iloc[0].ytotal)
            record["width"] = int(img_df.iloc[0].xtotal)

            objs = []
            for _, row in img_df.iterrows():
                xmin = int(row.xmin)
                ymin = int(row.ymin)
                xmax = int(row.xmax)
                ymax = int(row.ymax)

                obj = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [],
                    "category_id": classes.index(row.label),
                    "iscrowd": 0
                }
                # Esto es un artilugio para poder entrenar una mask-RCNN, mejor no usarlo si no se tienen mascaras
                # poly = [
                #     (xmin, ymin), (xmax, ymin),
                #     (xmax, ymax), (xmin, ymax)
                # ]
                # poly = list(itertools.chain.from_iterable(poly))
                # obj["segmentation"] = [poly]
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts


    @staticmethod
    def create_dataset_json(df, classes, output_json_path, imgs_dir_path=None):
        images = []
        annotations = []
        categories = {}
        img_names = df.filename.unique()
        for img_id, img_name in tqdm(enumerate(img_names), total=len(img_names)):
            # print("Procesando: {}".format(anno_path))
            img_df = df[df.filename == img_name]
            file_path = f'{imgs_dir_path}/{img_name}' if imgs_dir_path is not None else img_name

            images.append({
                "file_name": file_path,
                "height": int(img_df.iloc[0].ytotal),
                "width": int(img_df.iloc[0].xtotal),
                "id": img_id
            })
            for _, row in img_df.iterrows():
                class_id = classes.index(row.label) + 1
                xmin = int(row.xmin)
                ymin = int(row.ymin)
                xmax = int(row.xmax)
                ymax = int(row.ymax)

                annotation_id = len(annotations)+1
                annotations.append({
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": [
                        xmin,
                        ymin,
                        xmax-xmin,
                        ymax-ymin
                    ],
                    "area": (xmax-xmin)*(ymax-ymin),
                    "segmentation": [],
                    "category_id": class_id,
                    "id": annotation_id
                })

        categories = [{"id": i+1, "name": v} for i, v in enumerate(classes)]

        with open(output_json_path, 'w') as f:
            json.dump({'images': images, 'annotations': annotations, 'categories': categories}, f, indent=4)


class CocoTrainer(DefaultTrainer):


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
