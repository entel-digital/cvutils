from cvutils.pipeline_task.pipeline_task import PipelineTask
from ultralytics import YOLO
import torch

class YOLODetector(PipelineTask):
    def __init__(self, model_path, device='cuda', verbose=False, classes=None):
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.classes = classes
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.model.fuse()

    def map(self, data_list):
        for data in data_list:
            image = data["image"]
            
            result = self.model.predict(
                source=image,
                conf=0.5,
                classes=self.classes,
                verbose=self.verbose
            )[0]
            
            data['image'] = result.plot()
            data['results'] = result.boxes

        return data_list