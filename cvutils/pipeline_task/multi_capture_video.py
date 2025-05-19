import time
import cv2
from threading import Thread
from queue import Queue
from cvutils.pipeline_task.pipeline_task import PipelineTask
from cvutils.capture.webcam_video_capture import WebcamVideoCapture

class MultiCaptureVideo(PipelineTask):
    """Pipeline task to capture multiple video streams."""

    def __init__(self, sources, target_fps=30):
        self.captures = [
            WebcamVideoCapture(
                src=src,
                fps=target_fps, 
                queue_size=20, 
                use_hw_accel=False,
                width=1920,
                height=1080,    
                name=f"Camera_{i}"
            ).start() 
            for i, src in enumerate(sources)
        ]
        self.sources = sources
        self.frame_num = 0

    def generator(self):
        while any([cap.running() for cap in self.captures]):
            data_list = []
            for i, cap in enumerate(self.captures):
                if cap.running():
                    # timeout de 2 segundos
                    image = cap.read(timeout=2.0)

                    # Manejar el caso donde no llega frame a tiempo
                    if image is None:
                        continue

                    data = {
                        "frame_num": self.frame_num,
                        "source_index": i,
                        "image_id": f"{i}_{self.frame_num:06d}",
                        "image": image
                    }
                    data_list.append(data)

            self.frame_num += 1

            if data_list:
                yield data_list


    def cleanup(self):
        for cap in self.captures:
            cap.stop()