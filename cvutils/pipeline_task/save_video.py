import cv2
import os

from cvutils.pipeline_task.pipeline_task import PipelineTask


class SaveVideo(PipelineTask):
    """Pipeline task to save video."""

    def __init__(self, src, filename, fps=30, fourcc='MJPG'):
        dirname = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dirname, exist_ok=True)
        self.src = src
        self.filename = filename
        self.fps = fps
        self.writer = None
        self.fourcc = fourcc
        super().__init__()

    def map(self, data):
        if isinstance(data, list):
            for item in data:
                frame = item[self.src]
                if self.writer is None:
                    # Inicializar el writer con el primer frame
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
                    self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
                self.writer.write(frame)
        else:
            frame = data[self.src]
            if self.writer is None:
                # Inicializar el writer con el primer frame
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
                self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
            self.writer.write(frame)
        return data

    def close(self):
        if self.writer:
            self.writer.release()