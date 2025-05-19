import time
import cv2
from collections import deque

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):

        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.frame_width = int(self.stream.get(3))
        self.frame_height = int(self.stream.get(4))
        self.stopped = False

    def start(self):
        from threading import Thread

        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

class VideoWritte:
    def __init__(self, filename, size, fps = 15, codec_str = 'mp4v') -> None:
        self.writter = cv2.VideoWriter(filename,
                                cv2.VideoWriter_fourcc(*codec_str),
                                fps, size)

class CountsPerSec:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.frame_times = deque(maxlen=buffer_size)

    def start(self):
        self.frame_times.clear()
        self.frame_times.append(time.time())
        return self

    def increment(self):
        self.frame_times.append(time.time())

    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0.0
        elapsed_time = self.frame_times[-1] - self.frame_times[0]
        if elapsed_time == 0:
            return 0.0
        # El número de intervalos es uno menos que el número de tiempos registrados
        fps = (len(self.frame_times) - 1) / elapsed_time
        return fps


class FamesInferences:
    def __init__(self, columns: list):
        self.columns = columns
    def store_row(row_dict: dict):
        pass
    def to_df(self):
        pass
