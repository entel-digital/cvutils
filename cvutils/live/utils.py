import time
import cv2

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
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self.full_counter = 0
        self.tick_counter = 0
        self._tick_secs = 1

    def start(self):
        self._start_time = time.time()
        return self

    def increment(self):
        self.full_counter += 1
        self.tick_counter += 1

    def get_fps(self):
        elapsed_time = time.time() - self._start_time
        fps = self.tick_counter / elapsed_time
        if elapsed_time > self._tick_secs:
            self._start_time = time.time()
            self.tick_counter = 0
        
        return fps


class FamesInferences:
    def __init__(self, columns: list):
        self.columns = columns
    def store_row(row_dict: dict):
        pass
    def to_df(self):
        pass