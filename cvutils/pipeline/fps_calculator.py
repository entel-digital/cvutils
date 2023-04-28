from tupuedes.pipeline import Pipeline
from tupuedes.live.utils import CountsPerSec


class FPSCalculator(Pipeline):
    def __init__(self):
        self.cps = CountsPerSec().start()
        super().__init__()

    def map(self, data):
        frame_rate = self.cps.get_fps()
        data['fps'] = frame_rate
        self.cps.increment()

        return data
