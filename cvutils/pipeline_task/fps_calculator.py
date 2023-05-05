from cvutils.pipeline_task.pipeline_task import PipelineTask
from cvutils.live.utils import CountsPerSec


class FPSCalculator(PipelineTask):
    def __init__(self):
        self.cps = CountsPerSec().start()
        super().__init__()

    def map(self, data):
        frame_rate = self.cps.get_fps()
        data['fps'] = frame_rate
        self.cps.increment()

        return data
