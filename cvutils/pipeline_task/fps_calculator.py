from cvutils.pipeline_task.pipeline_task import PipelineTask
from cvutils.live.utils import CountsPerSec

class FPSCalculator(PipelineTask):
    def __init__(self, buffer_size=30):
        super().__init__()
        self.cps = CountsPerSec(buffer_size=buffer_size).start()

    def map(self, data_list):
        for data in data_list:
            self.cps.increment()
            frame_rate = self.cps.get_fps()
            data['fps'] = frame_rate
        return data_list