import pandas as pd
from pathlib import Path
from .pipeline import Pipeline


class SaveData(Pipeline):
    """Simple Pipeline task to save annotations to a file using pandas.
       Annotations are passed as an array of rows, inside data['src'] dict
       in 'columns' order
    """

    def __init__(self, src, columns, output_file):
        self.output_file = output_file
        self.src = src
        self.columns = columns
        self.accumulator = []
        super().__init__()

    def map(self, data):
        self.accumulator += data[self.src]
        return data

    def cleanup(self):
        df = pd.DataFrame(data=self.accumulator, columns=self.columns)
        output_path = Path(self.output_file)
        if output_path.suffix == '.feather':
            df.to_feather(self.output_file)
        if output_path.suffix == '.csv':
            df.to_csv(self.output_file)
