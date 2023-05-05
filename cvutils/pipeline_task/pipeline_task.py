class PipelineTask(object):
    """Common pipeline_task class fo all pipeline_task tasks."""

    def __init__(self, source=None):
        self.source = source

    def __iter__(self):
        return self.generator()

    def generator(self):
        """Yields the pipeline_task data."""

        while self.has_next():
            try:
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __and__(self, other):
        """Allows to connect the pipeline_task using | operator."""

        if other is not None:
            other.source = self.generator()
            return other
        else:
            return self

    def filter(self, data):
        """Overwrite to filter out the pipeline_task data."""

        return True

    def map(self, data):
        """Overwrite to map the pipeline_task data."""

        return data

    def has_next(self):
        """Overwrite to stop the generator in certain conditions."""

        return True

