import math

from torch.utils import data

from tools_data import loadJSON


class DayDataset(data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, filepath):
        super(DayDataset).__init__()
        self.data = loadJSON(filepath)
        self.fields = [
            ('text', data.Field()),
            ('name', data.Field())
        ]

    def __getitem__(self, index):
        return self.data[index]['text']

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return data.ConcatDataset([self, other])


class DayDatasetIterable(data.IterableDataset):
    def __init__(self, start, end):
        super(DayDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
