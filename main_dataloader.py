import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class JSONDataset(IterableDataset):
    def __init__(self, file_path, chunkSize=1000):
        self.file_path = file_path
        self.chunksize = chunkSize

    def __iter__(self):
        reader = pd.read_json(self.file_path, lines=True, chunksize=self.chunksize)
        for chunk in reader:
            yield (np.asarray(chunk['text'].values), np.asarray(chunk['label'].values))


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass