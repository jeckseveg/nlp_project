import pandas as pd
import nump as np
from torch.utils.data import DataLoader, IterableDataset


class ChunkDataset(IterableDataset):
    def __init__(self, file_paths):
        super(ChunkDataset).__init__()
        self.file_paths = file_paths
        self.dataset_size = 0
        for file_path in file_paths:
            self.dataset_size += len(pd.read_parquet(file_path))

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return ChunkDatasetIterator(self.file_paths)
        else:
            return ChunkDatasetIterator(
                [elem for ind, elem in enumerate(self.file_paths) if (ind % worker_info.num_workers) == worker_info.id])
    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        inputs = data['text']
        labels = data['label']
        return (inputs, labels)
