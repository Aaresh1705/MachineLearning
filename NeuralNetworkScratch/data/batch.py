import numpy as np


class Batch:
    def __init__(self, data, label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / batch_size))

    def __iter__(self):
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_data = self.data[start_idx:end_idx]
            batch_label = self.label[start_idx:end_idx]

            yield batch_data, batch_label

    def __len__(self):
        return self.num_batches

    def get_data(self):
        return self.data, self.label
