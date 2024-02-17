import svetoch.tensor as ten

class TensorDataset:
    def __init__(self, *tensors):
        for t in tensors:
            assert t.shape[0] == tensors[0].shape[0], 'tensors must be of the same shape'
        self.tensors = tensors

    def __getitem__(self, key):
        return tuple(t[key] for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            idxs = ten.randperm(n)
        else:
            idxs = ten.arange(n)
        for i in range(0, n, self.batch_size):
            yield self.dataset[idxs[i : i+self.batch_size].v]

    def __len__(self):
        return len(self.dataset)
