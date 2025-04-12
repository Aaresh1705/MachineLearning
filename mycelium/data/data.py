import numpy as np

class Split:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __bool__(self):
        return False

class RandomSplit(Split):
    def __init__(self, X, y, test_split):
        super(RandomSplit, self).__init__(X, y)

        indices = np.arange(X.shape[0])
        test_size = int(X.shape[0] * test_split)

        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        self.train = Split(X[train_idx], y[train_idx])
        self.test = Split(X[test_idx], y[test_idx])

    def __bool__(self):
        return True


class Data:
    def __init__(self, data, label, *, test_split=0.2, split_type=None, shuffle=True):
        if shuffle:
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)

            data = data[indices]
            label = label[indices]

        if split_type and 1 > test_split > 0:
            split = split_type(data, label, test_split)
            self.train = split.train
            self.test = split.test
        else:
            self.train = Split(data, label)
            self.test = Split(np.array([]), np.array([]))



if __name__ == '__main__':
    from sin_data import sin_data
    X, y = sin_data()

    data = Data(X, y, split_type=RandomSplit)

    print((data.train.X[0]))
    print((X[0]))
