import tensorflow as tf
import numpy as np
import pickle
from abc import abstractmethod

class BaseDataset():

    def __init__(self, name, path, training):
        self.training = training
        self.name = name
        self.path = path
        self.data = []
        

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        val = self.data[index]

        try:
            img = imread(val) if isinstance(val, str) else val
        except:
            img = None
        return img

    def generator(self, batch_size):
        while True:
            start = 0
            total = len(self)

            while start < total:
                end = np.min([start + batch_size, total])
                items = []

                for ix in range(start, end):
                    item = self[ix]
                    if item is not None:
                        items.append(item)

                start = end
                yield np.array(items)

    @property
    def data(self):
        if len(self._data) == 0:
            self._data = self.load()
            np.random.shuffle(self._data)

        return self._data

    @abstractmethod
    def load(self):
        return []



class Cifar10Dataset(BaseDataset):
    def __init__(self, path, training=True):
        super(Cifar10Dataset, self).__init__('Cifar_10', path, training)

    def load(self):
        data = []
        if self.training:
            for i in range(1, 6):
                filename = '{}/data_batch_{}'.format(self.path, i)
                with open(file, 'rb') as fo:
                    batch_data = pickle.load(fo, encoding='bytes')

                if len(data) > 0:
                    data = np.vstack((data, batch_data[b'data']))
                else:
                    data = batch_data[b'data']

        else:
            filename = '{}/test_batch'.format(self.path)
            batch_data = unpickle(filename)
            data = batch_data[b'data']

        w = 32
        h = 32
        s = w * h
        data = np.array(data)
        data = np.dstack((data[:, :s], data[:, s:2 * s], data[:, 2 * s:]))
        data = data.reshape((-1, w, h, 3))
        return data
