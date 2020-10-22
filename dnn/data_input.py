from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import dtypes


def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


class DataSet(object):
    def __init__(self,
               images,
               labels,
               dtype=dtypes.float16,
               reshape=True):

      """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float16` to rescale into
        `[0, 1]`.
      """
      dtype = dtypes.as_dtype(dtype).base_dtype
      if dtype not in (dtypes.uint8, dtypes.float16):
        raise TypeError('Invalid image dtype %r, expected uint8 or float16' %
                      dtype)
    
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
          assert images.shape[3] == 1
          images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
    
      self._images = images
      self._labels = labels
      self._epochs_completed = 0
      self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
    
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start: end], self._labels[start: end]

    def reset_epochs(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        print('Epochs reset', flush=True)


def load_data(train_file, validation_file, skip_rows=0):
    train_dat = iter_loadtxt(train_file, delimiter='\t', skiprows=skip_rows)
    validation_dat = iter_loadtxt(validation_file,delimiter='\t',
                                  skiprows=skip_rows)
    
    print('Training samples: {}'.format(train_dat[:, 0:-1].shape[0]))
    print('Validation samples: {}'.format(validation_dat[:, 0:-1].shape[0]))
    
    class DataSets(object):
        pass
    
    data_sets=DataSets()
    
    data_sets.train=DataSet(images=train_dat[:, 0:-1],
                            labels=train_dat[:, -1].astype(int),
                            reshape=False)
    
    data_sets.validation=DataSet(images=validation_dat[:,0:-1],
                                 labels=validation_dat[:,-1].astype(int),
                                 reshape=False)
    return data_sets


def load_test(data_file, labels=True, skip_rows=0):
    """
    If no labels are available, e.g, patient RNA-seq data
    """
    dat = iter_loadtxt(data_file, delimiter='\t', skiprows=skip_rows)
    if labels:
        Dat = DataSet(images=dat[:, 0:-1], labels=dat[:, -1].astype(int), reshape=False)
    else:
        nrow = dat.shape[0]
        Dat = DataSet(images=dat, labels=np.full(nrow, np.nan), reshape=False)
    return Dat


def numpy_input(data_file, delimiter='\t', skiprows=0):
    print('Data file: {}'.format(data_file))
    dat = iter_loadtxt(filename=data_file, delimiter=delimiter, skiprows=skiprows)
    x = dat[:, :-1]
    y = dat[:, -1]
    print('X size: {}'.format(x.shape))
    print('y size: {}'.format(y.shape))
    return x, y
