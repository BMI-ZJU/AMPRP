class DataSet(object):
	def __init__(x, y):
		self._x = x
		self._y = y
		self._num_examples = self._x.shape[0]
		self._epoch_completed = 0

	def next_batch(batch_size=128):
        if batch_size > self.num_examples:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))

        start = self._index_in_epoch
        if start + batch_size > self.num_examples:
            self._epoch_completed += 1
            rest_num_examples = self._num_examples - start
            x_rest_part = self._x[start:self._num_examples]
            y_rest_part = self._y[start:self._num_examples]

            self._shuffle()  # 打乱
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self._x[start:end]
            y_new_part = self._y[start:end]
            return (np.concatenate((x_rest_part, y_part), axis=0),
                    np.concatenate((x_rest_part, y_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._x[start:end], self._y[start:end]

    @property
    def num_examples(self):
    	return self._num_examples

    @property
    def examples(self):
    	return self._x
    
    @property
    def labels(self):
    	return self._y

    @property
    def epoch_completed(self):
    	return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
    	self._epoch_completed = value
    
