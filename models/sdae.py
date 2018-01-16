import numpy as np
import tensorflow as tf

from data.read_data import DataSet
from models.dae import DAE


class SDAE(object):
    """
    simple implementation of stacked denoising auto encoders

    | Ref paper `Stacked Denoising Autoencoders <http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf>`_
    """
    def __init__(self, n_input, hiddens, transfer=tf.nn.sigmoid, corrupt_level=0.3,
                 optimizer=tf.train.AdamOptimizer(), epochs=1000, name="sdae", sess=None):
        """
        simple implementation of stacked denosing auto encoders

        :param int n_input: number of input
        :param list hiddens: list of the number of units in each hidden layer
        :param transfer: transfer (or activation) function
        :param float corrupt_level: the ratio of corrupted
        :param optimizer: tf.optimizer.Optimizer
        :param int epochs: epoch of training operation
        :param sess: tf.Session(), accept the master session to integrate the model
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_input = n_input
            self.stacks = len(hiddens)
            self.corrupt_level = corrupt_level
            self.epochs = epochs
            self.x = tf.placeholder(tf.float32, [None, n_input], name="input")

            self.transfer_func = transfer
            self.optimizer = optimizer

            self.sess = sess if sess is not None else tf.Session()

            self.daes = self._init_daes(self.n_input, hiddens)

            self.hidden = self.x
            for dae in self.daes:
                self.hidden = dae(self.hidden)

            self.rec = self.hidden
            for dae in reversed(self.daes):
                self.rec = dae.decode_func(self.rec)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def __call__(self, x):
        """
        as a component of parent model

        :param x: input tensor
        :return: hidden representation tensor
        """
        x_copy = x
        for dae in self.daes:
            x_copy = dae(x_copy)
        return x_copy

    def _init_daes(self, n_input, hiddens):
        """init the stacked layers of denoising auto encoders

        :param n_input: num of input
        :param hiddens: list of num of hidden layers
        :return: layers of dae
        """
        daes = []
        for i in range(len(hiddens)):
            if i == 0:
                dae = DAE(n_input, hiddens[i],
                          transfer=self.transfer_func,
                          corrupt_level=self.corrupt_level,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                daes.append(dae)
            else:
                dae = DAE(hiddens[i-1], hiddens[i],
                          transfer=self.transfer_func,
                          corrupt_level=self.corrupt_level,
                          optimizer=self.optimizer,
                          name="dae_{}".format(i),
                          sess=self.sess)
                daes.append(dae)
        return daes

    def pre_train(self, data_set, batch_size=128):
        """pre train the model

        :param data.read_data.DataSet data_set: the training data set
        :param batch_size: `batch size` of data
        """
        for i in range(self.stacks):
            while data_set.epoch_completed < self.epochs:
                x, _ = data_set.next_batch(batch_size)
                self.daes[i].train_op(x)
            x = self.daes[i].encode(data_set.examples)
            data_set = DataSet(x, data_set.labels)

    def encode(self, x):
        """get the hidden representation

        :param x: data input
        :return: hidden representation
        """
        return self.sess.run(self.hidden, feed_dict={self.x: x})

    def reconstruct(self, x):
        """get the reconstructed data

        :param x: data input
        :return: reconstructed data
        """
        return self.sess.run(self.rec, feed_dict={self.x: x})

    @property
    def vars(self):
        return [w for dae in self.daes for w in dae.vars]


if __name__ == "__main__":
    input_n = 20
    hiddens_n = [10, 5]
    sdae = SDAE(input_n, hiddens_n)
    train_x = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]])
    text_x = np.array([[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]])
    data = DataSet(train_x, np.arange(train_x.shape[0]))
    sdae.pre_train(data, 10)

    rec_x = sdae.reconstruct(text_x)
    print(rec_x)
