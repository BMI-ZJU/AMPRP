from functools import reduce

import tensorflow as tf
import numpy as np

from models.util import concat_set
from models.dae import xavier_init
from models.sdae import SDAE


class MPRP(object):
    """
    Simple implementation of multi-problem risk prediction for ACS
    """
    def __init__(self, n_input, hiddens, n_class, data_set_name, transfer=tf.nn.sigmoid, corrupt_level=0.3,
                 optimizer=tf.train.AdamOptimizer(), epochs=1000, pre_train=True, name="mprp", sess=None):
        """implementation of `multi-problem risk predictor`

        All the hyper parameters that can be adjusted are `{hiddens, transfer, optimizer, epochs, corrupt_level,
        pre_train}`. n_input and n_class are not adjustable because these are determined by the sample data and
        the problem you are trying to solve

        :param n_input: number of input
        :param hiddens: list of the number of units in each hidden layer
        :param n_class: the number of classes to predict
        :param data_set_name: list of , for example : ``["UA", "SA", "MI"]``
        :param transfer: transfer function
        :param corrupt_level: the ratio of input to corrupt when pre_train
        :param optimizer: tf.nn.optimizer
        :param epochs: training epochs
        :param pre_train: whether to choose pre-training
        :param name: variable scope name
        :param sess: tf.Session()
        """
        self.n_input = n_input
        self.hiddens = hiddens
        self.n_class = n_class
        self.data_set_name = data_set_name
        self.set_num = len(self.data_set_name)
        self.corrupt_level = corrupt_level
        self.epochs = epochs
        self._pre_train = pre_train
        self.name = name
        with tf.variable_scope(self.name):
            self.transfer_func = transfer
            self.optimizer = optimizer

            self.sess = sess if sess is not None else tf.Session()

            self.sdaes = self._init_sdaes()

            self.inputs = dict()
            self.hidden_reps = dict()
            self.weights = dict()
            self.biases = dict()
            self.outputs = dict()
            self.prediction = dict()  # model predictionion
            self.y_ = dict()       # real label
            self.losses = dict()
            for _name in self.data_set_name:
                self.inputs[_name] = tf.placeholder(tf.float32, [None, n_input], name="{}_input".format(_name))
                self.hidden_reps[_name] = tf.concat((self.sdaes[_name](self.inputs[_name]),
                                                     self.sdaes["share"](self.inputs[_name])),
                                                    axis=1)
                self.weights[_name] = tf.Variable(xavier_init(2*hiddens[-1], n_class),
                                                  dtype=tf.float32,
                                                  name="{}_out_weight".format(_name))
                self.biases[_name] = tf.Variable(tf.zeros(self.n_class), name="{}_out_bias".format(_name))
                self.outputs[_name] = self.hidden_reps[_name] @ self.weights[_name] + self.biases[_name]
                self.prediction[_name] = tf.nn.softmax(self.outputs[_name])
                self.y_[_name] = tf.placeholder(tf.float32, [None, n_class], name="{}_label".format(_name))
                self.losses[_name] = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_[_name], self.outputs[_name]))

            # combine all the losses
            g_vars = [v for _name in self.data_set_name for v in self.sdaes[_name].vars]
            g_vars[len(g_vars):] = [self.weights[_name] for _name in self.data_set_name]
            g_vars[len(g_vars):] = [self.biases[_name] for _name in self.data_set_name]
            g_vars[len(g_vars):] = [v for v in self.sdaes['share'].vars]

            self.tol_loss = reduce(lambda x, y: x + y, [self.losses[k] for k in self.data_set_name])
            self.train_op = self.optimizer.minimize(self.tol_loss, var_list=g_vars)

        # init
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _init_sdaes(self):
        sdaes = dict()
        sdaes['share'] = SDAE(self.n_input,
                              self.hiddens,
                              transfer=self.transfer_func,
                              corrupt_level=self.corrupt_level,
                              optimizer=self.optimizer,
                              epochs=self.epochs,
                              name="share_sdae",
                              sess=self.sess)

        for name in self.data_set_name:
            sdaes[name] = SDAE(self.n_input,
                               self.hiddens,
                               transfer=self.transfer_func,
                               corrupt_level=self.corrupt_level,
                               optimizer=self.optimizer,
                               epochs=self.epochs,
                               name="{}_sdae".format(name),
                               sess=self.sess)
        return sdaes

    def pre_train_op(self, data_sets, batch_size):
        """at first, pre_train the stacked denoising autoencoders

        :param data_sets: a dict of data_set, for example ``{'UA': set1, 'SA': set2, 'MI': set3}``, and the name of key
                            is best to be uppercase
        :param batch_size: `batch size` of data
        """
        for key, value in data_sets.items():
            self.sdaes[key].pre_train(value, batch_size)

        new_set = concat_set(data_sets)
        self.sdaes['share'].pre_train(new_set, batch_size)

    def train_process(self, data_sets, batch_size=128):
        if self._pre_train:
            self.pre_train_op(data_sets, batch_size)
            for key, value in data_sets.items():
                value.epoch_completed = 0

        # while epochs less than the epoch_complete of the data_set which has most examples, continue training the model
        name = self.data_set_name[0]
        max_n = -1
        for set_name in self.data_set_name:
            if data_sets[set_name].num_examples > max_n:
                max_n = data_sets[set_name].num_examples
                name = set_name

        while data_sets[name].epoch_completed < self.epochs:
            example_s = [data_sets[_name].next_batch(batch_size) for _name in self.data_set_name]
            example_dict = {}
            label_dict = {}
            for i in range(self.set_num):
                example_dict[self.inputs[self.data_set_name[i]]] = example_s[i][0]
                label_dict[self.y_[self.data_set_name[i]]] = example_s[i][1]
            feed_dict = {**example_dict, **label_dict}
            loss, _ = self.sess.run((self.tol_loss, self.train_op), feed_dict=feed_dict)

    def predict(self, x, **kwargs):
        """make prediction for examples

        :param x: if you give the set_name ,x represents examples from a certain data set
                    else the x should be given in dict() format ``{"UA": set1, "SA": set2, "MI": set3}``
        :param kwargs: receive the 'set_name' parameter
        :return prediction: the prediction of given examples
        """
        if "set_name" in kwargs:
            set_name = kwargs['set_name']
            if set_name not in self.data_set_name:
                raise KeyError("This model can not make prediction for example from the {} data set".format(set_name))
            return self.sess.run(self.prediction[set_name], feed_dict={self.inputs[set_name]: x})

        else:
            pred = tuple([self.sess.run(self.prediction[name], feed_dict={self.inputs[name]: x[name].examples})
                          for name in self.data_set_name])
            return np.vstack(pred)
