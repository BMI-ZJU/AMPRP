import numbers

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.ops import random_ops, math_ops, array_ops


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def mask_noise(x, keep_prop):
    with tf.variable_scope("mask"):
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prop, numbers.Real) and not 0 < keep_prop <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prop)
        keep_prop = ops.convert_to_tensor(keep_prop,
                                          dtype=x.dtype,
                                          name="keep_prob")
        keep_prop.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prop) == 1:
            return x

        random_tensor = keep_prop
        shape = array_ops.shape(x)
        random_tensor += random_ops.random_uniform(shape,
                                                   dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return ret


class DAE(object):
    """
    simple implementation of denosing auto encoder
    """

    def __init__(self, n_visible, n_hidden, transfer=tf.nn.sigmoid, corrupt_level=0.3,
                 optimizer=tf.train.AdamOptimizer(), name="dae", sess=None):
        """
        simple implementation of denoising auto encoder

        :param n_visible: number of units in visible layer
        :param n_hidden: number of units in hidden layer
        :param transfer: transfer function
        :param corrupt_level: thr ratio of corrupt
        :param optimizer: tf.optimizer.Optimizer
        :param sess: tf.Session(), accept the master session to integrate the model
        """
        self.name = name
        with tf.variable_scope(self.name):
            self.n_visible = n_visible
            self.n_hidden = n_hidden
            self.transfer_func = transfer
            self.corrupt = tf.placeholder(tf.float32, name="corrupt_level")
            self.corrupt_level = corrupt_level
            self.weights = self._weights_init()

            self.x = tf.placeholder(tf.float32, [None, self.n_visible], name="input")  # input placeholder
            self.tilde_x = mask_noise(self.x, 1 - self.corrupt)
            self.hidden = self.transfer_func(self.tilde_x @ self.weights['w1'] + self.weights['b1'], name="hidden")
            self.reconstruction = self.hidden @ self.weights['w2'] + self.weights['b2']
            self.rec = self.transfer_func(self.reconstruction, name="reconstruction")

            self.cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.x, self.reconstruction), name="dae_cost")
            self.optimizer = optimizer.minimize(self.cost)

            init = tf.global_variables_initializer()
            self.sess = sess if sess is not None else tf.Session()
            self.sess.run(init)

    def __call__(self, x, corrupt_level=0.3):
        """*without dropout (corrupt)*

        :param x: input tensor
        :param corrupt_level: the ratio of corrupt
        :return: output tensor
        """
        return self.transfer_func(x @ self.weights['w1'] + self.weights['b1'])

    def _weights_init(self):
        with tf.variable_scope("weights"):
            weights = dict()
            weights['w1'] = tf.Variable(xavier_init(self.n_visible, self.n_hidden), dtype=tf.float32, name="w1")
            weights['b1'] = tf.Variable(tf.zeros(self.n_hidden), name="b1")
            weights['w2'] = tf.transpose(weights['w1'], name="w2")
            weights['b2'] = tf.Variable(tf.zeros(self.n_visible), name="b2")
            return weights

    def train_op(self, x):
        """Single training process

        :param x: one batch training examples
        :return: current cost function value of this batch training examples
        """
        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x, self.corrupt: self.corrupt_level})
        return cost

    def cal_total_cost(self, x):
        """calculate the current cost value of all training examples

        :param x: the whole set of training example
        :return: current cost value of all training examples
        """
        return self.sess.run(self.cost, feed_dict={self.x: x, self.corrupt: self.corrupt_level})

    def encode(self, x):
        """
        get the hidden representation

        :param x: data input
        :return: hidden representation
        """
        return self.sess.run(self.hidden, feed_dict={self.x: x, self.corrupt: self.corrupt_level})

    def decode(self, hidden):
        """different with `decode_func`, this method accepts np.ndarrays and return np.ndarrays

        :param hidden: hidden representation
        :return: reconstructed data
        """
        return self.sess.run(self.rec, feed_dict={self.hidden: hidden})

    def decode_func(self, hidden):
        """different with `decode`, this method accepts tensor and return tensor

        :param hidden: hidden tensor
        :return: reconstructed tensor
        """
        return self.transfer_func(hidden @ self.weights['w2'] + self.weights['b2'])

    def reconstruct(self, x):
        """

        :param x: origin data input
        :return: reconstructed data
        """
        return self.sess.run(self.rec, feed_dict={self.x: x, self.corrupt: self.corrupt_level})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])

    @property
    def vars(self):
        return [self.weights['w1'], self.weights['b1']]


if __name__ == "__main__":
    print("denoising auto encoder test")
    visible_n = 10
    hidden_n = 5
    train_x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
    text_x = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    dae = DAE(visible_n, hidden_n)

    for i in range(1000):
        dae.train_op(train_x)

    pred = dae.reconstruct(text_x)
    print(pred)
