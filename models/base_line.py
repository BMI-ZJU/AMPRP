import tensorflow as tf

from models.dae import xavier_init
from models.sdae import SDAE


class MLP(object):
    def __init__(self, n_input, hiddens, n_class, transfer=tf.nn.sigmoid, corrupt_level=0.3,
                 optimizer=tf.train.AdamOptimizer(), epochs=1000, pre_train=True, name='mlp', sess=None):
        self.name = name
        self._pre_train = pre_train
        self._epochs = epochs
        with tf.variable_scope(self.name):
            self.sess = sess if sess is not None else tf.Session()
            self.input = tf.placeholder(tf.float32, [None, n_input])
            self.sdae = SDAE(n_input, hiddens, transfer, corrupt_level, optimizer, epochs, name)
            self.hidden_rep = self.sdae(self.input)
            self.y_ = tf.placeholder(tf.float32, [None, n_class])
            self.out_weight = tf.Variable(xavier_init(hiddens[-1], n_class),
                                          dtype=tf.float32, name='out_weight')
            self.out_bias = tf.Variable(tf.zeros(n_class), name='out_bias')
            self.output = self.hidden_rep @ self.out_weight + self.out_bias
            self.prediction = tf.nn.softmax(self.output)

            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_, self.output))

            self.train_op = optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def pre_train(self, x, batch_size):
        self.sdae.pre_train(x, batch_size)

    def train_process(self, data_set, batch_size):
        if self._pre_train:
            self.pre_train(data_set, batch_size)
            data_set.epoch_completed = 0

        while data_set.epoch_completed < self._epochs:
            x, y = data_set.next_batch(batch_size)
            cost, _ = self.sess.run((self.loss, self.train_op), feed_dict={self.input: x, self.y_: y})

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.input: x})


class LR(object):
    def __init__(self, n_input, n_class, epochs=1000, optimizer=tf.train.AdamOptimizer(), sess=None, name='softmax_model'):
        self.n_input = n_input
        self.n_class = n_class
        self.epochs = epochs
        self.optimizer = optimizer
        self.sess = sess if sess is not None else tf.Session()
        self.name = name
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input], name="input")
            self.weight = tf.Variable(xavier_init(self.n_input, self.n_class), dtype=tf.float32, name="weight")
            self.bias = tf.Variable(tf.zeros(self.n_class), dtype=tf.float32, name="bias")
            self.output = self.input @ self.weight + self.bias
            self.prediction = tf.nn.softmax(self.output)
            self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_class], name='label')

            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_, self.output))

            self.train_op = self.optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def train_process(self, data_set, batch_size):
        while data_set.epoch_completed < self.epochs:
            x, y = data_set.next_batch(batch_size)
            loss, _ = self.sess.run((self.loss, self.train_op), feed_dict={self.input: x, self.y_: y})

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.input: x})
