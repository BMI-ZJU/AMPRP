import numpy as np
from functools import reduce

from data.read_data import DataSet


def vstack(mat1, mat2):
    """stack arrays in sequence vertically

    :param mat1: the first matrix
    :param mat2: the second matrix
    :return: concatenated matrix
    """
    return np.vstack((mat1, mat2))


def concat_set(data_sets):
    new_set = [value for _, value in data_sets.items()]
    new_x = map(lambda x: x.examples, new_set)
    new_x = reduce(vstack, new_x)
    new_y = map(lambda x: x.labels, new_set)
    new_y = reduce(vstack, new_y)
    return DataSet(new_x, new_y)
