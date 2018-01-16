import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

x_file = "../resources/x.csv"
y_file = "../resources/y.csv"
set_label_file = "../resources/set_label.csv"

k_folds = 5
test_size = 1 / k_folds
train_size = 1 - test_size


if __name__ == "__main__":
    tol_x = pd.read_csv(x_file, encoding='gbk').as_matrix()
    tol_y = pd.read_csv(y_file, encoding='gbk').as_matrix()
    set_label = pd.read_csv(set_label_file, encoding='gbk')
    set_label = set_label.iloc[:, 0]

    tol_x = tol_x[set_label == "UA"]
    tol_y = tol_y[set_label == "UA"]

    sss = StratifiedShuffleSplit(k_folds, test_size, train_size)
    for train_index, test_index in sss.split(tol_x, tol_y):
        train_x = tol_x[train_index]
        train_y = tol_y[train_index]

        test_x = tol_x[test_index]
        test_y = tol_y[test_index]

        print('......')
