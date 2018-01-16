# -*- coding: utf-8 -*-
import csv

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve

from data.read_data import DataSet
from models.amprp import AMPRP
from models.base_line import MLP, LR
from models.mprp import MPRP

x_file = "../resources/x.csv"
y_file = "../resources/y.csv"
set_label_file = "../resources/set_label.csv"

k_folds = 5
test_size = 1 / k_folds
train_size = 1 - test_size
batch_size = 64
data_set_name = ['MI', 'UA', 'SA']
random_state = 1


def evaluate(tol_label, tol_pred):
    y_true = np.argmax(tol_label, axis=1)
    y_score = tol_pred[:, 1]
    y_pred = np.argmax(tol_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    with open('../result/AMPRP_roc.csv', 'a', newline='') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',')
        f_writer.writerow(fpr)
        f_writer.writerow(tpr)
        f_writer.writerow([])
    return accuracy, auc, precision, recall, f_score


# Baseline method, i.e. sdae nets
def base_line(hiddens):
    tol_x = pd.read_csv(x_file, encoding='gbk').as_matrix()
    tol_y = pd.read_csv(y_file, encoding='gbk').as_matrix()
    set_label = pd.read_csv(set_label_file, encoding='gbk')
    set_label = set_label.iloc[:, 0]

    tol_x = tol_x[set_label == "MI"]
    tol_y = tol_y[set_label == "MI"]

    n_input = tol_x.shape[1]
    n_class = 2

    tol_pred = np.zeros(shape=(0, 2))
    tol_label = np.zeros(shape=(0, 2))

    sss = StratifiedShuffleSplit(k_folds, test_size, train_size, random_state=1)
    for train_index, test_index in sss.split(tol_x, tol_y):
        train_x = tol_x[train_index]
        train_y = tol_y[train_index]

        train_data_set = DataSet(train_x, train_y)

        test_x = tol_x[test_index]
        test_y = tol_y[test_index]

        model = MLP(n_input, hiddens, n_class, pre_train=False, transfer=tf.nn.relu)
        # model = LR(n_input, n_class)
        model.train_process(train_data_set, batch_size)

        pred = model.predict(test_x)
        tol_pred = np.vstack((tol_pred, pred))
        tol_label = np.vstack((tol_label, test_y))

        del model

    y_true = np.argmax(tol_label, axis=1)
    y_score = tol_pred[:, 1]
    y_true1 = np.argmin(tol_label, axis=1)
    y_score1 = tol_pred[:, 0]
    y_pred = np.argmax(tol_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)

    with open('../result/MLP_roc.csv', 'a', newline='') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',')
        f_writer.writerow(fpr)
        f_writer.writerow(tpr)
        f_writer.writerow([])

    print('acc={}, auc={}, precision={}, recall={}, f_score={}'.format(accuracy, auc, precision, recall, f_score))
    np.savetxt('../resources/LR_result.csv', np.hstack((y_true, y_score)), delimiter=',')
    del tol_x, tol_y, tol_label, tol_pred


def main(omega):
    tol_x = pd.read_csv(x_file, encoding='gbk')
    tol_y = pd.read_csv(y_file, encoding='gbk')
    set_label = pd.read_csv(set_label_file, encoding='gbk')
    set_label = set_label.iloc[:, 0]

    MI_x = tol_x[set_label == 'MI'].as_matrix()
    MI_y = tol_y[set_label == 'MI'].as_matrix()
    UA_x = tol_x[set_label == 'UA'].as_matrix()
    UA_y = tol_y[set_label == 'UA'].as_matrix()
    SA_x = tol_x[set_label == 'SA'].as_matrix()
    SA_y = tol_y[set_label == 'SA'].as_matrix()

    MI_split = StratifiedShuffleSplit(k_folds, test_size, train_size, random_state).split(MI_x, MI_y)
    UA_split = StratifiedShuffleSplit(k_folds, test_size, train_size, random_state).split(UA_x, UA_y)
    SA_split = StratifiedShuffleSplit(k_folds, test_size, train_size, random_state).split(SA_x, SA_y)

    n_input = MI_x.shape[1]
    hiddens = [200, 200]
    n_class = 2

    tol_pred = np.zeros(shape=(0, 2))
    tol_label = np.zeros(shape=(0, 2))
    MI_tol_pred = np.zeros(shape=(0, 2))
    MI_tol_label = np.zeros(shape=(0, 2))
    UA_tol_pred = np.zeros(shape=(0, 2))
    UA_tol_label = np.zeros(shape=(0, 2))
    SA_tol_pred = np.zeros(shape=(0, 2))
    SA_tol_label = np.zeros(shape=(0, 2))

    for ith_fold in range(k_folds):
        print('{} th fold of {} folds'.format(ith_fold, k_folds))
        MI_train_index, MI_test_index = next(MI_split)
        UA_train_index, UA_test_index = next(UA_split)
        SA_train_index, SA_test_index = next(SA_split)

        train_sets = {'MI': DataSet(MI_x[MI_train_index], MI_y[MI_train_index]),
                      'UA': DataSet(UA_x[UA_train_index], UA_y[UA_train_index]),
                      'SA': DataSet(SA_x[SA_train_index], SA_y[SA_train_index])}

        test_sets = {'MI': DataSet(MI_x[MI_test_index], MI_y[MI_test_index]),
                     'UA': DataSet(UA_x[UA_test_index], UA_y[UA_test_index]),
                     'SA': DataSet(SA_x[SA_test_index], SA_y[SA_test_index])}

        model = AMPRP(n_input, hiddens, n_class, data_set_name, transfer=tf.nn.sigmoid, omega=omega, pre_train=True, epochs=1000)
        # model = MPRP(n_input, hiddens, n_class, data_set_name, transfer=tf.nn.sigmoid, pre_train=True, epochs=1000)
        model.train_process(train_sets, batch_size=batch_size)

        pred = model.predict(test_sets)

        MI_pred = model.predict(MI_x[MI_test_index], set_name='MI')
        MI_tol_pred = np.vstack((MI_tol_pred, MI_pred))
        UA_pred = model.predict(UA_x[UA_test_index], set_name='UA')
        UA_tol_pred = np.vstack((UA_tol_pred, UA_pred))
        SA_pred = model.predict(SA_x[SA_test_index], set_name='SA')
        SA_tol_pred = np.vstack((SA_tol_pred, SA_pred))

        MI_tol_label = np.vstack((MI_tol_label, MI_y[MI_test_index]))
        UA_tol_label = np.vstack((UA_tol_label, UA_y[UA_test_index]))
        SA_tol_label = np.vstack((SA_tol_label, SA_y[SA_test_index]))

        del model

    MI_eval = evaluate(MI_tol_label, MI_tol_pred)
    UA_eval = evaluate(UA_tol_label, UA_tol_pred)
    SA_eval = evaluate(SA_tol_label, SA_tol_pred)
    tol_label = np.vstack((MI_tol_label, UA_tol_label, SA_tol_label))
    tol_pred = np.vstack((MI_tol_pred, UA_tol_pred, SA_tol_pred))
    y_true = np.argmax(tol_label, axis=1)
    y_score = tol_pred[:, 1]
    np.savetxt('../resources/pred_result.csv', np.vstack((y_true, y_score)), delimiter=',')
    avg_eval = evaluate(tol_label, tol_pred)
    print(avg_eval)

    with open('../resources/MPRP_eval.csv', 'a', newline='') as csvfile:
        csvfile.write('Result of MPRP, omega={}, hiddens={} \n'.format(omega, hiddens))
        f_writer = csv.writer(csvfile, delimiter=',')
        f_writer.writerow(MI_eval)
        f_writer.writerow(UA_eval)
        f_writer.writerow(SA_eval)
        f_writer.writerow(avg_eval)

    del tol_x, tol_y, tol_pred, tol_label
    del MI_x, MI_y, UA_x, UA_y, SA_x, SA_y
    del MI_tol_pred, MI_tol_label, UA_tol_pred, UA_tol_label, SA_tol_label, SA_tol_pred


if __name__ == "__main__":
    main(0.1)
