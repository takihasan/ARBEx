#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch


def precision_single_class(y_true, y_pred, cls):
    """
    Precision over a single class

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        cls: int
            class over which to calculate precision
    """

    y_true = (y_true == cls).astype(int)  # single class true labels
    y_pred = (y_pred == cls).astype(int)  # single class predicted labels

    true_pos = y_true * y_pred  # true positives
    false_pos = y_pred - true_pos  # false positives

    true_pos = np.sum(true_pos)
    false_pos = np.sum(false_pos)

    if true_pos + false_pos == 0.0:
        return 0.0

    return true_pos / (true_pos + false_pos)


def recall_single_class(y_true, y_pred, cls):
    """
    Recall over a single class

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        cls: int
            class over which to calculate precision
    """

    y_true = (y_true == cls).astype(int)  # single class true labels
    y_pred = (y_pred == cls).astype(int)  # single class predicted labels

    true_pos = y_true * y_pred  # true positives
    false_neg = (1 - y_pred) * y_true  # false negatives

    true_pos = np.sum(true_pos)
    false_neg = np.sum(false_neg)

    if true_pos + false_neg == 0.0:
        return 0.0

    return true_pos / (true_pos + false_neg)


def f1_single_class(y_true, y_pred, cls):
    """
    F1 over a single class

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        cls: int
            class over which to calculate precision
    """
    precision = precision_single_class(y_true, y_pred, cls)
    recall = recall_single_class(y_true, y_pred, cls)

    if precision + recall == 0.0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def accuracy_single_class(y_true, y_pred, cls):
    """
    Accuracy over a single class

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        cls: int
            class over which to calculate precision
    """
    y_true = (y_true == cls)  # single class true labels
    y_pred = (y_pred == cls)  # single class predicted labels
    acc = (y_true == y_pred).mean()
    return acc


def accuracy(y_true, y_pred):
    """
    Accuracy

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
    """
    acc = (y_true == y_pred).mean()
    return acc


def f1_average(y_true, y_pred, n_classes):
    """
    F1 average over all classes

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        n_classes: int
            total number of classes
    """

    f1_scores = [f1_single_class(y_true, y_pred, i) for i in range(n_classes)]
    f1_scores = np.array(f1_scores)
    return np.sum(f1_scores) / len(f1_scores)


def precision_average(y_true, y_pred, n_classes):
    """
    Precision average over all classes

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        n_classes: int
            total number of classes
    """

    prec_scores = [precision_single_class(y_true, y_pred, i) for i in range(n_classes)]
    return sum(prec_scores) / len(prec_scores)


def recall_average(y_true, y_pred, n_classes):
    """
    Recall average over all classes

    Args:
        y_true: array
            array of true labels
        y_pred: array
            array of predicted labels
        n_classes: int
            total number of classes
    """

    recall_scores = [recall_single_class(y_true, y_pred, i) for i in range(n_classes)]
    return sum(recall_scores) / len(recall_scores)


class Meter():
    def __init__(self, factor_ema=0.99, n_classes=8):
        self.factor_ema = 0.99
        self.n_classes = n_classes

        self.labels_true = []
        self.labels_pred = []

        self.metrics = {'accuracy', 'f1', 'precision', 'recall'}

        self.getter = {}
        self.getter['accuracy'] = lambda x, y: accuracy(x, y)
        self.getter['f1'] = lambda x, y: f1_average(x, y, self.n_classes)
        self.getter['precision'] = lambda x, y: precision_average(x, y, self.n_classes)
        self.getter['recall'] = lambda x, y: recall_average(x, y, self.n_classes)

        self.all = {}
        self.ema = {}
        self.total = {}
        for m in self.metrics:
            self.all[m] = []
            self.ema[m] = 0.0
            self.total[m] = 0.0

        self.log_ema = {}
        self.log_all = {}

    def add(self, labels_pred, labels_true):
        """
        add new measurements
        """
        if isinstance(labels_pred, torch.Tensor):
            labels_pred = labels_pred.cpu().numpy()
        if isinstance(labels_true, torch.Tensor):
            labels_true = labels_true.cpu().numpy()

        #
        self.labels_true.append(labels_true)
        self.labels_pred.append(labels_pred)

        # get metrics for batch
        for m in self.metrics:
            self.all[m].append(self.getter[m](labels_true, labels_pred))
            self.ema[m] = self.factor_ema * self.ema[m] + (1 - self.factor_ema) * self.all[m][-1]

    def get_metrics_total(self):
        labels_true, labels_pred = self._concat()  # turn list into array
        for m in self.metrics:
            self.total[m] = self.getter[m](labels_true, labels_pred)
        return self.total

    def get_metrics_ema(self):
        return self.ema

    def get_metrics_last(self):
        return {key: val[-1] for key, val in self.all.items()}

    def get_metrics_all(self):
        return self.all

    def _concat(self):
        if isinstance(self.labels_true, list):
            labels_true = np.concatenate(self.labels_true)
        if isinstance(self.labels_pred, list):
            labels_pred = np.concatenate(self.labels_pred)
        self.labels_true = []
        self.labels_pred = []
        return labels_true, labels_pred

    def log(self, name, val):
        if name not in self.log_ema:
            self.log_ema[name] = val
            self.log_all[name] = [val]
        else:
            self.log_ema[name] = self.log_ema[name] * self.factor_ema + \
                    (1 - self.factor_ema) * val
            self.log_all[name].append(val)

    def get_log_last(self):
        return {key: val[-1] for key, val in self.log_all.items()}

    def get_log_ema(self):
        return self.log_ema

    def get_log_all(self):
        return self.log_all

    def get_message(self, loss=False):
        msg = ""
        metrics_last = self.get_metrics_last()
        metrics_ema = self.get_metrics_ema()
        # accuracy
        acc_iter = metrics_last['accuracy'] * 100
        acc_ema = metrics_ema['accuracy'] * 100
        msg += f'[ACC_ITER: {acc_iter:.2f}]'
        msg += f'[ACC_EMA: {acc_ema:.2f}]'
        # f1
        f1_iter = metrics_last['f1'] * 100
        f1_ema = metrics_ema['f1'] * 100
        msg += f'[F1_ITER: {f1_iter:.2f}]'
        msg += f'[F1_EMA: {f1_ema:.2f}]'

        if loss:
            log_last = self.get_log_last()
            log_ema = self.get_log_ema()
            # loss ce
            loss_ce_iter = log_last['loss_ce']
            loss_ce_ema = log_ema['loss_ce']
            msg += f'[LOSS_CE_ITER: {loss_ce_iter:.2f}]'
            msg += f'[LOSS_CE_EMA: {loss_ce_ema:.2f}]'
            # loss mu
            loss_mu_iter = log_last['loss_mu']
            loss_mu_ema = log_ema['loss_mu']
            msg += f'[LOSS_MU_ITER: {loss_mu_iter:.2f}]'
            msg += f'[LOSS_MU_EMA: {loss_mu_ema:.2f}]'
            # loss center
            loss_center_iter = log_last['loss_center']
            loss_center_ema = log_ema['loss_center']
            msg += f'[LOSS_CENTER_ITER: {loss_center_iter:.2f}]'
            msg += f'[LOSS_CENTER_EMA: {loss_center_ema:.2f}]'

        return msg


if __name__ == '__main__':
    m = Meter(n_classes=2)
    a = np.array([1, 1, 1])
    b = np.array([1, 1, 0])
    m.add(a, b)
    m.add(a, b)
    m.add(a, b)
    m.add(a, b)
    m.add(a, a)
    print(m.get_metrics_last())
    print(m.get_metrics_ema())
    print(m.get_metrics_total())
