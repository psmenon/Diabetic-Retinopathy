# Obtained from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/classification.py

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def cohen_kappa_score(log_preds, targs, labels=None, weights='quadratic', sample_weight=None):
    """
    Parameters
    ----------
    y1 : array, shape = [n_samples]
        Labels assigned by the first annotator.
    y2 : array, shape = [n_samples]
        Labels assigned by the second annotator. The kappa statistic is
        symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to select a
        subset of labels. If None, all labels that appear at least once in
        ``y1`` or ``y2`` are used.
    weights : str, optional
        List of weighting type to calculate the score. None means no weighted;
        "linear" means linear weighted; "quadratic" means quadratic weighted.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    kappa : float
    
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    
    """
    preds = torch.exp(log_preds)
    preds = torch.max(preds, dim=1)[1]
    
    confusion = confusion_matrix(preds, targs, labels=labels,
                                 sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if  weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k