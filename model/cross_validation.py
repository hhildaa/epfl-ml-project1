import numpy as np

from scripts.proj1_helpers import *
from dataprocess.process import *
from scripts.implementations import *


def cross_validation(y, X, k_indices, k, degree, gamma, lambda_, max_iters, batch_size):
    """return the loss of ridge regression."""
    
    X, _ = remove_tiny_features(X)
    
    X_test = X[k_indices[k]]
    y_test = y[k_indices[k]]
    X_train = np.vstack([X[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])
    y_train = np.hstack([y[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])
    
    # remove outliers
    # impute values when they are missing (value -999 I think) (e.g. by median)
    
    X_test = build_poly(X_test, degree)
    X_train = build_poly(X_train, degree)
     
    X_train = standardize(X_train)
    X_test = standardize(X_test)
    
    w0 = np.zeros(X_train.shape[1])
    w, loss = reg_logistic_regression(y=y_train, tx=X_train, lambda_=lambda_, initial_w=w0, max_iters=max_iters, gamma=gamma, batch_size=batch_size)
    
    y_train_pred = predict_labels(w, X_train)
    y_test_pred = predict_labels(w, X_test)
    
    acc_train = accuracy(y_train_pred, y_train)
    acc_test = accuracy(y_test_pred, y_test)
    return acc_train, acc_test