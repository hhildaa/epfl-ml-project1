import numpy as np

from scripts.proj1_helpers import *
from dataprocess.process import *
from scripts.implementations import *


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, X, k_indices, k, degree, gamma, lambda_, max_iters, batch_size, algorithm, params):
    """return the loss of ridge regression."""
    
    # Remove degenerated features
    X, _ = remove_tiny_features(X)
    
    X_test = X[k_indices[k]]
    y_test = y[k_indices[k]]
    X_train = np.vstack([X[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])
    y_train = np.hstack([y[k_indices[i]] for i in range(k_indices.shape[0]) if not i == k])
    
    # Imputing median
    if(params['impute_median']):
        X_train, median = impute_median(X_train, None)
        X_test, _ = impute_median(X_test, median)

    # Remove outliers
    if(params['remove_outliers']):
        X_train, upper_quart, lower_quart = bound_outliers(X_train, None, None)
        X_test, _, _ = bound_outliers(X_test, upper_quart, lower_quart)

    # Feature expansion
    if(params['feature_expansion']):
        X_train = build_poly(X_train, degree)
        X_test = build_poly(X_test, degree)

    # Standardize features
    X_train, mean, std = standardize(X_train, None, None)
    X_test, _, _ = standardize(X_test, mean, std)
    


    w0 = np.zeros(X_train.shape[1])
    w, loss = None, None

    if(algorithm == 'reg_logistic'):
        w, loss = reg_logistic_regression(y=y_train, 
                                          tx=X_train, 
                                          lambda_=lambda_, 
                                          initial_w=w0, 
                                          max_iters=max_iters, 
                                          gamma=gamma, 
                                          batch_size=batch_size)

    if(algorithm == 'logistic'):
        w, loss = logistic_regression(y=y_train, 
                                      tx=X_train, 
                                      initial_w=w0, 
                                      max_iters=max_iters, 
                                      gamma=gamma, 
                                      batch_size=batch_size)

    if(algorithm == 'least_squares_GD'):
        w, loss = least_squares_GD(y=y_train, 
                                   tx=X_train,  
                                   initial_w=w0, 
                                   max_iters=max_iters, 
                                   gamma=gamma)


    if(algorithm == 'least_squares_SGD'):
        w, loss = least_squares_SGD(y=y_train, 
                                    tx=X_train,  
                                    initial_w=w0, 
                                    max_iters=max_iters, 
                                    gamma=gamma)
    


    
    y_train_pred = predict_labels(w, X_train)
    y_test_pred = predict_labels(w, X_test)
    
    acc_train = accuracy(y_train_pred, y_train)
    acc_test = accuracy(y_test_pred, y_test)
    return acc_train, acc_test, y_test_pred, y_test



def k_fold_cross_validation(y, X, k_fold, lambdas, degrees, max_iters, batch_size, gamma, algorithm, verbose, params):
        # split data in k fold
        k_indices = build_k_indices(y, k_fold)

        # define lists to store the accuracies of training data and test data
        accs_train = np.zeros((len(degrees), len(lambdas)))
        accs_test = np.zeros((len(degrees), len(lambdas)))

        all_preds = {}
        all_labels = {}
        for id_degree, degree in enumerate(degrees):
            for id_lambda, lambda_ in enumerate(lambdas):
                cur_acc_train = np.zeros(k_fold)
                cur_acc_test = np.zeros(k_fold)

                preds = []
                labels = []
                for k in range(k_fold):
                    acc_train, acc_test, y_test_pred, y_test = cross_validation(y=y, X=X, k_indices=k_indices, k=k, 
                                                                                degree=degree, 
                                                                                gamma=gamma, 
                                                                                lambda_=lambda_, 
                                                                                max_iters=max_iters, 
                                                                                batch_size=batch_size,
                                                                                algorithm=algorithm,
                                                                                params=params)

                    cur_acc_train[k] = acc_train
                    cur_acc_test[k] = acc_test
                    preds.append(y_test_pred)
                    labels.append(y_test)

                all_preds[(degree, lambda_)] = preds
                all_labels[(degree, lambda_)] = labels

                accs_train[id_degree, id_lambda] = cur_acc_train.mean()
                accs_test[id_degree, id_lambda] = cur_acc_test.mean()
                if(verbose):
                    print(f"Degree: {degree:2}, Lambda: {lambda_:6}: Train: {cur_acc_train.mean():.4f} +- {cur_acc_train.std():.4f}, Test: {cur_acc_test.mean():.4f} +- {cur_acc_test.std():.4f}")

        id_degree, id_lambda = np.unravel_index(np.argmax(accs_test), accs_test.shape)
        best_degree, best_lambda = degrees[id_degree], lambdas[id_lambda]
        best_accuracy = np.max(accs_test)

        best_preds = all_preds[(best_degree, best_lambda)]                    
        best_labels = all_labels[(best_degree, best_lambda)]
        return best_accuracy, best_degree, best_lambda, best_preds, best_labels