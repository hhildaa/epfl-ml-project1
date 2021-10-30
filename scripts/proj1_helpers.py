# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, competition=False):
    """Generates class predictions given weights, and a test data matrix"""
    
    negative_class = 0
    if(competition):
        negative_class = -1
    
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = negative_class
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

def standardize(X):
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-20)
    return X

def build_poly(X, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    d = X.shape[1]
    X_poly = []
    for i in range(0, degree + 1):
        X_poly.append(X ** (i + 1))
        
    X_poly.append(np.ones((X.shape[0], 1)))
    X_poly = np.concatenate(X_poly, axis=1)
    
    
    """
    IT SEEMS THAT CURRENTLY SIN/COS FEATURES HARM PERFORMANCE
    
    # add sin and cos to basis
    X_sin = np.sin(X)
    X_cos = np.cos(X)
    X_poly = np.concatenate((X_poly, X_sin, X_cos), axis=1)
    """
    
    """
    # cross terms of second degree
    X_cross = []
    for i in range(d):
        for j in range(d):
            if i != j:
                X_cross.append((X[:, i] * X[:, j]).reshape(-1, 1))
                
    X_cross = np.concatenate(X_cross, axis=1)
    X_final = np.concatenate((X_poly, X_cross), axis=1)    
    return X_final
    """
    
    return X_poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)

def split_data_by_feature(y, X, ids, feature_id, train=True):
    unique_values = np.unique(X[:, feature_id])
    X_new = np.delete(X, feature_id, axis=1)
    splits = {}
    for value in unique_values:
        X_cur = X_new[np.where(X[:, feature_id] == value)]
        y_cur = None
        if(train):
            y_cur = y[np.where(X[:, feature_id] == value)]
        else:
            y_cur = ids[np.where(X[:, feature_id] == value)]
        ids_cur = ids[np.where(X[:, feature_id] == value)]
        splits[value] = (X_cur, y_cur, ids_cur)
        
    return splits