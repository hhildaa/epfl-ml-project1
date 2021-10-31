# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np



def load_csv_data(data_path, sub_sample=False):
    """
    Loads data.
    
    Input:
    data_path: path to data
    sub_sample: if True, sample only 50 datapoints instead of whole dataset
    
    returns:
    y: class labels
    input_data: features
    ids: event ids
    """
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
    """
    Generates class predictions.
    
    Input: 
    weights: weights (model)
    data: test data matrix
    competition: if True, negative predictions are -1 instead of 0

    Returns:
    y_pred: predictions for data
    """
    
    negative_class = 0
    if(competition):
        negative_class = -1
    
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = negative_class
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd.

    Input:
    ids: event ids associated with each prediction
    y_pred: predicted class labels
    name: string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def accuracy(y_pred, y_true):
    """Calculate accuracy between y_true and y_pred"""
    return np.sum(y_pred == y_true) / len(y_true)

def standardize(X, mean=None, std=None):
    """
    Standardizes values in X

    Input:
    X: data
    mean: custom means for each column, use means of X if None
    std: custom stds for each column, use stds of X if None
    """
    if mean is None:
        mean = np.mean(X, axis=0)

    if std is None:
        std = np.std(X, axis=0)
    
    X = (X - mean) / (std + 1e-20)
    return X, mean, std

def build_poly(X, degree):
    """
    Calculate polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Input:
    X: data
    degree: max degree to calculate polynomial for

    Returns:
    X_poly: X with polynomial features
    """
    
    d = X.shape[1]
    X_poly = []
    for i in range(1, degree + 1):
        X_poly.append(X ** i)
        
    X_poly = np.concatenate(X_poly, axis=1)
    

    # add sin and cos to basis
    X_sin = np.sin(X)
    X_cos = np.cos(X)
    X_poly = np.concatenate((X_poly, X_sin, X_cos), axis=1)

    return X_poly


def split_data_by_feature(y, X, ids, feature_id, train=True):
    """
    split data on feature given in feature_id (assuming categorical)

    Input:
    y: class labels
    X: features
    ids: ids of columns in X
    feature_id: id of feature to split on
    train: if False, do not add y to split

    Returns:
    splits: split of y, X and ids on feature_id
    """
    unique_values = np.unique(X[:, feature_id])
    X_new = np.delete(X, feature_id, axis=1)
    splits = {}
    for value in unique_values:
        X_cur = X_new[np.where(X[:, feature_id] == value)]
        y_cur = None
        if(train):
            y_cur = y[np.where(X[:, feature_id] == value)]

        ids_cur = ids[np.where(X[:, feature_id] == value)]
        splits[value] = (X_cur, y_cur, ids_cur)
        
    return splits