import os
os.sys.path.append('./scripts')

import numpy as np

def remove_custom_features(X, custom_feature_ids):
    X_cleaned = np.delete(X, custom_feature_ids, axis=1)
    return X_cleaned

def remove_tiny_features(X, threshold=1):
    d = X.shape[1]
    remove_features = []
    
    for i in range(d):
        unique_values = np.unique(X[:, i])
        if(len(unique_values) <= threshold):
            remove_features.append(i)

    X_cleaned = np.delete(X, remove_features, axis=1)
    return X_cleaned, remove_features

def remove_outliers(X, y, custom_range=None):
    """Remove outliers, i.e. rows with values that are farther than 1.5 IQR from mean"""
    custom_keep = np.copy(X)
    if custom_range is not None:
        if len(custom_range) != X.shape[1]:
            raise ValueError("custom_range has wrong length!")
        for i in range(len(custom_range)):
            if custom_range[i] is not None:
                lower, upper = custom_range[i]
                custom_keep[:,i] = np.logical_and(custom_keep[:,i] >= lower, custom_keep[:,i] <= upper)
            else:
                custom_keep[:, i] = True
    else:
        custom_keep[:,:] = True
    upper_quart = np.quantile(X, .75, axis=0)
    lower_quart = np.quantile(X, .25, axis=0)
    IQR = upper_quart - lower_quart
    
    lower_bound = lower_quart - 1.5*IQR
    upper_bound = upper_quart + 1.5*IQR
    
    lower_bound = np.reshape(lower_bound, (1, len(lower_bound)))
    upper_bound = np.reshape(upper_bound, (1, len(upper_bound)))
    
    to_keep = np.all(np.logical_and(custom_keep, X >= lower_bound, X <= upper_bound), axis=1)

    X = X[to_keep, :]
    y = y[to_keep]
    
    return X, y

def bound_outliers(X, upper_quart=None, lower_quart=None):
    """
    Bound outliers, i.e. rows with values that are farther than 1.5 IQR
    from Lower/Upper quartile, to this quartile
    """
    if upper_quart is None:
        upper_quart = np.quantile(X, .75, axis=0)
    if lower_quart is None:
        lower_quart = np.quantile(X, .25, axis=0)
    IQR = upper_quart - lower_quart
    
    lower_bound = lower_quart - 1.5*IQR
    upper_bound = upper_quart + 1.5*IQR
    
    for i in range(X.shape[1]):
        too_low = (X[:,i] < lower_bound[i])
        X[too_low,i] = lower_bound[i]
        
        too_high = (X[:,i] > upper_bound[i])
        X[too_high,i] = upper_bound[i]
    
    return X, upper_quart, lower_quart

def impute_median(X, train_medians=None, nan_val=-999):
    """
        Replace nan_val by median in X
        Custom medians to use same median for training and testing can be 
        put into train_medians (otherwise use col medians of X)
        
        Returns X (median-imputed) and train_medians
    """
    if train_medians is None:
        train_medians = list()
        for i in range(X.shape[1]):
            missing_vals = (X[:,i] != nan_val)
            train_medians.append(np.median(X[missing_vals,i]))
    for i in range(X.shape[1]):
        missing_vals = (X[:,i] == nan_val)
        X[missing_vals,i] = train_medians[i] 
    return X, train_medians
