import os
os.sys.path.append('./scripts')

import numpy as np

def remove_custom_features(X, custom_feature_ids):
    """
    Remove columns with given features from data

    Input:
    X -- data
    custom_feature_ids -- list with ids of custom feature columns

    returns:
    X_cleaned -- data with custom feature columns removed
    """
    X_cleaned = np.delete(X, custom_feature_ids, axis=1)
    return X_cleaned

def remove_tiny_features(X, threshold=1):
    """
    Remove columns with low number of unique values

    Input:
    X -- data
    threshold -- max number of unique values to remove column

    returns:
    X_cleaned -- data with custom feature columns removed
    remove_features -- ids of columns that were removed
    """
    d = X.shape[1]
    remove_features = []
    
    for i in range(d):
        unique_values = np.unique(X[:, i])
        if(len(unique_values) <= threshold):
            remove_features.append(i)

    X_cleaned = np.delete(X, remove_features, axis=1)
    return X_cleaned, remove_features

def remove_outliers(X, y, custom_range=None):
    """
    Remove outliers, i.e. rows with any value that is farther than 1.5 IQR from upper or lower quartile, resp. 
    
    Input:
    X -- data without prediction variable
    y -- prediction variable
    custom_range -- list of (min, max) pairs to define custom ranges in which data should fall
                    for no change, tuples can be None or custom_range can be None
    
    returns:
    X -- data with rows containing an outlier removed
    y -- prediction variable for X
    """
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

    Input:
    X -- data without prediction variable
    upper_quart -- list of upper quartiles for each column of X (None to use upper quartiles from X)
    lower_quart -- list of lower quartiles for each column of X (None to use lower quartiles from X)
    
    returns:
    X -- data with outliers bounded
    upper_quart -- upper_quart input or list of upper quartiles for each column of X if upper_quart was None
    lower_quart -- lower_quart input or list of lower quartiles for each column of X if lower_quart was None
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
    Replace missing values in data by median

    Input:
    X -- data
    train_medians -- list of medians for each column of X (None to use medians from X)
    nan_val -- value to regard as missing (NaN)
    
    Returns: 
    X -- data (median-imputed)
    train_medians -- train_medians input or medians from each column of X if train_medians was None
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
