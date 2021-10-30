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
        if(len(unique_values) <=threshold):
            remove_features.append(i)

    X_cleaned = np.delete(X, remove_features, axis=1)
    return X_cleaned, remove_features



