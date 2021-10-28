# imports, and load data

import numpy as np
import matplotlib.pyplot as plt

DATA_TRAIN_PATH = 'data/train.csv'
y_train, X_train, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)
features = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",",names=True).dtype.names

# visulaize correlated features

def correlated_pairs(data):
    """
    Create a list of pairs of correlated columns of the given data.

    Input:
        data: a matrix, where each column is a sample of data
    """
    col_num=data.shape[1]
    corr_pairs = []
    for i in range(col_num):
        for j in range(i+1, col_num):
            corr = np.corrcoef(data[:, i],data[:, j])
            if abs(corr[0,1])>0.9: 
                corr_pairs.append((i,j))
    return corr_pairs

def correlation_plot(data, features):
    """
    Plot correlated columns of data in scatter plots.

    Input:
        data: a matrix, where each column is a sample of data
        features: an array of names of the columns
    """
    #NEEDED DATA CLEANING
    corr_pairs = correlated_pairs(data)
    fig, axs = plt.subplots(len(corr_pairs), figsize=(6,6*len(corr_pairs)))
    for i in range(len(corr_pairs)): 
        axs[i].scatter(data[:, corr_pairs[i][0]],data[:, corr_pairs[i][1]], s=0.1)
        axs[i].set_title(f'{features[corr_pairs[i][0]]} and {features[corr_pairs[i][1]]}')

    #fig.suptitle('Correlated features',  fontsize=20)
    fig.tight_layout()

#correlation_plot(X_train, features)

# plot correlation heatmap

def correlation_matrix(data):
    """
    Make a matrix of the correlations of the columns of data.

    Input:
        data: a matrix, where each column is a sample of data
        features: an array of names of the columns
    """
    col_num=data.shape[1]
    corr_map = np.zeros((col_num, col_num))
    for i in range(col_num):
        for j in range(col_num):
            corr = np.corrcoef(data[:,i], data[:, j])
            corr_map[i,j] = corr[0,1]
    return corr_map

def correlation_heatmap(data, features):
    """
    Visualize correlation of the columns of a dataset as a heatmap.

    Input:
        data: a matrix, where each column is a sample of data
        features: an array of names of the columns
    """

    corr_map = correlation_matrix(data)
    col_num=data.shape[1]
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(corr_map)

    ax.set_xticks(np.arange(col_num))
    ax.set_yticks(np.arange(col_num))

    ax.set_xticklabels(features[2:])
    ax.set_yticklabels(features[2:])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(col_num):
        for j in range(col_num):
            text = ax.text(j, i, round(corr_map[i, j], 2),
                        ha="center", va="center", color="w")

    ax.set_title("Heatmap of correlation between features", fontsize=20)
    fig.tight_layout()
    plt.show()

#correlation_heatmap(X_train,features)