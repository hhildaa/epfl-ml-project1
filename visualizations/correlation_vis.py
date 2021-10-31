# imports, and load data
import numpy as np
import matplotlib.pyplot as plt

# visualize correlated features

def correlated_pairs(data):
    """
    Create a list of pairs of correlated columns of the given data.

    Input:
    data: a matrix, where each column is a sample of data

    Output: 
    corr_pairs: the index pairs of the most correlated features (i.e. correlation>0.9)
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
    corr_pairs = correlated_pairs(data)
    fig, axs = plt.subplots(len(corr_pairs), figsize=(6,6*len(corr_pairs)))
    for i in range(len(corr_pairs)): 
        axs[i].scatter(data[:, corr_pairs[i][0]],data[:, corr_pairs[i][1]], s=0.1)
        axs[i].set_title(f'{features[corr_pairs[i][0]]} and {features[corr_pairs[i][1]]}')

    #fig.suptitle('Correlated features',  fontsize=20)
    fig.tight_layout()

# plot correlation heatmap

def correlation_matrix(data):
    """
    Make a matrix of the correlations of the columns of data.

    Input:
    data: a matrix, where each column is a sample of data
    features: an array of names of the columns

    Output: 
    corr_map: a matrix of correlation values
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
    fig, ax = plt.subplots(figsize=(max(0.5*col_num,10) , max(0.5*col_num,10)))
    im = ax.imshow(corr_map)

    ax.set_xticks(np.arange(len(features[2:])))
    ax.set_yticks(np.arange(len(features[2:])))

    ax.set_xticklabels(features[2:], fontsize=12)
    ax.set_yticklabels(features[2:], fontsize=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(col_num):
        for j in range(col_num):
            text = ax.text(j, i, round(corr_map[i, j], 2),
                        ha="center", va="center", color="w", fontsize=12)

    ax.set_title("Heatmap of correlation between features", fontsize=20)
    fig.tight_layout()
    plt.show()
