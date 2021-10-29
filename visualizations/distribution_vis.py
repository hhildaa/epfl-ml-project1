# imports and load data

import numpy as np
import matplotlib.pyplot as plt

DATA_TRAIN_PATH = 'data/train.csv'
y_train, X_train, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)
features = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",",names=True).dtype.names

#plot the distributions of each features

def distibutions_plot(data, features):
    """
    This function visualize with histograms the distribution of each columns of the given data.

    Inputs: 
    data: a matrix, where each column is a sample of data from a distribution
    features: expect an array of names of the columns
    """
    #NEEDED DATA CLEANING
    col_num=data.shape[1]

    fig, axs = plt.subplots(col_num, figsize=(3,3*col_num))
    for i in range(col_num):
        axs[i].hist(data[:,i], bins=100) 
        axs[i].set_title(f'{features[i]}')

    #fig.suptitle('Distributions of the values of different features', fontsize=20)
    fig.tight_layout()

def distributions_compare_plot(data1, data2, features):
    """
    This function visualize with histograms the distribution of each columns of the given datas. Each plots visualize two datasets from the same feature

    Inputs: 
    data1: a matrix, where each column is a sample of data from a same distribution
    data2: a matrix with same number of columns, where each column is a sample from the same feature as the data1
    features: expect an array of names of the columns
    """
    #NEEDED DATA CLEANING
    col_num=data1.shape[1]

    fig, axs = plt.subplots(col_num, figsize=(3,3*col_num))
    for i in range(col_num):
        axs[i].hist(data1[:,i], bins=100) 
        axs[i].hist(data2[:,i], bins=100)
        axs[i].set_title(f'{features[i]}')

    #fig.suptitle('Distributions of the values of different features', fontsize=20)
    fig.tight_layout()