import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from utils import *


def perform_pca(real, fake):
    """ Perform a principal component analysis (PCA) on the data and visualize on a 2D plane

    Args:
        real (torch.Tensor): the real data for pca
        fake (torch.Tensor): the generated data for pca

    Returns:
        None
    """
    labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))

    data = torch.cat((real, fake), 0)

    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # PCA projection to 2D
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # visualize the 2D
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    scatter = plt.scatter(df['PC1'], df['PC2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Real and Fake Data")
    plt.show()

    print("Explained variance ratio: " + str(pca.explained_variance_ratio_))
    print("Singular values: " + str(pca.singular_values_))

def perform_pca_har_classes(data, labels, classes):
    """ Perform a principal component analysis (PCA) on the data and visualize on a 2D plane

    Args:
        data (torch.Tensor): the data with all classes for pca
        labels (torch.Tensor): the class labels for the data
        classes (list<strings>): the names of the classes labels

    Returns:
        None
    """

    pca = PCA(n_components=2)
    components = pca.fit_transform(data.detach().numpy())

    # PCA projection to 2D
    df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])

    # visualize the 2D
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    scatter = plt.scatter(df['PC1'], df['PC2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA with Real and Fake Data")
    plt.show()

    print("Explained variance ratio: " + str(pca.explained_variance_ratio_))
    print("Singular values: " + str(pca.singular_values_))

def graph_two_features(real, fake):
    """ Graphs real and fake data with only two features

    Args:
        real (torch.Tensor): the real data with two features
        fake (torch.Tensor): the generated data with two features

    Returns:
        None
    """
    labels = np.concatenate((np.ones(len(real)), np.zeros(len(fake))))

    data = torch.cat((real, fake), 0).detach().numpy()

    # PCA projection to 2D
    df = pd.DataFrame(data=data, columns=['X1', 'X2'])

    # visualize the 2D
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    scatter = plt.scatter(df['X1'], df['X2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Real and Fake Data")
    plt.show()

def make_histograms(data, num_features):
    """ Make a histogram for every feature in the provided data set and save in a folder

    Args:
        data (torch.Tensor): the data to visualize with histograms

    Returns:
        None
    """
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    for col in range(num_features):
        hist = torch.histc(data[0:, col], min=-1, max=1, bins=10, out=None)

        x = np.linspace(-1, 1, 10)
        plt.cla()
        plt.bar(x, hist, align='center', color=colors[col%6])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        # plt.savefig(f'./histograms/fake/{col}.png')
