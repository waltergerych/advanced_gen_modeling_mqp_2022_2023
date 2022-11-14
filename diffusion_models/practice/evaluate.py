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
    scatter = plt.scatter(df['PC1'], df['PC2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
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
    scatter = plt.scatter(df['X1'], df['X2'], c=labels, alpha=.8, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Fake', 'Real'])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Real and Fake Data")
    plt.show()
