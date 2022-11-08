import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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